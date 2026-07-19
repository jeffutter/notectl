//! Embedder facade: batch embedding with query/document prefix injection.
//!
//! Provides the high-level API for:
//! - Embedding single texts or batches
//! - Applying task-specific prompt prefixes (query vs document)
//! - Matryoshka truncation + L2 normalization
//! - Integration with the search pipeline
//!
//! CPU inference is wrapped in `tokio::task::spawn_blocking` so the shared tokio
//! runtime used by the HTTP/MCP server is never stalled by heavy tensor computation.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Tensor};
use tokenizers::Tokenizer;

use super::download::{self, DownloadError};
use super::model::{
    EmbeddingModelConfig, LoadedModel, ModelLoadError, load_model, normalize_embedding,
    truncate_and_pad,
};

/// Task type for prompt prefix injection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    /// Query for retrieval: "task: search result | query: {content}"
    RetrievalQuery,
    /// Document for retrieval: "title: {title} | text: {content}"
    RetrievalDocument,
}

impl TaskType {
    /// Apply the appropriate prompt prefix to the content.
    pub fn apply_prefix(&self, content: &str, title: Option<&str>) -> String {
        match self {
            TaskType::RetrievalQuery => {
                format!("task: search result | query: {content}")
            }
            TaskType::RetrievalDocument => {
                let title = title.unwrap_or("none");
                format!("title: {title} | text: {content}")
            }
        }
    }
}

/// Configuration for the embedder
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Target output dimension (supports MRL: 768, 512, 256, 128)
    pub output_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Data type for inference
    pub dtype: DType,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            output_dim: 768,
            max_seq_len: 2048,
            dtype: DType::F32,
        }
    }
}

impl EmbeddingConfig {
    /// Build an EmbeddingConfig from the authoritative SearchConfig.
    pub fn from_search_config(sc: &notectl_core::config::SearchConfig) -> Self {
        Self {
            output_dim: sc.embedding_dim as usize,
            max_seq_len: sc.max_seq_tokens,
            dtype: DType::F32,
        }
    }
}

/// Error type for embedding operations
#[derive(Debug)]
pub enum EmbedError {
    /// Model not downloaded or corrupted
    ModelNotFound(PathBuf),
    /// Failed to download model
    Download(DownloadError),
    /// Failed to load model
    Load(ModelLoadError),
    /// Candle inference error
    Inference(String),
    /// Tokenization error
    Tokenization(String),
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbedError::ModelNotFound(path) => {
                write!(
                    f,
                    "Model not found at: {}. Run index first to download weights.",
                    path.display()
                )
            }
            EmbedError::Download(e) => write!(f, "Download error: {e}"),
            EmbedError::Load(e) => write!(f, "Load error: {e}"),
            EmbedError::Inference(msg) => write!(f, "Inference error: {msg}"),
            EmbedError::Tokenization(msg) => write!(f, "Tokenization error: {msg}"),
        }
    }
}

impl std::error::Error for EmbedError {}

impl From<DownloadError> for EmbedError {
    fn from(e: DownloadError) -> Self {
        EmbedError::Download(e)
    }
}

impl From<ModelLoadError> for EmbedError {
    fn from(e: ModelLoadError) -> Self {
        EmbedError::Load(e)
    }
}

impl From<candle_core::Error> for EmbedError {
    fn from(e: candle_core::Error) -> Self {
        EmbedError::Inference(format!("Candle error: {e}"))
    }
}

/// The main embedder that handles batching and prefix injection.
///
/// The model is wrapped in `Arc<Mutex<...>>` to allow safe concurrent access
/// from multiple blocking threads (via spawn_blocking) while keeping the
/// embedder usable from async contexts.
pub struct Embedder {
    /// Loaded model wrapped for interior mutability across thread boundaries.
    model: Option<Arc<Mutex<LoadedModel>>>,
    /// Tokenizer (immutable after loading, safe to share via Arc).
    tokenizer: Option<Tokenizer>,
    /// Device.
    device: Device,
    /// Embedding configuration.
    config: EmbeddingConfig,
    /// Model cache directory.
    cache_dir: PathBuf,
}

impl Embedder {
    /// Create a new embedder (model is loaded lazily on first embed call).
    pub fn new(cache_dir: PathBuf, config: EmbeddingConfig) -> Self {
        Self {
            model: None,
            tokenizer: None,
            device: Device::Cpu,
            config,
            cache_dir,
        }
    }

    /// Initialize the model and tokenizer (called automatically on first embed if needed).
    fn ensure_loaded(&mut self) -> Result<(), EmbedError> {
        if self.model.is_some() && self.tokenizer.is_some() {
            return Ok(());
        }

        // Check if model is downloaded.
        if !download::is_model_ready(&self.cache_dir) {
            return Err(EmbedError::ModelNotFound(self.cache_dir.clone()));
        }

        // Load model.
        let embedding_config = EmbeddingModelConfig {
            output_dim: self.config.output_dim,
            max_seq_len: self.config.max_seq_len,
            dtype: self.config.dtype,
        };

        let loaded = load_model(&self.cache_dir, &self.device, &embedding_config)?;
        self.model = Some(Arc::new(Mutex::new(loaded)));

        // Load tokenizer.
        let tokenizer_path = self.cache_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedError::Tokenization(format!("Failed to load tokenizer: {e}")))?;
        self.tokenizer = Some(tokenizer);

        Ok(())
    }

    /// Async entry point: embed a single text with the specified task type.
    ///
    /// Runs tokenization + inference on a blocking thread pool via
    /// `tokio::task::spawn_blocking` so the shared tokio runtime is never stalled.
    pub async fn embed_single(
        &mut self,
        text: &str,
        title: Option<&str>,
        task: TaskType,
    ) -> Result<Vec<f32>, EmbedError> {
        self.ensure_loaded()?;

        let prefixed = task.apply_prefix(text, title);
        let output_dim = self.config.output_dim;
        let model_arc = self
            .model
            .as_ref()
            .ok_or_else(|| EmbedError::Inference("Model not loaded".into()))?
            .clone();
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| EmbedError::Inference("Tokenizer not loaded".into()))?
            .clone();

        tokio::task::spawn_blocking(move || {
            let mut model = model_arc
                .lock()
                .map_err(|e| EmbedError::Inference(format!("Model lock poisoned: {e}")))?;
            inner_embed_text(&mut model, &tokenizer, &prefixed, output_dim)
        })
        .await
        .map_err(|e| EmbedError::Inference(format!("spawn_blocking panicked: {e}")))?
    }

    /// Async entry point: embed a batch of texts with the specified task type.
    ///
    /// Each text is embedded via its own `spawn_blocking` call, awaited to completion
    /// before the next begins — processing is fully sequential. Returns a vector of
    /// embedding vectors in the same order as the input.
    pub async fn embed_batch(
        &mut self,
        texts: &[String],
        titles: &[Option<String>],
        task: TaskType,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.len() != titles.len() {
            return Err(EmbedError::Inference(
                "texts and titles must have the same length".to_string(),
            ));
        }

        let mut results = Vec::with_capacity(texts.len());
        for (text, title) in texts.iter().zip(titles.iter()) {
            let embedding = self.embed_single(text, title.as_deref(), task).await?;
            results.push(embedding);
        }

        Ok(results)
    }

    /// Get or initialize the model cache directory.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Check if the model is ready (downloaded and loadable).
    pub fn is_ready(&self) -> bool {
        download::is_model_ready(&self.cache_dir)
    }
}

/// Core inference logic: tokenize, run encoder forward, pool, project, normalize.
///
/// Takes the loaded model and tokenizer directly so it can be called from
/// `spawn_blocking` without capturing `&mut self`.
fn inner_embed_text(
    model: &mut LoadedModel,
    tokenizer: &Tokenizer,
    text: &str,
    output_dim: usize,
) -> Result<Vec<f32>, EmbedError> {
    // Tokenize.
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| EmbedError::Tokenization(format!("Tokenization failed: {e}")))?;

    let token_ids = encoding.get_ids();

    if token_ids.len() > model.embedding_config.max_seq_len {
        tracing::warn!(
            "Text too long: {} tokens > {} max, truncating",
            token_ids.len(),
            model.embedding_config.max_seq_len
        );
    }

    // Truncate and pad to max_seq_len using shared helper.
    let pad_id = model.pad_token_id;
    let padded_ids = truncate_and_pad(token_ids, model.embedding_config.max_seq_len, pad_id);

    // Attention mask: 1.0 for real tokens, 0.0 for padding.
    let attention_mask: Vec<f32> = padded_ids
        .iter()
        .map(|&id| if id == pad_id { 0.0 } else { 1.0 })
        .collect();

    // Create input tensor (batch size 1).
    let input_ids = Tensor::new(padded_ids.as_slice(), &model.device)
        .map_err(|e| EmbedError::Inference(format!("Failed to create input tensor: {e}")))?
        .unsqueeze(0)
        .map_err(|e| EmbedError::Inference(format!("Failed to unsqueeze: {e}")))?;

    let pad_tensor = Tensor::new(attention_mask.as_slice(), &model.device)
        .map_err(|e| EmbedError::Inference(format!("Failed to create mask tensor: {e}")))?
        .unsqueeze(0)
        .map_err(|e| EmbedError::Inference(format!("Failed to unsqueeze mask: {e}")))?;

    // Run encoder forward pass — returns full hidden states [1, seq_len, hidden].
    let hidden_states = model
        .model
        .forward(&input_ids, Some(&pad_tensor))
        .map_err(|e| EmbedError::Inference(format!("Encoder forward failed: {e}")))?;

    // Apply mean pooling over the sequence dimension, using pad_tensor so that
    // padding positions (0.0) are excluded from the average.
    let pooled = super::model::mean_pooling(&hidden_states, &pad_tensor)
        .map_err(|e| EmbedError::Inference(format!("Mean pooling failed: {e}")))?;

    // Apply Dense projection head (2_Dense tanh → 3_Dense linear).
    let projected = model
        .projection_head
        .forward(&pooled)
        .map_err(|e| EmbedError::Inference(format!("Projection failed: {e}")))?;

    // Extract the embedding vector (batch size 1, so squeeze dim 0).
    let embedding = projected
        .squeeze(0)
        .map_err(|e| EmbedError::Inference(format!("Failed to squeeze: {e}")))?;

    // Convert to f32 vec.
    let embedding_f32 = embedding
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()
        .map_err(|e| EmbedError::Inference(format!("Failed to extract vector: {e}")))?;

    // Apply matryoshka truncation + L2 normalization.
    Ok(normalize_embedding(&embedding_f32, output_dim))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_prefix_query() {
        let task = TaskType::RetrievalQuery;
        let prefixed = task.apply_prefix("test content", None);
        assert_eq!(prefixed, "task: search result | query: test content");
    }

    #[test]
    fn test_task_prefix_document_with_title() {
        let task = TaskType::RetrievalDocument;
        let prefixed = task.apply_prefix("test content", Some("My Title"));
        assert_eq!(prefixed, "title: My Title | text: test content");
    }

    #[test]
    fn test_task_prefix_document_without_title() {
        let task = TaskType::RetrievalDocument;
        let prefixed = task.apply_prefix("test content", None);
        assert_eq!(prefixed, "title: none | text: test content");
    }

    #[test]
    fn test_default_embedding_config() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.output_dim, 768);
        assert_eq!(config.max_seq_len, 2048);
    }

    #[test]
    fn test_embedder_creation() {
        let embedder = Embedder::new(
            PathBuf::from("/tmp/test-models"),
            EmbeddingConfig::default(),
        );
        assert_eq!(embedder.cache_dir(), Path::new("/tmp/test-models"));
        assert!(!embedder.is_ready()); // Model not downloaded yet
    }

    #[test]
    fn test_normalize_embedding_unit_vector() {
        let vec = vec![0.0, 1.0, 0.0, 0.0];
        let result = normalize_embedding(&vec, 4);
        assert_eq!(result, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_embedding_zero_vector() {
        let vec = vec![0.0, 0.0, 0.0];
        let result = normalize_embedding(&vec, 3);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_embedding_config_from_search_config() {
        use notectl_core::config::SearchConfig;

        let sc = SearchConfig {
            embedding_dim: 512,
            max_seq_tokens: 1024,
            ..Default::default()
        };

        let ec = EmbeddingConfig::from_search_config(&sc);
        assert_eq!(ec.output_dim, 512);
        assert_eq!(ec.max_seq_len, 1024);
        assert_eq!(ec.dtype, DType::F32);
    }

    #[tokio::test]
    async fn test_embed_batch_length_mismatch() {
        let mut embedder = Embedder::new(
            PathBuf::from("/tmp/test-models"),
            EmbeddingConfig::default(),
        );

        // This should fail before even trying to load the model.
        let result = embedder
            .embed_batch(
                &["text1".to_string(), "text2".to_string()],
                &[Some("title1".to_string())], // Mismatched length
                TaskType::RetrievalDocument,
            )
            .await;

        assert!(result.is_err());
    }

    /// Test helper: simulate embed_batch's text/title pairing and return prefixed strings.
    /// This lets us verify title-to-text pairing without loading the model.
    fn embed_batch_prefixed(
        texts: &[String],
        titles: &[Option<String>],
        task: TaskType,
    ) -> Vec<String> {
        assert_eq!(
            texts.len(),
            titles.len(),
            "texts and titles must have the same length"
        );

        texts
            .iter()
            .zip(titles.iter())
            .map(|(text, title)| task.apply_prefix(text, title.as_deref()))
            .collect()
    }

    #[test]
    fn test_embed_batch_title_pairing() {
        // Verify each text pairs with its own corresponding title via flat zip loop.
        let texts = vec![
            "text0".into(),
            "text1".into(),
            "text2".into(),
            "text3".into(),
            "text4".into(),
        ];
        let titles = vec![
            Some("T0".into()),
            Some("T1".into()),
            Some("T2".into()),
            Some("T3".into()),
            Some("T4".into()),
        ];

        let prefixed = embed_batch_prefixed(&texts, &titles, TaskType::RetrievalDocument);

        assert_eq!(prefixed.len(), 5);
        assert_eq!(prefixed[0], "title: T0 | text: text0");
        assert_eq!(prefixed[1], "title: T1 | text: text1");
        assert_eq!(prefixed[2], "title: T2 | text: text2");
        assert_eq!(prefixed[3], "title: T3 | text: text3");
        assert_eq!(prefixed[4], "title: T4 | text: text4");
    }

    #[test]
    fn test_embed_batch_with_none_titles() {
        // Some titles are None; verify they get "none" fallback and pairing is correct.
        let texts = vec!["x0".into(), "x1".into(), "x2".into(), "x3".into()];
        let titles = vec![Some("X0".into()), None, Some("X2".into()), None];

        let prefixed = embed_batch_prefixed(&texts, &titles, TaskType::RetrievalDocument);

        assert_eq!(prefixed.len(), 4);
        assert_eq!(prefixed[0], "title: X0 | text: x0");
        assert_eq!(prefixed[1], "title: none | text: x1");
        assert_eq!(prefixed[2], "title: X2 | text: x2");
        assert_eq!(prefixed[3], "title: none | text: x3");
    }
}
