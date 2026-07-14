//! Embedder facade: batch embedding with query/document prefix injection.
//!
//! Provides the high-level API for:
//! - Embedding single texts or batches
//! - Applying task-specific prompt prefixes (query vs document)
//! - Matryoshka truncation + L2 normalization
//! - Integration with the search pipeline

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use tokenizers::Tokenizer;

use super::download::{self, DownloadError};
use super::model::{
    EmbeddingModelConfig, LoadedModel, ModelLoadError, load_model, normalize_embedding,
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
    /// Batch size for embedding (higher = faster but more memory)
    pub batch_size: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            output_dim: 768,
            max_seq_len: 2048,
            dtype: DType::F32,
            batch_size: 32,
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
    /// Batch size exceeded
    BatchExceeded(usize),
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
            EmbedError::BatchExceeded(size) => {
                write!(f, "Batch size {size} exceeds maximum")
            }
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

/// The main embedder that handles batching and prefix injection
pub struct Embedder {
    /// Loaded model (lazy-loaded on first use)
    model: Option<LoadedModel>,
    /// Tokenizer
    tokenizer: Option<Tokenizer>,
    /// Device
    device: Device,
    /// Embedding configuration
    config: EmbeddingConfig,
    /// Model cache directory
    cache_dir: PathBuf,
}

impl Embedder {
    /// Create a new embedder (model is loaded lazily on first embed call)
    pub fn new(cache_dir: PathBuf, config: EmbeddingConfig) -> Self {
        Self {
            model: None,
            tokenizer: None,
            device: Device::Cpu,
            config,
            cache_dir,
        }
    }

    /// Initialize the model and tokenizer (called automatically on first embed if needed)
    fn ensure_loaded(&mut self) -> Result<(), EmbedError> {
        if self.model.is_some() && self.tokenizer.is_some() {
            return Ok(());
        }

        // Check if model is downloaded
        if !download::is_model_ready(&self.cache_dir) {
            return Err(EmbedError::ModelNotFound(self.cache_dir.clone()));
        }

        // Load model
        let embedding_config = EmbeddingModelConfig {
            output_dim: self.config.output_dim,
            max_seq_len: self.config.max_seq_len,
            dtype: self.config.dtype,
        };

        let loaded = load_model(&self.cache_dir, &self.device, &embedding_config)?;
        self.model = Some(loaded);

        // Load tokenizer
        let tokenizer_path = self.cache_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedError::Tokenization(format!("Failed to load tokenizer: {e}")))?;
        self.tokenizer = Some(tokenizer);

        Ok(())
    }

    /// Embed a single text with the specified task type.
    ///
    /// Applies the appropriate prompt prefix based on task type,
    /// tokenizes, runs inference, and returns a normalized embedding vector.
    pub fn embed_single(
        &mut self,
        text: &str,
        title: Option<&str>,
        task: TaskType,
    ) -> Result<Vec<f32>, EmbedError> {
        self.ensure_loaded()?;

        let prefixed = task.apply_prefix(text, title);
        self.embed_text(&prefixed)
    }

    /// Embed a batch of texts with the specified task type.
    ///
    /// Processes in batches to manage memory usage. Returns a vector of embedding vectors.
    pub fn embed_batch(
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

        for (batch_idx, chunk) in texts.chunks(self.config.batch_size).enumerate() {
            let start = batch_idx * self.config.batch_size;
            let end = start + chunk.len();
            let title_chunk: Vec<Option<&str>> =
                titles[start..end].iter().map(|t| t.as_deref()).collect();

            // This is a simplified batch implementation - in practice, you'd want to
            // pad and stack tensors for true batched inference
            for (i, text) in chunk.iter().enumerate() {
                let title = title_chunk.get(i).copied().flatten();
                let embedding = self.embed_single(text, title, task)?;
                results.push(embedding);
            }
        }

        Ok(results)
    }

    /// Internal: embed a single text (already prefixed) using the loaded model.
    fn embed_text(&mut self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| EmbedError::Inference("Model not loaded".to_string()))?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| EmbedError::Tokenization("Tokenizer not loaded".to_string()))?;

        // Tokenize
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|e| EmbedError::Tokenization(format!("Tokenization failed: {e}")))?;

        let token_ids = encoding.get_ids();
        if token_ids.len() > self.config.max_seq_len {
            tracing::warn!(
                "Text too long: {} tokens > {} max, truncating",
                token_ids.len(),
                self.config.max_seq_len
            );
        }

        // Truncate to max length
        let token_ids: Vec<u32> = token_ids
            .iter()
            .take(self.config.max_seq_len)
            .copied()
            .collect();

        // Create input tensor (batch size 1)
        let input_ids = Tensor::new(token_ids.as_slice(), &model.device)
            .map_err(|e| EmbedError::Inference(format!("Failed to create tensor: {e}")))?;
        let input_ids = input_ids
            .unsqueeze(0)
            .map_err(|e| EmbedError::Inference(format!("Failed to unsqueeze: {e}")))?;

        // Run forward pass (Gemma-3 handles causal masking internally)
        // Safety: Gemma3Model::forward only mutates internal state tracking (seqlen_offset);
        // the model weights and projection head are not modified during inference.
        let model_ref: &mut candle_transformers::models::gemma3::Model =
            unsafe { &mut *std::ptr::addr_of!(model.model).cast_mut() };
        let hidden_states = model_ref
            .forward(&input_ids, 0)
            .map_err(|e| EmbedError::Inference(format!("Forward pass failed: {e}")))?;

        // Create attention mask (all 1s for non-padded tokens)
        let attention_mask = Tensor::ones(input_ids.shape().clone(), DType::F32, &model.device)
            .map_err(|e| EmbedError::Inference(format!("Failed to create attention mask: {e}")))?;

        // Apply mean pooling
        let pooled = super::model::mean_pooling(&hidden_states, &attention_mask)
            .map_err(|e| EmbedError::Inference(format!("Mean pooling failed: {e}")))?;

        // Apply Dense projection head
        let projected = model
            .projection_head
            .forward(&pooled)
            .map_err(|e| EmbedError::Inference(format!("Projection failed: {e}")))?;

        // Extract the embedding vector (batch size 1, so squeeze dim 0)
        let embedding = projected
            .squeeze(0)
            .map_err(|e| EmbedError::Inference(format!("Failed to squeeze: {e}")))?;

        // Convert to f32 vec
        let embedding_f32 = embedding
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()
            .map_err(|e| EmbedError::Inference(format!("Failed to extract vector: {e}")))?;

        // Apply matryoshka truncation + L2 normalization
        Ok(normalize_embedding(&embedding_f32, self.config.output_dim))
    }

    /// Get or initialize the model cache directory
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Check if the model is ready (downloaded and loadable)
    pub fn is_ready(&self) -> bool {
        download::is_model_ready(&self.cache_dir)
    }
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
        assert_eq!(config.batch_size, 32);
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
    fn test_embed_batch_length_mismatch() {
        let mut embedder = Embedder::new(
            PathBuf::from("/tmp/test-models"),
            EmbeddingConfig::default(),
        );

        // This should fail before even trying to load the model
        let result = embedder.embed_batch(
            &["text1".to_string(), "text2".to_string()],
            &[Some("title1".to_string())], // Mismatched length
            TaskType::RetrievalDocument,
        );

        assert!(result.is_err());
    }
}
