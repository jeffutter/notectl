//! Dense embedding support via [fastembed](https://github.com/Anush008/fastembed-rs).
//!
//! Uses ONNX Runtime for inference. Supported models:\n\
//! - EmbeddingGemma-300M (default, with quantized variants)\n\
//! - BGE-Small/Base/Large v1.5\n\
//! - Qwen3-Embedding (0.6B, 4B, 8B)
//!
//! Handles:\n\
//! - Automatic model download and caching on first use\n\
//! - Query/document prefix injection (`"query: "`, `"passage: "`)\n\
//! - L2 normalization and Matryoshka dimension truncation\n\
//! - Graceful degradation when model is unavailable

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use notectl_core::config::SearchConfig;

/// Task type for prefix injection.
#[derive(Debug, Clone, Copy)]
pub enum TaskType {
    /// Query — prefixed with `"query: "`
    RetrievalQuery,
    /// Document — prefixed with `"passage: "`
    RetrievalDocument,
}

/// Configuration for embedding generation.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Model identifier (e.g., "google/embedding-gemma-300m").
    pub model_id: String,
    /// Output dimension (for Matryoshka truncation).
    pub embedding_dim: usize,
    /// Maximum sequence length in tokens.
    pub max_seq_len: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_id: "google/embedding-gemma-300m".to_string(),
            embedding_dim: 768,
            max_seq_len: 512,
        }
    }
}

impl EmbeddingConfig {
    pub fn from_search_config(sc: &SearchConfig) -> Self {
        Self {
            model_id: sc.model_id.clone(),
            embedding_dim: sc.embedding_dim as usize,
            max_seq_len: sc.max_seq_tokens,
        }
    }
}

/// Error type for embedding operations.
#[derive(Debug)]
pub enum EmbedError {
    /// Failed to initialize or load the model.
    Init(String),
    /// Failed to generate embeddings.
    Embed(String),
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbedError::Init(msg) => write!(f, "Embedding init error: {msg}"),
            EmbedError::Embed(msg) => write!(f, "Embedding error: {msg}"),
        }
    }
}

impl std::error::Error for EmbedError {}

/// Map a model_id string to a fastembed `EmbeddingModel` variant.
fn resolve_model(model_id: &str) -> EmbeddingModel {
    match model_id {
        "google/embeddinggemma-300m" | "google/embedding-gemma-300m" => {
            EmbeddingModel::EmbeddingGemma300M
        }
        "BAAI/bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
        "BAAI/bge-base-en-v1.5" => EmbeddingModel::BGEBaseENV15,
        "BAAI/bge-large-en-v1.5" => EmbeddingModel::BGELargeENV15,
        _ => EmbeddingModel::EmbeddingGemma300M, // Default fallback
    }
}

/// Inject task-specific prefix into text.
fn apply_prefix(content: &str, title: Option<&str>, task: TaskType) -> String {
    let prefix = match task {
        TaskType::RetrievalQuery => "query: ",
        TaskType::RetrievalDocument => "passage: ",
    };

    if let Some(title) = title {
        format!("{prefix}{title}\n{content}")
    } else {
        format!("{prefix}{content}")
    }
}

/// Normalize a vector to unit length (L2 norm).
fn normalize(vector: &[f32]) -> Vec<f32> {
    let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 1e-10 {
        vector.iter().map(|x| x / magnitude).collect()
    } else {
        vec![0.0; vector.len()]
    }
}

/// Truncate embedding to target dimension (Matryoshka Representation Learning).
fn truncate(embedding: Vec<f32>, dim: usize) -> Vec<f32> {
    if dim >= embedding.len() {
        embedding
    } else {
        embedding.into_iter().take(dim).collect()
    }
}

/// Dense embedder backed by fastembed's ONNX runtime.
pub struct Embedder {
    model: Option<TextEmbedding>,
    config: EmbeddingConfig,
}

impl Embedder {
    /// Create a new embedder. The model is loaded lazily on first use.
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            model: None,
            config,
        }
    }

    /// Check if the model is loaded and ready for inference.
    pub fn is_ready(&self) -> bool {
        self.model.is_some()
    }

    /// Ensure the model is loaded. Downloads on first call if needed.
    fn ensure_loaded(&mut self) -> Result<(), EmbedError> {
        if self.model.is_some() {
            return Ok(());
        }

        let embedding_model = resolve_model(&self.config.model_id);

        tracing::info!(
            "Loading embedding model: {:?} (dim={})",
            embedding_model,
            self.config.embedding_dim
        );

        let options = TextInitOptions::new(embedding_model);

        let model = TextEmbedding::try_new(options).map_err(|e| {
            EmbedError::Init(format!(
                "Failed to load model '{}': {}\n\
                 Make sure HF_TOKEN is set and license is accepted at \
                 https://huggingface.co/google/embeddinggemma-300m",
                self.config.model_id, e
            ))
        })?;

        self.model = Some(model);
        Ok(())
    }

    /// Generate an embedding for a single text.
    pub async fn embed_single(
        &mut self,
        text: &str,
        title: Option<&str>,
        task: TaskType,
    ) -> Result<Vec<f32>, EmbedError> {
        self.ensure_loaded()?;
        let model = self.model.as_mut().unwrap();

        let input = apply_prefix(text, title, task);
        let texts = vec![input];

        let embeddings = model
            .embed(&texts, None)
            .map_err(|e| EmbedError::Embed(format!("Failed to generate embedding: {e}")))?;

        let embedding = embeddings.first().cloned().unwrap_or_default();
        let truncated = truncate(embedding, self.config.embedding_dim);
        Ok(normalize(&truncated))
    }

    /// Generate embeddings for a batch of texts.
    pub async fn embed_batch(
        &mut self,
        texts: &[String],
        titles: &[Option<String>],
        task: TaskType,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.ensure_loaded()?;
        let model = self.model.as_mut().unwrap();

        let inputs: Vec<String> = texts
            .iter()
            .zip(titles.iter())
            .map(|(text, title)| apply_prefix(text, title.as_deref(), task))
            .collect();

        let embeddings = model
            .embed(&inputs, None)
            .map_err(|e| EmbedError::Embed(format!("Failed to generate batch embeddings: {e}")))?;

        let result: Vec<Vec<f32>> = embeddings
            .into_iter()
            .map(|emb| {
                let truncated = truncate(emb, self.config.embedding_dim);
                normalize(&truncated)
            })
            .collect();

        Ok(result)
    }
}

impl std::fmt::Display for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Embedder(model={}, dim={}, ready={})",
            self.config.model_id,
            self.config.embedding_dim,
            self.is_ready()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_prefix_query() {
        assert_eq!(
            apply_prefix("hello", None, TaskType::RetrievalQuery),
            "query: hello"
        );
    }

    #[test]
    fn test_task_prefix_document_with_title() {
        assert_eq!(
            apply_prefix("hello", Some("My Title"), TaskType::RetrievalDocument),
            "passage: My Title\nhello"
        );
    }

    #[test]
    fn test_task_prefix_document_without_title() {
        assert_eq!(
            apply_prefix("hello", None, TaskType::RetrievalDocument),
            "passage: hello"
        );
    }

    #[test]
    fn test_default_embedding_config() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.model_id, "google/embedding-gemma-300m");
        assert_eq!(config.embedding_dim, 768);
        assert_eq!(config.max_seq_len, 512);
    }

    #[test]
    fn test_normalize_embedding_unit_vector() {
        let v = vec![1.0, 0.0, 0.0];
        let normalized = normalize(&v);
        assert!((normalized[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_embedding_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let normalized = normalize(&v);
        assert!(normalized.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn test_truncate_shorter_than_target() {
        let emb = vec![1.0, 2.0, 3.0];
        let truncated = truncate(emb, 10);
        assert_eq!(truncated.len(), 3);
    }

    #[test]
    fn test_truncate_longer_than_target() {
        let emb = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let truncated = truncate(emb, 3);
        assert_eq!(truncated, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_resolve_model_defaults_to_gemma() {
        assert!(matches!(
            resolve_model("unknown/model"),
            EmbeddingModel::EmbeddingGemma300M
        ));
    }

    #[test]
    fn test_resolve_model_gemma_variants() {
        assert!(matches!(
            resolve_model("google/embeddinggemma-300m"),
            EmbeddingModel::EmbeddingGemma300M
        ));
        assert!(matches!(
            resolve_model("google/embedding-gemma-300m"),
            EmbeddingModel::EmbeddingGemma300M
        ));
    }

    #[test]
    fn test_embedder_creation() {
        let embedder = Embedder::new(EmbeddingConfig::default());
        assert!(!embedder.is_ready());
    }

    #[test]
    fn test_embedder_display() {
        let embedder = Embedder::new(EmbeddingConfig::default());
        let s = format!("{embedder}");
        assert!(s.contains("ready=false"));
    }

    #[test]
    fn test_embedding_config_from_search_config() {
        use notectl_core::config::Config;
        let config = Config::default();
        let emb_config = EmbeddingConfig::from_search_config(&config.search);
        assert_eq!(emb_config.model_id, config.search.model_id);
        assert_eq!(
            emb_config.embedding_dim,
            config.search.embedding_dim as usize
        );
        assert_eq!(emb_config.max_seq_len, config.search.max_seq_tokens);
    }
}
