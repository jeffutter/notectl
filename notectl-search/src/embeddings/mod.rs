//! Dense embedding support via an OpenAI-compatible `/v1/embeddings` HTTP endpoint.
//!
//! No model runs in-process — `notectl` never loads or downloads anything
//! itself. Callers point `embedding_api_base` at any server that speaks the
//! OpenAI embeddings API (llama.cpp/llama-swap, vLLM, Ollama, OpenAI itself,
//! etc.) and this module just POSTs to it.
//!
//! Handles:
//! - Query/document prefix injection (`"query: "`, `"passage: "`)
//! - L2 normalization and Matryoshka dimension truncation
//! - Graceful degradation when the endpoint is unavailable or unconfigured

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
    /// Model identifier sent as `"model"` in the request body (e.g. "qwen3-embedding:0.6b").
    pub model_id: String,
    /// Output dimension ceiling (for Matryoshka truncation). Values at or
    /// above a model's native output length are a no-op — truncate() only
    /// shrinks, never pads.
    pub embedding_dim: usize,
    /// Base URL for an OpenAI-compatible embeddings endpoint (e.g.
    /// "https://host/v1"). `/embeddings` is appended to build the request
    /// URL. `None` means dense embeddings are unavailable.
    pub api_base: Option<String>,
    /// Optional bearer token for the embedding API.
    pub api_key: Option<String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            embedding_dim: 4096,
            api_base: None,
            api_key: None,
        }
    }
}

impl EmbeddingConfig {
    /// Build from a [`SearchConfig`]. Returns `None` when no embedding API
    /// base URL is configured — dense embeddings are simply unavailable,
    /// not an error. Callers pass `None` on to `IndexBuilder`/search's
    /// existing `Option<&mut Embedder>` handling, which already treats "no
    /// embedder" as "BM25 keyword search only."
    pub fn from_search_config(sc: &SearchConfig) -> Option<Self> {
        let api_base = sc.embedding_api_base.clone()?;
        Some(Self {
            model_id: sc.model_id.clone(),
            embedding_dim: sc.embedding_dim as usize,
            api_base: Some(api_base),
            api_key: sc.embedding_api_key.clone(),
        })
    }
}

/// Error type for embedding operations.
#[derive(Debug)]
pub enum EmbedError {
    /// The embedder isn't configured with an API base URL.
    Init(String),
    /// The HTTP request failed, or the server returned an error/unparseable response.
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

/// Request body for `POST {api_base}/embeddings`.
#[derive(serde::Serialize)]
struct EmbeddingsRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

/// One entry in the OpenAI-compatible embeddings response.
#[derive(serde::Deserialize)]
struct EmbeddingObject {
    embedding: Vec<f32>,
    index: usize,
}

/// Response body from `POST {api_base}/embeddings`.
#[derive(serde::Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingObject>,
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

/// Dense embedder backed by an OpenAI-compatible HTTP endpoint.
pub struct Embedder {
    client: reqwest::Client,
    config: EmbeddingConfig,
}

impl Embedder {
    /// Create a new embedder for the given config.
    ///
    /// Generous timeout: servers like llama-swap commonly unload idle models
    /// to free VRAM, so the first request after a while can trigger a cold
    /// model load (which can take well over a minute) before any actual
    /// embedding work starts. Once warm, real requests are much faster than
    /// this ceiling.
    pub fn new(config: EmbeddingConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .unwrap_or_default();
        Self { client, config }
    }

    /// Return the configured model ID.
    pub fn model_id(&self) -> &str {
        &self.config.model_id
    }

    /// POST a batch of already-prefixed texts and return their embeddings,
    /// in input order.
    async fn embed_raw(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let api_base =
            self.config.api_base.as_deref().ok_or_else(|| {
                EmbedError::Init("no embedding API base URL configured".to_string())
            })?;
        let url = format!("{}/embeddings", api_base.trim_end_matches('/'));

        let mut request = self.client.post(&url).json(&EmbeddingsRequest {
            model: &self.config.model_id,
            input: inputs,
        });
        if let Some(key) = &self.config.api_key {
            request = request.bearer_auth(key);
        }

        let response = request
            .send()
            .await
            .map_err(|e| EmbedError::Embed(format!("Failed to reach {url}: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(EmbedError::Embed(format!(
                "{url} returned {status}: {body}"
            )));
        }

        let mut parsed: EmbeddingsResponse = response
            .json()
            .await
            .map_err(|e| EmbedError::Embed(format!("Failed to parse response from {url}: {e}")))?;

        parsed.data.sort_by_key(|d| d.index);
        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }

    /// Generate an embedding for a single text.
    pub async fn embed_single(
        &mut self,
        text: &str,
        title: Option<&str>,
        task: TaskType,
    ) -> Result<Vec<f32>, EmbedError> {
        let input = apply_prefix(text, title, task);
        let embeddings = self.embed_raw(&[input]).await?;

        let embedding = embeddings.into_iter().next().unwrap_or_default();
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
        // Delegate to batched version with a single batch
        self.embed_batch_in_batches(texts, titles, task, texts.len())
            .await
    }

    /// Generate embeddings in smaller batches — `batch_size` controls how
    /// many texts go into each HTTP request, letting callers keep request
    /// payloads (and thus memory) bounded regardless of total input size.
    pub async fn embed_batch_in_batches(
        &mut self,
        texts: &[String],
        titles: &[Option<String>],
        task: TaskType,
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut result = Vec::with_capacity(texts.len());
        let effective_batch = batch_size.max(1);

        for chunk in texts.chunks(effective_batch) {
            let start = result.len();
            let chunk_titles = &titles[start..start + chunk.len()];

            let inputs: Vec<String> = chunk
                .iter()
                .zip(chunk_titles.iter())
                .map(|(text, title)| apply_prefix(text, title.as_deref(), task))
                .collect();

            let embeddings = self.embed_raw(&inputs).await?;

            for emb in embeddings {
                let truncated = truncate(emb, self.config.embedding_dim);
                result.push(normalize(&truncated));
            }
        }

        Ok(result)
    }
}

impl std::fmt::Display for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Embedder(model={}, dim={}, api_base={})",
            self.config.model_id,
            self.config.embedding_dim,
            self.config.api_base.as_deref().unwrap_or("<none>"),
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
        assert_eq!(config.model_id, "");
        assert_eq!(config.embedding_dim, 4096);
        assert!(config.api_base.is_none());
        assert!(config.api_key.is_none());
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
    fn test_embedder_creation_and_display() {
        let embedder = Embedder::new(EmbeddingConfig::default());
        let s = format!("{embedder}");
        assert!(s.contains("api_base=<none>"));
    }

    #[test]
    fn test_embedding_config_from_search_config_none_without_api_base() {
        use notectl_core::config::Config;
        let config = Config::default();
        assert!(config.search.embedding_api_base.is_none());
        assert!(EmbeddingConfig::from_search_config(&config.search).is_none());
    }

    #[test]
    fn test_embedding_config_from_search_config_some_with_api_base() {
        use notectl_core::config::{Config, SearchConfig};
        let mut config = Config::default();
        config.search = SearchConfig {
            embedding_api_base: Some("https://example.com/v1".to_string()),
            ..config.search
        };

        let emb_config = EmbeddingConfig::from_search_config(&config.search).unwrap();
        assert_eq!(emb_config.model_id, config.search.model_id);
        assert_eq!(
            emb_config.embedding_dim,
            config.search.embedding_dim as usize
        );
        assert_eq!(
            emb_config.api_base.as_deref(),
            Some("https://example.com/v1")
        );
    }

    #[tokio::test]
    async fn test_embed_single_without_api_base_returns_init_error() {
        let mut embedder = Embedder::new(EmbeddingConfig::default());
        let result = embedder
            .embed_single("hello", None, TaskType::RetrievalQuery)
            .await;
        assert!(matches!(result, Err(EmbedError::Init(_))));
    }
}
