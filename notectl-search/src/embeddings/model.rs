//! Gemma-3 model loading with mean pooling + Dense projection layers.
//!
//! Implements the embedding wrapper that:
//! - Loads candle Gemma-3 backbone for encoder-style attention
//! - Applies mean pooling to hidden states
//! - Loads and applies two sequential Dense projection layers

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::gemma3::{Config as Gemma3Config, Model as Gemma3Model};

use super::download::{self, DownloadError};

/// Configuration for the embedding model
#[derive(Debug, Clone)]
pub struct EmbeddingModelConfig {
    /// Output dimension (768 default, supports MRL: 512, 256, 128)
    pub output_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Data type for inference (f32 or bf16)
    pub dtype: DType,
}

impl Default for EmbeddingModelConfig {
    fn default() -> Self {
        Self {
            output_dim: 768,
            max_seq_len: 2048,
            dtype: DType::F32,
        }
    }
}

/// Mean pooling layer (just a utility function, no parameters)
fn mean_pooling(hidden_states: &Tensor, attention_mask: &Tensor) -> CandleResult<Tensor> {
    // attention_mask: (batch, seq_len) - 1 for real tokens, 0 for padding
    // hidden_states: (batch, seq_len, hidden_dim)

    let mask = attention_mask.to_dtype(hidden_states.dtype())?;
    let mask = mask.unsqueeze(2)?; // (batch, seq_len, 1)

    // Masked hidden states
    let masked = hidden_states.broadcast_mul(&mask)?;

    // Sum along sequence dimension
    let sum = masked.sum(1)?; // (batch, hidden_dim)

    // Count of real tokens per sequence
    let mask_sum = mask.sum(1)?.clip_min(1.0)?; // Avoid division by zero

    // Average
    sum.broadcast_div(&mask_sum)
}

/// Single Dense projection layer
#[derive(Debug)]
pub struct DenseLayer {
    linear: Linear,
    activation: Option<String>, // "tanh" or None for linear
}

impl DenseLayer {
    pub fn new(
        vb: VarBuilder,
        input_dim: usize,
        output_dim: usize,
        activation: Option<&str>,
    ) -> CandleResult<Self> {
        let linear = candle_nn::linear(input_dim, output_dim, vb)?;
        Ok(Self {
            linear,
            activation: activation.map(String::from),
        })
    }

    pub fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let xs = self.linear.forward(xs)?;

        match self.activation.as_deref() {
            Some("tanh") => xs.tanh(),
            _ => Ok(xs),
        }
    }
}

/// Two sequential Dense projection layers for EmbeddingGemma
#[derive(Debug)]
pub struct DenseProjectionHead {
    dense_2: DenseLayer, // 768 -> 3072 with tanh
    dense_3: DenseLayer, // 3072 -> 768 (or target dim) linear
}

impl DenseProjectionHead {
    pub fn new(vb: VarBuilder) -> CandleResult<Self> {
        let dense_2 = DenseLayer::new(vb.pp("2_Dense"), 768, 3072, Some("tanh"))?;
        let dense_3 = DenseLayer::new(vb.pp("3_Dense"), 3072, 768, None)?;
        Ok(Self { dense_2, dense_3 })
    }

    pub fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let xs = self.dense_2.forward(xs)?;
        self.dense_3.forward(&xs)
    }
}

/// Loaded embedding model with all components
pub struct LoadedModel {
    /// Gemma-3 backbone
    pub model: Gemma3Model,
    /// Config for the backbone
    pub config: Gemma3Config,
    /// Dense projection head (2 sequential Dense layers)
    pub projection_head: DenseProjectionHead,
    /// Embedding model configuration
    pub embedding_config: EmbeddingModelConfig,
    /// Device used for inference
    pub device: Device,
}

/// Error type for model loading
#[derive(Debug)]
pub enum ModelLoadError {
    Download(DownloadError),
    Candle(CandleError),
    Config(String),
    Io(std::io::Error),
}

impl std::fmt::Display for ModelLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelLoadError::Download(e) => write!(f, "Download error: {e}"),
            ModelLoadError::Candle(e) => write!(f, "Candle error: {e}"),
            ModelLoadError::Config(msg) => write!(f, "Config error: {msg}"),
            ModelLoadError::Io(err) => write!(f, "IO error: {err}"),
        }
    }
}

impl std::error::Error for ModelLoadError {}

impl From<DownloadError> for ModelLoadError {
    fn from(e: DownloadError) -> Self {
        ModelLoadError::Download(e)
    }
}

impl From<candle_core::Error> for ModelLoadError {
    fn from(e: candle_core::Error) -> Self {
        ModelLoadError::Candle(CandleError(e))
    }
}

impl From<std::io::Error> for ModelLoadError {
    fn from(e: std::io::Error) -> Self {
        ModelLoadError::Io(e)
    }
}

/// Wrapper around candle_core::Error for better error messages
#[derive(Debug)]
pub struct CandleError(candle_core::Error);

impl std::fmt::Display for CandleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for CandleError {}

/// Load the embedding model from the specified cache directory.
///
/// This loads:
/// 1. Gemma-3 backbone weights and config
/// 2. Pooling config (verifies mean pooling)
/// 3. Two Dense projection layers from sentence-transformers format
pub fn load_model(
    cache_dir: &Path,
    device: &Device,
    embedding_config: &EmbeddingModelConfig,
) -> Result<LoadedModel, ModelLoadError> {
    tracing::info!(
        "Loading EmbeddingGemma-300M model from {}",
        cache_dir.display()
    );

    // Load Gemma-3 config
    let config_path = cache_dir.join("config.json");
    let config_json = std::fs::read_to_string(&config_path).map_err(ModelLoadError::Io)?;
    let gemma_config: Gemma3Config = serde_json::from_str(&config_json)
        .map_err(|e| ModelLoadError::Config(format!("Failed to parse Gemma-3 config: {e}")))?;

    // Load backbone weights
    let weights_path = cache_dir.join("model.safetensors");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], embedding_config.dtype, device)?
    };

    let model = Gemma3Model::new(gemma_config.clone(), vb)?;
    tracing::info!("Loaded Gemma-3 backbone");

    // Load pooling config and verify mean pooling
    let pooling_config_path = cache_dir.join("1_Pooling/config.json");
    let pooling_json = std::fs::read_to_string(&pooling_config_path).map_err(ModelLoadError::Io)?;
    let pooling_config: serde_json::Value = serde_json::from_str(&pooling_json)
        .map_err(|e| ModelLoadError::Config(format!("Failed to parse pooling config: {e}")))?;

    let pooling_mode = pooling_config
        .get("pooling_mode")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            ModelLoadError::Config("Pooling config missing 'pooling_mode' field".to_string())
        })?;

    if pooling_mode != "mean" {
        return Err(ModelLoadError::Config(format!(
            "Expected mean pooling, got: {pooling_mode}"
        )));
    }
    tracing::info!("Verified mean pooling configuration");

    // Load Dense projection head
    let projection_vb = VarBuilder::from_mmaped_safetensors(
        &[
            cache_dir.join("2_Dense/model.safetensors"),
            cache_dir.join("3_Dense/model.safetensors"),
        ],
        DType::F32,
        device,
    )?;

    let projection_head = DenseProjectionHead::new(projection_vb)?;
    tracing::info!("Loaded Dense projection head (768 -> 3072 -> 768)");

    Ok(LoadedModel {
        model,
        config: gemma_config,
        projection_head,
        embedding_config: embedding_config.clone(),
        device: device.clone(),
    })
}

/// Matryoshka truncation: truncate vector to target dimension if smaller than original
fn matryoshka_truncate(vec: &[f32], target_dim: usize) -> Vec<f32> {
    if vec.len() <= target_dim {
        return vec.to_vec();
    }
    vec[..target_dim].to_vec()
}

/// L2 normalize a vector in-place
fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

/// Apply matryoshka truncation + L2 normalization to a single embedding vector
pub fn normalize_embedding(vec: &[f32], target_dim: usize) -> Vec<f32> {
    let mut result = matryoshka_truncate(vec, target_dim);
    l2_normalize(&mut result);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matryoshka_truncate_exact() {
        let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = matryoshka_truncate(&vec, 5);
        assert_eq!(result, vec);
    }

    #[test]
    fn test_matryoshka_truncate_smaller() {
        let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = matryoshka_truncate(&vec, 3);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_matryoshka_truncate_larger() {
        let vec = vec![1.0, 2.0];
        let result = matryoshka_truncate(&vec, 5);
        assert_eq!(result, vec);
    }

    #[test]
    fn test_l2_normalize_unit_vector() {
        let mut vec = vec![3.0, 4.0];
        l2_normalize(&mut vec);
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let mut vec = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut vec);
        assert_eq!(vec, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_embedding_full() {
        let vec = vec![3.0, 4.0];
        let result = normalize_embedding(&vec, 2);
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_embedding_with_truncation() {
        let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = normalize_embedding(&vec, 3);
        assert_eq!(result.len(), 3);
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_default_embedding_config() {
        let config = EmbeddingModelConfig::default();
        assert_eq!(config.output_dim, 768);
        assert_eq!(config.max_seq_len, 2048);
        assert_eq!(config.dtype, DType::F32);
    }
}
