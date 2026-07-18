//! Gemma-3 encoder model for embedding generation.
//!
//! Implements the embedding wrapper that:
//! - Loads candle Gemma-3 backbone configured for **bidirectional** (encoder-style) attention
//! - Applies mean pooling to full-sequence hidden states
//! - Loads and applies two sequential Dense projection layers from sentence-transformers format
//!
//! Key difference from the causal decoder variant: every token attends to every other token
//! (within sliding-window bands where configured), producing rich contextual representations
//! suitable for similarity search.

use std::sync::Arc;

use candle_core::{D, DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_b as linear};
use candle_transformers::models::gemma3::Config as Gemma3Config;

use std::path::Path;

use super::download::DownloadError;

/// Repeat KV heads `n_rep` times for Grouped Query Attention (GQA).
fn repeat_kv(xs: Tensor, n_rep: usize) -> CandleResult<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        // Using cat is faster than a broadcast as it avoids going through a potentially
        // strided copy.
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

// ---------------------------------------------------------------------------
// Reusable building blocks (mirrors candle-transformers internals that are
// private there). Kept module-private here.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> CandleResult<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&(&self.weight + 1.0)?)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Gemma3Config, dev: &Device) -> CandleResult<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> CandleResult<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)] // matches candle-transformers naming
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: candle_nn::Activation,
}

impl MLP {
    fn new(cfg: &Gemma3Config, vb: VarBuilder) -> CandleResult<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear(hidden_sz, intermediate_sz, false, vb.pp("gate_proj"))?;
        let up_proj = linear(hidden_sz, intermediate_sz, false, vb.pp("up_proj"))?;
        let down_proj = linear(intermediate_sz, hidden_sz, false, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_activation,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

// ---------------------------------------------------------------------------
// Encoder-specific components: bidirectional attention, no KV cache.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct EncoderAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    attn_logit_softcapping: Option<f64>,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl EncoderAttention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Gemma3Config,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = cfg.head_dim;
        let bias = cfg.attention_bias;
        let q_proj = linear(hidden_sz, num_heads * head_dim, bias, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("v_proj"))?;
        let o_proj = linear(num_heads * head_dim, hidden_sz, bias, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            attn_logit_softcapping: cfg.attn_logit_softcapping,
            rotary_emb,
        })
    }

    /// Bidirectional attention forward pass.
    ///
    /// Unlike the causal decoder variant, this does NOT use a KV cache and does
    /// NOT apply a causal mask. The `attention_mask` argument combines structural
    /// masking (sliding-window bands) with padding masking in NEG_INFINITY form.
    fn forward(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> CandleResult<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let query_states = self.q_norm.forward(&query_states)?;
        let key_states = self.k_norm.forward(&key_states)?;

        // No seqlen_offset for encoder — we always process the full sequence at once.
        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, 0)?;

        // No KV cache: use current key/value states directly.
        let key_states = repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

        // Logit softcapping (applied before masking so the cap operates on raw scores).
        let attn_weights = match self.attn_logit_softcapping {
            None => attn_weights,
            Some(sc) => ((attn_weights / sc)?.tanh()? * sc)?,
        };

        // Add structural + padding mask (NEG_INFINITY for masked positions).
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        attn_weights
            .matmul(&value_states)?
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }
}

#[derive(Debug, Clone)]
struct EncoderLayer {
    self_attn: EncoderAttention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl EncoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Gemma3Config,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let self_attn = EncoderAttention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let pre_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> CandleResult<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask)?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        residual + xs
    }
}

// ---------------------------------------------------------------------------
// Gemma3Encoder — the bidirectional backbone for embedding generation.
// ---------------------------------------------------------------------------

/// Bidirectional (encoder-style) Gemma-3 model for sentence embeddings.
///
/// Mirrors `candle_transformers::models::gemma3::Model` but with two critical
/// differences:
/// 1. **Bidirectional attention**: full attention on designated layers, centered
///    sliding-window bands on the rest — no causal masking.
/// 2. **Full hidden states output**: returns `[batch, seq_len, hidden]` instead
///    of last-token logits, enabling mean pooling for sentence embeddings.
///
/// No KV cache is used since embedding inference processes complete sequences
/// in a single forward pass.
#[derive(Debug)]
pub struct Gemma3Encoder {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<EncoderLayer>,
    norm: RmsNorm,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    sliding_window: usize,
    sliding_window_pattern: usize,
}

impl Gemma3Encoder {
    /// Build the encoder from weights and config.
    pub fn new(cfg: &Gemma3Config, vb: VarBuilder) -> CandleResult<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            // EncoderLayer construction is uniform — masking is per-layer in forward().
            let layer = EncoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            sliding_window: cfg.sliding_window,
            sliding_window_pattern: cfg.sliding_window_pattern,
        })
    }

    /// Forward pass returning full sequence hidden states `[batch, seq_len, hidden]`.
    ///
    /// Unlike the causal decoder's `forward` which returns last-token logits, this
    /// returns every position's representation so mean pooling can produce a fixed-
    /// size sentence embedding.
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;

        // Token embeddings scaled by sqrt(hidden_size).
        let mut xs = self.embed_tokens.forward(input_ids)?;
        xs = (xs * (self.hidden_size as f64).sqrt())?;

        // Extract mask-building parameters before borrowing layers mutably.
        let sliding_window = self.sliding_window;
        let sliding_window_pattern = self.sliding_window_pattern;
        let device = &self.device;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // Build per-layer bidirectional mask using extracted params.
            let is_full = (layer_idx + 1) % sliding_window_pattern == 0;
            let structural: Vec<f32> = if is_full {
                vec![0.0; seq_len * seq_len]
            } else {
                let half = sliding_window / 2;
                (0..seq_len)
                    .flat_map(|i| {
                        (0..seq_len).map(move |j| {
                            if (i as i64 - j as i64).abs() < half as i64 {
                                0.0
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                    })
                    .collect()
            };

            let mut layer_mask = Tensor::from_slice(&structural, (seq_len, seq_len), device)?
                .to_dtype(self.dtype)?;
            layer_mask = layer_mask.unsqueeze(0)?.unsqueeze(0)?;

            if let Some(pad_mask) = attention_mask {
                let pad_f32 = pad_mask.to_dtype(DType::F32)?;
                let one = Tensor::new(1.0f32, pad_mask.device())?;
                let inv_pad = (&one - &pad_f32)?.to_dtype(self.dtype)?;
                let inv_pad_4d = inv_pad.unsqueeze(1)?.unsqueeze(1)?;
                let pad_contrib = inv_pad_4d.broadcast_as((_b_size, 1, seq_len, seq_len))?;
                layer_mask = layer_mask.broadcast_add(&pad_contrib)?;
            }

            let layer_mask = layer_mask.expand((_b_size, 1, seq_len, seq_len))?;
            xs = layer.forward(&xs, Some(&layer_mask))?;
        }

        // Final RMSNorm — do NOT apply lm_head (no vocabulary projection for embeddings).
        self.norm.forward(&xs)
    }
}

// ---------------------------------------------------------------------------
// Public configuration and loaded model types.
// ---------------------------------------------------------------------------

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
pub fn mean_pooling(hidden_states: &Tensor, attention_mask: &Tensor) -> CandleResult<Tensor> {
    // attention_mask: (batch, seq_len) - 1 for real tokens, 0 for padding
    // hidden_states: (batch, seq_len, hidden_dim)

    let mask = attention_mask.to_dtype(hidden_states.dtype())?;
    let mask = mask.unsqueeze(2)?; // (batch, seq_len, 1)

    // Masked hidden states
    let masked = hidden_states.broadcast_mul(&mask)?;

    // Sum along sequence dimension
    let sum = masked.sum(1)?; // (batch, hidden_dim)

    // Count of real tokens per sequence (add epsilon to avoid division by zero)
    let mask_count = mask.sum(1)?;
    let mask_sum = (mask_count.to_dtype(DType::F32)? + 1e-8)?;

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

/// Loaded embedding model with all components.
///
/// Holds the Gemma-3 **encoder** (bidirectional attention, no KV cache) plus
/// the sentence-transformers pooling config and Dense projection head.
pub struct LoadedModel {
    /// Gemma-3 encoder backbone (bidirectional attention).
    pub model: Gemma3Encoder,
    /// Config for the backbone.
    pub config: Gemma3Config,
    /// Pad token ID extracted from config.json.
    pub pad_token_id: u32,
    /// Dense projection head (2 sequential Dense layers).
    pub projection_head: DenseProjectionHead,
    /// Embedding model configuration.
    pub embedding_config: EmbeddingModelConfig,
    /// Device used for inference.
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
/// 1. Gemma-3 backbone weights and config as a **bidirectional encoder**
/// 2. Pooling config (verifies mean pooling)
/// 3. Two Dense projection layers from sentence-transformers format
pub fn load_model(
    cache_dir: &Path,
    device: &Device,
    embedding_config: &EmbeddingModelConfig,
) -> Result<LoadedModel, ModelLoadError> {
    tracing::info!(
        "Loading EmbeddingGemma-300M encoder model from {}",
        cache_dir.display()
    );

    // Load config.json as raw JSON first to extract pad_token_id and other fields
    // that may not be in candle-transformers' Gemma3Config schema.
    let config_path = cache_dir.join("config.json");
    let config_json = std::fs::read_to_string(&config_path).map_err(ModelLoadError::Io)?;
    let config_value: serde_json::Value = serde_json::from_str(&config_json)
        .map_err(|e| ModelLoadError::Config(format!("Failed to parse config JSON: {e}")))?;

    // Extract pad_token_id separately (not in Gemma3Config struct).
    let pad_token_id = config_value
        .get("pad_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    tracing::debug!("Extracted pad_token_id: {pad_token_id}");

    // Parse standard Gemma3Config (serde ignores unknown fields by default).
    let gemma_config: Gemma3Config = serde_json::from_str(&config_json)
        .map_err(|e| ModelLoadError::Config(format!("Failed to parse Gemma-3 config: {e}")))?;

    tracing::info!(
        "Gemma-3 config: {} layers, hidden={} head_dim={} sliding_window={} pattern={}",
        gemma_config.num_hidden_layers,
        gemma_config.hidden_size,
        gemma_config.head_dim,
        gemma_config.sliding_window,
        gemma_config.sliding_window_pattern
    );

    // Load backbone weights and create the bidirectional encoder.
    let weights_path = cache_dir.join("model.safetensors");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], embedding_config.dtype, device)?
    };

    let model = Gemma3Encoder::new(&gemma_config, vb)
        .map_err(|e| ModelLoadError::Candle(CandleError(e)))?;
    tracing::info!("Loaded Gemma-3 encoder backbone (bidirectional attention)");

    // Load pooling config and verify mean pooling.
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

    // Load Dense projection head.
    let projection_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[
                cache_dir.join("2_Dense/model.safetensors"),
                cache_dir.join("3_Dense/model.safetensors"),
            ],
            DType::F32,
            device,
        )?
    };

    let projection_head = DenseProjectionHead::new(projection_vb)?;
    tracing::info!("Loaded Dense projection head (768 -> 3072 -> 768)");

    Ok(LoadedModel {
        model,
        config: gemma_config,
        pad_token_id,
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

/// Apply matryoshka truncation + L2 normalization to a single embedding vector.
///
/// Truncates the input to `target_dim` if it is longer (Matryoshka representation
/// learning), then applies L2 normalization so the output has unit length.
///
/// # Example
/// ```
/// use notectl_search::embeddings::model::normalize_embedding;
///
/// let vec = vec![3.0, 4.0, 0.0, 0.0];
/// let result = normalize_embedding(&vec, 2); // truncate to 2 dims
/// assert_eq!(result.len(), 2);
/// // [3, 4] normalized → [0.6, 0.8]
/// assert!((result[0] - 0.6).abs() < 1e-6);
/// assert!((result[1] - 0.8).abs() < 1e-6);
/// ```
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

/// Integration test: numerically validates the encoder against a known reference.
///
/// Gated behind `feature = "integration"` because it requires:
/// - Network access to download model weights on first run
/// - A valid `HF_TOKEN` with accepted license for `google/embeddinggemma-300m`
/// - Several minutes of CPU inference time
///
/// To populate REFERENCE_EMBEDDING / DOC_REFERENCE_EMBEDDING, run the model via
/// text-embeddings-inference (TEI) or a prior successful run of this test, then
/// paste the first N dimensions here. The assertion uses a tight tolerance (1e-4)
/// — if values drift, the encoder implementation has a bug (wrong attention mask,
/// incorrect layer ordering, etc.).
#[cfg(all(test, feature = "integration"))]
mod integration_tests {
    use super::*;
    use crate::embeddings::download;

    /// Reference embedding for "task: search result | query: hello world"
    /// produced by the text-embeddings-inference (TEI) reference implementation.
    ///
    /// TODO: Populate from a TEI run. Example format (768-dim, first 10 shown):
    /// ```text
    /// [-0.0234, 0.0156, -0.0891, 0.0423, -0.0067, 0.0312, -0.0178, 0.0645, -0.0023, 0.0189, ...]
    /// ```
    const REFERENCE_EMBEDDING: &[f32] = &[
        // TODO: Fill in reference values from a TEI run.
        // Until then, the test validates shape/dimension but not numerical correctness.
        0.0_f32, 0.0, 0.0, 0.0, 0.0,
    ];

    /// Reference embedding for "title: My Note | text: hello world"
    /// (document-text prefix path).
    ///
    /// TODO: Populate from a TEI run.
    const DOC_REFERENCE_EMBEDDING: &[f32] = &[
        // TODO: Fill in reference values from a TEI run.
        0.0_f32, 0.0, 0.0, 0.0, 0.0,
    ];

    const QUERY_TEST_INPUT: &str = "task: search result | query: hello world";
    const DOC_TEST_INPUT: &str = "title: My Note | text: hello world";

    /// Shared helper: load model, tokenize, encode, pool, project → embedding vec.
    fn get_embedding(input: &str) -> Vec<f32> {
        let cache_dir = download::default_cache_dir();
        let device = Device::Cpu;
        let embedding_config = EmbeddingModelConfig {
            output_dim: 768,
            max_seq_len: 2048,
            dtype: DType::F32,
        };

        let mut loaded = load_model(&cache_dir, &device, &embedding_config)
            .expect("Failed to load encoder model");

        let tokenizer_path = cache_dir.join("tokenizer.json");
        let tokenizer =
            tokenizers::Tokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer");

        let encoding = tokenizer.encode(input, false).expect("Tokenization failed");
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        let max_len = embedding_config.max_seq_len;
        let pad_id = loaded.pad_token_id;
        let mut padded = token_ids;
        padded.extend(std::iter::repeat_n(pad_id, max_len - padded.len()));

        let attention_mask: Vec<f32> = padded
            .iter()
            .map(|&id| if id == pad_id { 0.0 } else { 1.0 })
            .collect();

        let input_ids = Tensor::new(padded.as_slice(), &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let pad_tensor = Tensor::new(attention_mask.as_slice(), &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        let hidden_states = loaded
            .model
            .forward(&input_ids, Some(&pad_tensor))
            .expect("Encoder forward failed");

        // Verify shape: [1, seq_len, hidden_size].
        let dims = hidden_states.dims();
        assert_eq!(
            dims.len(),
            3,
            "Expected 3D hidden states, got {}D",
            dims.len()
        );
        assert_eq!(dims[0], 1, "Batch size should be 1");
        assert_eq!(
            dims[2], loaded.config.hidden_size,
            "Hidden dim mismatch: expected {}, got {}",
            loaded.config.hidden_size, dims[2]
        );

        // Mean pool + project.
        let pooling_mask = Tensor::ones(input_ids.shape().clone(), DType::F32, &device).unwrap();
        let pooled = mean_pooling(&hidden_states, &pooling_mask).expect("Mean pooling failed");
        let projected = loaded
            .projection_head
            .forward(&pooled)
            .expect("Projection failed");

        let embedding = projected.squeeze(0).unwrap();
        embedding.to_dtype(DType::F32).unwrap().to_vec1().unwrap()
    }

    /// Assert common properties shared by both query and document embeddings.
    fn assert_embedding_properties(embedding: &[f32], label: &str) {
        assert_eq!(
            embedding.len(),
            768,
            "{label}: Expected 768-dim embedding, got {}",
            embedding.len()
        );

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "{label}: Embedding should be L2-normalized, got norm {norm}"
        );
    }

    /// Assert that the first `ref_len` dimensions match a reference vector within tolerance.
    fn assert_matches_reference(embedding: &[f32], reference: &[f32], label: &str) {
        let ref_len = reference.len();
        for i in 0..ref_len.min(768) {
            let diff = (embedding[i] - reference[i]).abs();
            assert!(
                diff < 1e-4,
                "{label}: Dimension {i} mismatch: got {}, expected {} (diff={:.6e})",
                embedding[i],
                reference[i],
                diff
            );
        }
        eprintln!("{label}: first {ref_len} dimensions match reference within 1e-4");
    }

    fn skip_if_model_not_ready() -> bool {
        let cache_dir = download::default_cache_dir();
        if !download::is_model_ready(&cache_dir) {
            eprintln!(
                "Skipping integration test: model not downloaded at {}. \
                 Run with `cargo test --features integration -p notectl-search` \
                 after ensuring HF_TOKEN is set.",
                cache_dir.display()
            );
            return true;
        }
        false
    }

    /// Validates the query-text embedding path against a known reference.
    #[test]
    fn test_encoder_produces_correct_dimension() {
        if skip_if_model_not_ready() {
            return;
        }

        let embedding = get_embedding(QUERY_TEST_INPUT);
        assert_embedding_properties(&embedding, "Query embedding");

        // Numerical check against reference (when populated).
        if !REFERENCE_EMBEDDING.is_empty() && REFERENCE_EMBEDDING[0] != 0.0 {
            assert_matches_reference(&embedding, REFERENCE_EMBEDDING, "Query embedding");
        } else {
            eprintln!(
                "Query embedding: shape/dim/norm verified, but REFERENCE_EMBEDDING not populated. \
                 Populate REFERENCE_EMBEDDING in model.rs to enable numerical validation."
            );
        }
    }

    /// Validates the document-text embedding path against a known reference.
    #[test]
    fn test_document_embedding_matches_reference() {
        if skip_if_model_not_ready() {
            return;
        }

        let embedding = get_embedding(DOC_TEST_INPUT);
        assert_embedding_properties(&embedding, "Document embedding");

        // Numerical check against reference (when populated).
        if !DOC_REFERENCE_EMBEDDING.is_empty() && DOC_REFERENCE_EMBEDDING[0] != 0.0 {
            assert_matches_reference(&embedding, DOC_REFERENCE_EMBEDDING, "Document embedding");
        } else {
            eprintln!(
                "Document embedding: shape/dim/norm verified, but DOC_REFERENCE_EMBEDDING not populated. \
                 Populate DOC_REFERENCE_EMBEDDING in model.rs to enable numerical validation."
            );
        }
    }
}
