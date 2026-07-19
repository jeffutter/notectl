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
    ///
    /// Note: EmbeddingGemma safetensors stores tensors WITHOUT the "model." prefix
    /// (e.g. "embed_tokens.weight" not "model.embed_tokens.weight"), unlike the
    /// causal decoder variant. We load directly from the root VarBuilder.
    pub fn new(cfg: &Gemma3Config, vb: VarBuilder) -> CandleResult<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            // EncoderLayer construction is uniform — masking is per-layer in forward().
            let layer = EncoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
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
                let pad_bias = padding_bias(pad_mask, seq_len, self.dtype, pad_mask.device())?;
                layer_mask = layer_mask.broadcast_add(&pad_bias)?;
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

/// Build a padding bias tensor that excludes padded key positions from attention.
///
/// Uses a large negative FINITE constant (-1e9) rather than `f32::NEG_INFINITY`
/// because `0.0 * NEG_INFINITY = NaN` would corrupt real-token logits.
/// The value -1e9 is large enough that `exp(-1e9)` underflows to 0.0 in f32 softmax,
/// fully excluding padded keys while avoiding NaN propagation.
///
/// # Arguments
/// * `pad_mask` - Shape `[batch, seq_len]`, 1.0 for real tokens, 0.0 for padding
/// * `seq_len` - Sequence length (for broadcast shape)
/// * `dtype` - Target data type
/// * `device` - Device to allocate on
///
/// # Returns
/// Tensor of shape `[batch, 1, seq_len, seq_len]` ready to add to the layer mask.
pub fn padding_bias(
    pad_mask: &Tensor,
    seq_len: usize,
    dtype: DType,
    device: &Device,
) -> CandleResult<Tensor> {
    const PADDING_BIAS: f64 = -1e9;

    let pad_f32 = pad_mask.to_dtype(DType::F32)?;
    let ones = Tensor::ones(pad_f32.shape().clone(), DType::F32, device)?;
    let inv_pad = (&ones - &pad_f32)?; // 0.0 for real tokens, 1.0 for padded
    let neg_bias = Tensor::new(PADDING_BIAS as f32, device)?;
    let bias = inv_pad.broadcast_mul(&neg_bias)?; // 0.0 for real, -1e9 for padded
    let bias = bias.to_dtype(dtype)?;

    // Broadcast from [batch, seq_len] to [batch, 1, seq_len, seq_len]
    let b_size = pad_mask.dims()[0];
    let bias = bias.unsqueeze(1)?.unsqueeze(1)?;
    bias.expand((b_size, 1, seq_len, seq_len))
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
        // EmbeddingGemma Dense layers store weights under "linear" prefix
        // (e.g. "2_Dense/linear.weight" in safetensors) without a bias.
        let linear = candle_nn::linear_no_bias(input_dim, output_dim, vb.pp("linear"))?;
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

/// Two sequential Dense projection layers for EmbeddingGemma.
///
/// Constructed directly from individual safetensors files rather than via a
/// combined VarBuilder, since each file has its own "linear.weight" tensor.
#[derive(Debug)]
pub struct DenseProjectionHead {
    pub dense_2: DenseLayer, // 768 -> 3072 with tanh
    pub dense_3: DenseLayer, // 3072 -> 768 (or target dim) linear
}

impl DenseProjectionHead {
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
    let mut config_value: serde_json::Value = serde_json::from_str(&config_json)
        .map_err(|e| ModelLoadError::Config(format!("Failed to parse config JSON: {e}")))?;

    // Normalize HF config field names for candle-transformers compatibility.
    // HF uses "_sliding_window_pattern" but candle-transformers expects "sliding_window_pattern".
    if let Some(pattern) = config_value
        .as_object_mut()
        .and_then(|obj| obj.remove("_sliding_window_pattern"))
    {
        config_value
            .as_object_mut()
            .unwrap()
            .insert("sliding_window_pattern".to_string(), pattern);
    }
    let config_json_normalized = serde_json::to_string(&config_value)
        .map_err(|e| ModelLoadError::Config(format!("Failed to re-serialize config JSON: {e}")))?;

    // Extract pad_token_id separately (not in Gemma3Config struct).
    let pad_token_id = config_value
        .get("pad_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    tracing::debug!("Extracted pad_token_id: {pad_token_id}");

    // Parse standard Gemma3Config (serde ignores unknown fields by default).
    let gemma_config: Gemma3Config = serde_json::from_str(&config_json_normalized)
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

    // Support both string pooling_mode and boolean flag formats.
    // HF sentence-transformers uses booleans: pooling_mode_mean_tokens: true
    // Some configs use a string: pooling_mode: "mean"
    let is_mean_pooling = match pooling_config.get("pooling_mode") {
        Some(serde_json::Value::String(s)) => s == "mean",
        _ => pooling_config
            .get("pooling_mode_mean_tokens")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    };

    if !is_mean_pooling {
        return Err(ModelLoadError::Config(
            "Expected mean pooling configuration".to_string(),
        ));
    }
    tracing::info!("Verified mean pooling configuration");

    // Load Dense projection head.
    // Each Dense layer has its own safetensors file with "linear.weight" tensor.
    // We must load them separately to avoid key conflicts.
    let dense_2_path = cache_dir.join("2_Dense/model.safetensors");
    let dense_2_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[dense_2_path], DType::F32, device)? };
    let dense_2 = DenseLayer::new(dense_2_vb, 768, 3072, Some("tanh"))?;

    let dense_3_path = cache_dir.join("3_Dense/model.safetensors");
    let dense_3_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[dense_3_path], DType::F32, device)? };
    let dense_3 = DenseLayer::new(dense_3_vb, 3072, 768, None)?;

    let projection_head = DenseProjectionHead { dense_2, dense_3 };
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

/// Truncate token IDs to `max_len` and pad with `pad_id`.
///
/// Shared helper used by production code (`inner_embed_text` in embed.rs) and
/// test helpers. Centralized here so truncation+padding policy lives in one place
/// and callers cannot diverge or panic on oversized input via usize underflow.
pub(crate) fn truncate_and_pad(token_ids: &[u32], max_len: usize, pad_id: u32) -> Vec<u32> {
    let actual_len = token_ids.len().min(max_len);
    let mut padded = Vec::with_capacity(max_len);
    padded.extend_from_slice(&token_ids[..actual_len]);
    padded.extend(std::iter::repeat_n(pad_id, max_len - actual_len));
    padded
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

    #[test]
    fn test_mean_pooling_respects_attention_mask() {
        // hidden_states: (1, 4, 2) — batch=1, seq_len=4, hidden_dim=2
        // Positions 0,1 are real tokens with values [1.0, 1.0] and [3.0, 3.0]
        // Positions 2,3 are padding with deliberately large outlier values [100.0, 100.0] and [200.0, 200.0]
        let hidden_states = Tensor::new(
            &[[
                [1.0f32, 1.0],  // pos 0: real token
                [3.0, 3.0],     // pos 1: real token
                [100.0, 100.0], // pos 2: padding (outlier)
                [200.0, 200.0], // pos 3: padding (outlier)
            ]],
            &Device::Cpu,
        )
        .unwrap();

        // pad_mask: (1, 4) — first 2 are real (1.0), last 2 are padding (0.0)
        let pad_mask = Tensor::new(&[[1.0f32, 1.0, 0.0, 0.0]], &Device::Cpu).unwrap();

        let result = mean_pooling(&hidden_states, &pad_mask).unwrap();
        let pooled: Vec<f32> = result.squeeze(0).unwrap().to_vec1().unwrap();

        // Mean of only real positions: ([1.0]+[3.0])/2 = [2.0, 2.0]
        assert_eq!(pooled.len(), 2);
        assert!(
            (pooled[0] - 2.0).abs() < 1e-6,
            "Expected 2.0, got {}",
            pooled[0]
        );
        assert!(
            (pooled[1] - 2.0).abs() < 1e-6,
            "Expected 2.0, got {}",
            pooled[1]
        );

        // Contrast: if an all-ones mask were used, padding positions would corrupt the result.
        // This documents why the fix matters.
        let ones_mask = Tensor::ones(pad_mask.shape().clone(), DType::F32, &Device::Cpu).unwrap();
        let wrong_result = mean_pooling(&hidden_states, &ones_mask).unwrap();
        let wrong_pooled: Vec<f32> = wrong_result.squeeze(0).unwrap().to_vec1().unwrap();
        // With all-ones mask: ([1+3+100+200]/4, same) = [76.0, 76.0] — very different!
        assert!(
            (wrong_pooled[0] - 76.0).abs() < 1e-6,
            "All-ones mask should give 76.0, got {}",
            wrong_pooled[0]
        );
        assert!(
            (pooled[0] - wrong_pooled[0]).abs() > 10.0,
            "Results must differ significantly between correct and incorrect masks"
        );
    }

    #[test]
    fn test_padding_bias_excludes_padded_positions() {
        // pad_mask: [1, 4] — first 2 are real tokens (1.0), last 2 are padding (0.0)
        let pad_mask = Tensor::new(&[[1.0f32, 1.0, 0.0, 0.0]], &Device::Cpu).unwrap();
        let bias = padding_bias(&pad_mask, 4, DType::F32, &Device::Cpu).unwrap();

        // Result shape: [1, 1, 4, 4]
        let dims = bias.dims();
        assert_eq!(dims, &[1, 1, 4, 4], "Expected [1, 1, 4, 4], got {:?}", dims);

        // Flatten and check values
        let flat = bias.flatten_all().unwrap();
        let values: Vec<f32> = flat.to_vec1().unwrap();
        // [batch=0, head=0, query_pos, key_pos]
        // For each query position, real key positions (0,1) should be ~0.0,
        // padded key positions (2,3) should be large negative (~-1e9).
        for q in 0..4 {
            for k in 0..4 {
                let idx = q * 4 + k;
                let val = values[idx];
                assert!(
                    val.is_finite(),
                    "Position [{}, {}]: got non-finite value",
                    q,
                    k
                );
                if k < 2 {
                    // Real key position: should be ~0.0
                    assert!(
                        val.abs() < 1e-4,
                        "Position [{}, {}]: expected ~0.0 for real key, got {}",
                        q,
                        k,
                        val
                    );
                } else {
                    // Padded key position: should be large negative
                    assert!(
                        val < -1e6,
                        "Position [{}, {}]: expected large negative for padded key, got {}",
                        q,
                        k,
                        val
                    );
                }
            }
        }
    }

    #[test]
    fn test_truncate_and_pad_over_length_does_not_panic() {
        // Input longer than max_len must truncate without usize underflow panic
        let input = vec![1u32; 2049];
        let result = truncate_and_pad(&input, 2048, 0);
        assert_eq!(result.len(), 2048);
        assert!(result.iter().all(|&x| x == 1));
    }

    #[test]
    fn test_truncate_and_pad_under_length_pads() {
        let input = vec![1u32; 10];
        let result = truncate_and_pad(&input, 2048, 50256);
        assert_eq!(result.len(), 2048);
        assert!(result[..10].iter().all(|&x| x == 1));
        assert!(result[10..].iter().all(|&x| x == 50256));
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
    #[allow(clippy::excessive_precision)]
    const REFERENCE_EMBEDDING: &[f32] = &[
        -0.0349282250_f32,
        0.0524583310_f32,
        0.0139811104_f32,
        -0.0048876028_f32,
        -0.0459638499_f32,
        -0.0164953321_f32,
        -0.0452917702_f32,
        -0.0471804962_f32,
        -0.0200994685_f32,
        -0.0449239276_f32,
        -0.0391916111_f32,
        -0.1013602316_f32,
        0.0111086955_f32,
        -0.0343020111_f32,
        0.0147559233_f32,
        -0.0394711159_f32,
        -0.0436390415_f32,
        -0.0307584815_f32,
        -0.0044019716_f32,
        -0.0361882150_f32,
        0.0129468134_f32,
        0.0394009165_f32,
        -0.0053750817_f32,
        -0.0146713946_f32,
        0.0401485488_f32,
        0.0545513444_f32,
        0.0097395675_f32,
        -0.0022008102_f32,
        0.0013190458_f32,
        0.0460765250_f32,
        0.0637934059_f32,
        0.0287040863_f32,
        0.0386991836_f32,
        -0.0190281738_f32,
        -0.0068360409_f32,
        0.0071221590_f32,
        -0.0433639400_f32,
        0.0003272521_f32,
        -0.0344003811_f32,
        0.0000722781_f32,
        -0.0259061698_f32,
        -0.0069462876_f32,
        0.0102423308_f32,
        -0.0090591246_f32,
        0.0090546450_f32,
        -0.0766480640_f32,
        -0.1313575804_f32,
        0.0350615531_f32,
        -0.0155984331_f32,
        -0.0239410847_f32,
        0.0296098813_f32,
        0.0204759426_f32,
        0.0295789316_f32,
        0.0635981932_f32,
        0.0063798837_f32,
        -0.0338534713_f32,
        0.0550304018_f32,
        -0.0270969141_f32,
        -0.0349636562_f32,
        -0.0192655548_f32,
        -0.0126224654_f32,
        -0.0078769932_f32,
        0.0379194990_f32,
        0.0185090732_f32,
        0.0285018235_f32,
        0.0239582676_f32,
        0.0070757908_f32,
        0.0110630747_f32,
        0.0465801321_f32,
        0.0064584482_f32,
        -0.0404024981_f32,
        0.0315957963_f32,
        0.0216051415_f32,
        0.0066163740_f32,
        0.0168739893_f32,
        0.0676316097_f32,
        0.0227373615_f32,
        -0.0002325179_f32,
        0.0033178090_f32,
        -0.0016238563_f32,
        0.0120818689_f32,
        -0.0409801193_f32,
        -0.0198300332_f32,
        -0.0606268123_f32,
        0.0685063899_f32,
        0.0639080554_f32,
        -0.0050824131_f32,
        -0.0053795301_f32,
        0.0285310354_f32,
        0.0051133004_f32,
        -0.0026060611_f32,
        -0.0242443774_f32,
        -0.0481572710_f32,
        -0.0027490207_f32,
        -0.0501635000_f32,
        0.0677650645_f32,
        -0.0391496196_f32,
        0.0648851022_f32,
        -0.0539397933_f32,
        -0.0000214912_f32,
        0.0039055760_f32,
        -0.0091801677_f32,
        0.0021946609_f32,
        -0.0282503087_f32,
        0.0413238779_f32,
        0.0689216182_f32,
        0.0377935059_f32,
        0.0450904705_f32,
        0.0102949059_f32,
        0.0395069942_f32,
        0.0387460180_f32,
        0.0364990309_f32,
        -0.0635267720_f32,
        -0.0286839064_f32,
        0.0024081678_f32,
        -0.0462765284_f32,
        -0.0568147711_f32,
        0.0402552895_f32,
        0.0025676936_f32,
        -0.0195107777_f32,
        -0.0111373970_f32,
        -0.0022184898_f32,
        -0.0745291933_f32,
        0.0247967187_f32,
        -0.0512879193_f32,
        0.0543581247_f32,
        0.0364535749_f32,
        0.0233767517_f32,
        0.0335834324_f32,
        -0.0233644713_f32,
        -0.0361356437_f32,
        -0.0155265993_f32,
        -0.0230696164_f32,
        0.0339181907_f32,
        -0.0341565795_f32,
        -0.0020242925_f32,
        0.0083346246_f32,
        0.0184933785_f32,
        -0.0481513664_f32,
        0.0327665284_f32,
        0.0021810704_f32,
        -0.1218738481_f32,
        -0.0143320989_f32,
        0.0639321730_f32,
        -0.0417836420_f32,
        -0.0171781629_f32,
        -0.0110095376_f32,
        -0.0488888212_f32,
        -0.0169199072_f32,
        -0.0196150318_f32,
        0.0225185938_f32,
        -0.0232688151_f32,
        0.0979719907_f32,
        0.0390930958_f32,
        -0.0257790927_f32,
        0.0581156127_f32,
        -0.0017127359_f32,
        -0.0117609771_f32,
        -0.0096620051_f32,
        -0.0259577949_f32,
        0.0358207040_f32,
        -0.0102975890_f32,
        0.0471830219_f32,
        -0.0187134333_f32,
        -0.0107871629_f32,
        0.0158885736_f32,
        -0.0229510814_f32,
        -0.0153014148_f32,
        -0.0250901431_f32,
        -0.0282770377_f32,
        -0.0027673428_f32,
        -0.0269467719_f32,
        0.0123846903_f32,
        -0.0012741211_f32,
        0.0025055516_f32,
        0.0611126460_f32,
        -0.0286443885_f32,
        -0.0680636019_f32,
        -0.0141369095_f32,
        0.0307317022_f32,
        0.0240872856_f32,
        0.0052831075_f32,
        0.0433933660_f32,
        0.0248002857_f32,
        -0.0093101030_f32,
        0.0184877161_f32,
        0.1126012057_f32,
        -0.0364216045_f32,
        0.0556012131_f32,
        -0.0075751892_f32,
        -0.0011371970_f32,
        -0.0313574858_f32,
        -0.0445878953_f32,
        -0.0442263782_f32,
        -0.0058000055_f32,
        -0.0261811074_f32,
        0.0405358300_f32,
        0.0186078511_f32,
        0.0896684602_f32,
        0.0343411155_f32,
        0.0063400231_f32,
        0.0004284943_f32,
        0.0011565387_f32,
        0.0691757724_f32,
        -0.0000241714_f32,
        -0.0323612317_f32,
        0.0775853842_f32,
        0.0014613208_f32,
        -0.0502395630_f32,
        0.0154531775_f32,
        0.0774058774_f32,
        -0.0020138666_f32,
        0.0383914523_f32,
        0.0395041890_f32,
        0.0497101657_f32,
        0.0370454416_f32,
        -0.0581961870_f32,
        -0.0249130130_f32,
        -0.0073086838_f32,
        -0.0408839993_f32,
        0.0423110053_f32,
        -0.0271588601_f32,
        -0.0446347855_f32,
        -0.0098427022_f32,
        0.0040987781_f32,
        -0.0083996626_f32,
        0.0106512969_f32,
        0.0184833836_f32,
        -0.0080138557_f32,
        -0.0489203073_f32,
        0.0209720340_f32,
        -0.0102851680_f32,
        -0.0548220798_f32,
        0.0309326220_f32,
        -0.0447145924_f32,
        -0.0135518936_f32,
        -0.0239673518_f32,
        0.0067174588_f32,
        0.0070351167_f32,
        0.0052932808_f32,
        -0.0504648909_f32,
        0.0348047018_f32,
        0.0130987307_f32,
        -0.0023103950_f32,
        0.0142444829_f32,
        -0.0120693706_f32,
        -0.0193287376_f32,
        -0.0137907583_f32,
        -0.0067226840_f32,
        -0.0019979698_f32,
        0.0021105933_f32,
        0.0108568249_f32,
        0.0156094618_f32,
        -0.0371889509_f32,
        0.0134577537_f32,
        0.0314430781_f32,
        -0.0733381882_f32,
        0.0304335505_f32,
        -0.0009740960_f32,
        0.0226816367_f32,
        -0.0268779136_f32,
        -0.0061317002_f32,
        -0.0310306270_f32,
        0.0015207620_f32,
        0.0005595590_f32,
        -0.0049729114_f32,
        -0.0016767521_f32,
        -0.0019168177_f32,
        0.0344905481_f32,
        -0.0082928892_f32,
        0.0143254548_f32,
        0.0355405435_f32,
        -0.0079336567_f32,
        0.0375875086_f32,
        0.0020489625_f32,
        0.0215592086_f32,
        -0.0506306998_f32,
        0.0098395543_f32,
        -0.0310507920_f32,
        0.0019620133_f32,
        0.0080856914_f32,
        0.0643069744_f32,
        0.0174537972_f32,
        0.0180952530_f32,
        0.0299181249_f32,
        -0.0056753093_f32,
        -0.0246040206_f32,
        -0.0128561258_f32,
        -0.0280814618_f32,
        -0.0216736961_f32,
        0.0073976568_f32,
        -0.0375976861_f32,
        0.0465228967_f32,
        -0.0045826477_f32,
        -0.0395423025_f32,
        -0.0081455037_f32,
        0.0206104573_f32,
        0.0447660014_f32,
        0.0044220532_f32,
        -0.0159630924_f32,
        0.0233107302_f32,
        0.0241691787_f32,
        -0.0681353584_f32,
        -0.0139924334_f32,
        -0.1048336476_f32,
        0.0042422237_f32,
        -0.0296306554_f32,
        0.0179240871_f32,
        -0.0019433456_f32,
        0.0272791628_f32,
        -0.0534571633_f32,
        0.0341812149_f32,
        -0.0267507993_f32,
        -0.0111057255_f32,
        0.0270615201_f32,
        -0.0190700274_f32,
        0.0462891050_f32,
        -0.0300486926_f32,
        -0.0024357308_f32,
        0.0091716014_f32,
        0.0304797273_f32,
        -0.0074448721_f32,
        0.0117567768_f32,
        0.0101607116_f32,
        -0.0091517940_f32,
        -0.0098608239_f32,
        0.0035909005_f32,
        -0.0089926077_f32,
        -0.0411942489_f32,
        0.0516561083_f32,
        -0.0351971462_f32,
        0.0014762823_f32,
        0.0323086493_f32,
        0.0174310673_f32,
        0.0227610357_f32,
        0.0381628685_f32,
        0.0820909515_f32,
        -0.0341519564_f32,
        0.0048987782_f32,
        0.0231275745_f32,
        -0.0071560894_f32,
        0.0063406778_f32,
        0.0164798331_f32,
        0.0242811758_f32,
        -0.0130572272_f32,
        0.0268054530_f32,
        0.0176040772_f32,
        -0.0202233549_f32,
        -0.0529228412_f32,
        -0.0241497587_f32,
        -0.0228912476_f32,
        0.0196276847_f32,
        0.0061314111_f32,
        -0.0432388075_f32,
        0.0028415225_f32,
        -0.0231605452_f32,
        0.0729457363_f32,
        0.0474122204_f32,
        -0.0448070392_f32,
        0.0123278918_f32,
        -0.0467012860_f32,
        -0.0994639471_f32,
        0.0632874966_f32,
        0.0464659296_f32,
        0.0195075218_f32,
        -0.0331719480_f32,
        -0.0103478320_f32,
        0.0760760382_f32,
        -0.0089697596_f32,
        0.0234278999_f32,
        -0.0345226489_f32,
        0.0340051278_f32,
        -0.0185859390_f32,
        -0.0147756459_f32,
        0.0230489857_f32,
        -0.0329779349_f32,
        0.0177419242_f32,
        -0.0009096723_f32,
        -0.0081767486_f32,
        0.0208942648_f32,
        -0.0374894254_f32,
        -0.0025995420_f32,
        -0.0871822909_f32,
        -0.0143331802_f32,
        0.0542068891_f32,
        0.0306015313_f32,
        -0.0152752548_f32,
        -0.0232978407_f32,
        -0.0069065760_f32,
        0.0475145616_f32,
        -0.0164074209_f32,
        -0.0226434078_f32,
        -0.0074637230_f32,
        0.0182670318_f32,
        0.0210876577_f32,
        0.0778785348_f32,
        0.0413118564_f32,
        0.0235703979_f32,
        0.0380185843_f32,
        0.0682927743_f32,
        0.0337492153_f32,
        -0.0491934158_f32,
        -0.0116623687_f32,
        0.0165962353_f32,
        0.0219696350_f32,
        -0.0052519813_f32,
        -0.0075175781_f32,
        0.0513210818_f32,
        0.0295921415_f32,
        0.0521426238_f32,
        0.0067865364_f32,
        0.0212819334_f32,
        0.0429900624_f32,
        0.0374500453_f32,
        -0.0244949106_f32,
        -0.0020888092_f32,
        -0.0369056463_f32,
        0.0249671005_f32,
        0.0029643883_f32,
        0.0140159177_f32,
        -0.0323776715_f32,
        -0.0063744267_f32,
        0.0089756902_f32,
        0.0050163870_f32,
        -0.0282134973_f32,
        0.0062286514_f32,
        -0.0178154353_f32,
        0.0117335934_f32,
        -0.0008290336_f32,
        -0.0792752281_f32,
        -0.0403343253_f32,
        -0.0132621340_f32,
        -0.0367146991_f32,
        -0.0719802156_f32,
        0.0371196382_f32,
        0.0165850334_f32,
        0.0188116562_f32,
        -0.0275103971_f32,
        -0.0257000178_f32,
        -0.0475016981_f32,
        0.0393344574_f32,
        0.0327370651_f32,
        0.0151055967_f32,
        0.0110769896_f32,
        -0.0272220690_f32,
        0.0206203852_f32,
        -0.0335153639_f32,
        0.0060584550_f32,
        0.0230882075_f32,
        -0.0170601085_f32,
        -0.0754025057_f32,
        -0.0042712521_f32,
        -0.0221750010_f32,
        -0.0143561717_f32,
        -0.0012510287_f32,
        -0.0173174199_f32,
        -0.0014491921_f32,
        0.0357087143_f32,
        -0.0413500927_f32,
        0.0471107140_f32,
        -0.0035250087_f32,
        0.0223005731_f32,
        -0.0701603219_f32,
        -0.0346595123_f32,
        -0.0610337481_f32,
        -0.0370770842_f32,
        0.0008438050_f32,
        -0.0096052829_f32,
        0.0144303301_f32,
        -0.0664260685_f32,
        -0.0105323736_f32,
        0.0140761994_f32,
        0.0001197310_f32,
        -0.0257431380_f32,
        -0.0575921051_f32,
        -0.0001953601_f32,
        -0.0098919794_f32,
        0.0045720427_f32,
        -0.0533865988_f32,
        -0.0681549013_f32,
        0.0154040800_f32,
        -0.0034725235_f32,
        0.0080191242_f32,
        -0.0445646718_f32,
        -0.0034970401_f32,
        0.0107999565_f32,
        0.0318594202_f32,
        -0.0043767216_f32,
        -0.0339908227_f32,
        0.0093577327_f32,
        -0.0230793580_f32,
        0.0171687193_f32,
        -0.0082457978_f32,
        0.0412068181_f32,
        0.0326388665_f32,
        0.0926216543_f32,
        0.0106749628_f32,
        0.0612037331_f32,
        0.0059743677_f32,
        0.0267093088_f32,
        -0.0050769309_f32,
        0.0442783758_f32,
        0.0778416917_f32,
        -0.0050336746_f32,
        0.0190791097_f32,
        -0.0297766887_f32,
        0.0233919639_f32,
        -0.0227432698_f32,
        0.0071070297_f32,
        -0.0119432472_f32,
        -0.0230563581_f32,
        -0.0217884462_f32,
        -0.0203727446_f32,
        0.0082420101_f32,
        0.0442822576_f32,
        0.0339948684_f32,
        0.0249916185_f32,
        0.0133100599_f32,
        0.0375269316_f32,
        0.0090978965_f32,
        -0.0062516499_f32,
        0.0135158757_f32,
        0.0284924041_f32,
        -0.0298781693_f32,
        -0.0347450599_f32,
        0.0052107116_f32,
        -0.0572635494_f32,
        -0.0188173763_f32,
        0.0330709033_f32,
        0.0841954872_f32,
        -0.0211038608_f32,
        0.0058431067_f32,
        0.0152682588_f32,
        0.0021376342_f32,
        -0.0357474610_f32,
        0.0249282271_f32,
        -0.0357809253_f32,
        -0.0040627779_f32,
        -0.0263943877_f32,
        -0.0370815098_f32,
        0.0084089488_f32,
        -0.0009230456_f32,
        0.0187452380_f32,
        -0.0043341569_f32,
        0.0053555495_f32,
        0.0059591890_f32,
        -0.0297590047_f32,
        -0.0200451780_f32,
        0.0450035185_f32,
        -0.0084334482_f32,
        0.0254917536_f32,
        -0.0255448576_f32,
        0.0242836382_f32,
        0.0357636251_f32,
        0.0208846610_f32,
        0.0004156965_f32,
        0.0677331537_f32,
        -0.0552350767_f32,
        0.0185661111_f32,
        -0.0354269482_f32,
        0.0396690182_f32,
        0.0195037872_f32,
        -0.0092244064_f32,
        0.0086559160_f32,
        0.0332544371_f32,
        -0.0507626608_f32,
        0.0083908420_f32,
        -0.0182832871_f32,
        0.0143392393_f32,
        0.0110658715_f32,
        -0.0035508650_f32,
        -0.0116938064_f32,
        0.0153362099_f32,
        -0.0092190560_f32,
        0.0247907769_f32,
        -0.0048067160_f32,
        0.0375973321_f32,
        0.0648970753_f32,
        -0.0535021275_f32,
        0.0074813385_f32,
        -0.0399161354_f32,
        -0.0069566579_f32,
        -0.0550117940_f32,
        0.0251469444_f32,
        0.0233458262_f32,
        -0.0212505478_f32,
        -0.0334347710_f32,
        -0.0353488438_f32,
        -0.0069832839_f32,
        0.0266135558_f32,
        -0.0201419462_f32,
        0.0257915594_f32,
        0.0138518307_f32,
        0.0303215366_f32,
        0.0352422930_f32,
        0.0172761250_f32,
        0.0423763804_f32,
        -0.0458974466_f32,
        0.0200201906_f32,
        -0.0535523407_f32,
        -0.0523644537_f32,
        -0.0162031669_f32,
        -0.0476774313_f32,
        -0.0218947195_f32,
        0.0623534434_f32,
        -0.0085096685_f32,
        0.0267311446_f32,
        0.0069135721_f32,
        -0.0987140536_f32,
        0.0258331392_f32,
        0.0399373993_f32,
        0.0267094597_f32,
        0.0878399089_f32,
        -0.0197967272_f32,
        -0.0580682904_f32,
        -0.0004737091_f32,
        0.0342713147_f32,
        -0.0798224807_f32,
        0.0413473286_f32,
        0.0074818670_f32,
        0.0613307878_f32,
        -0.0272356551_f32,
        0.0043104896_f32,
        0.0593005195_f32,
        -0.0338903442_f32,
        -0.0139216408_f32,
        -0.0656182170_f32,
        0.0042465362_f32,
        -0.0236415546_f32,
        -0.0206683874_f32,
        -0.0004182184_f32,
        -0.0165964440_f32,
        0.0148867331_f32,
        0.0065835360_f32,
        -0.0051847217_f32,
        -0.0334599391_f32,
        -0.0249460116_f32,
        -0.0336090960_f32,
        -0.0723929927_f32,
        -0.0113905165_f32,
        0.0761004314_f32,
        0.0257571153_f32,
        0.0271225888_f32,
        -0.0144354459_f32,
        0.0226180833_f32,
        0.0457739085_f32,
        0.0139026139_f32,
        -0.0404011570_f32,
        -0.0479466766_f32,
        0.0152299711_f32,
        -0.0023231979_f32,
        -0.0071941591_f32,
        0.0447358787_f32,
        0.0517934635_f32,
        -0.0289065894_f32,
        0.0736737922_f32,
        -0.0413613059_f32,
        0.0129903574_f32,
        0.0191575587_f32,
        -0.0228222515_f32,
        0.0132019538_f32,
        0.0334723443_f32,
        -0.0027141259_f32,
        -0.0797985494_f32,
        -0.0553696416_f32,
        -0.0163887702_f32,
        -0.0753942430_f32,
        0.0055507575_f32,
        -0.0162524972_f32,
        0.0066787852_f32,
        -0.0509655774_f32,
        -0.0166708604_f32,
        0.0365909673_f32,
        -0.0747479573_f32,
        0.0341969915_f32,
        -0.0343000740_f32,
        0.0186077040_f32,
        0.0412788801_f32,
        0.0292979106_f32,
        0.0283435807_f32,
        0.0060846377_f32,
        -0.0357411951_f32,
        0.0500006266_f32,
        0.0116722155_f32,
        0.0564493574_f32,
        0.0185351651_f32,
        0.0053898008_f32,
        -0.0202493668_f32,
        0.0166381691_f32,
        -0.0072923480_f32,
        0.0062960549_f32,
        -0.0095190033_f32,
        0.0098204166_f32,
        -0.0290356930_f32,
        -0.0656791106_f32,
        0.0591278188_f32,
        -0.0161691662_f32,
        0.0243434180_f32,
        0.0232964121_f32,
        0.0309442002_f32,
        0.0915784314_f32,
        0.0489864312_f32,
        -0.0017544471_f32,
        0.0073016211_f32,
        0.0067703198_f32,
        -0.0687831938_f32,
        -0.0438307859_f32,
        -0.0351276621_f32,
        0.0201995037_f32,
        0.0406524017_f32,
        -0.0349703990_f32,
        0.0021968132_f32,
        0.0092676282_f32,
        0.0088969553_f32,
        0.0528202243_f32,
        -0.0196557138_f32,
        -0.0820971355_f32,
        -0.0399927720_f32,
        0.0057649757_f32,
        -0.0483954512_f32,
        0.1144972518_f32,
        0.0043106140_f32,
        -0.0474873520_f32,
        0.0102593424_f32,
        -0.0213474855_f32,
        -0.0032376135_f32,
        -0.0334208570_f32,
        -0.0061156577_f32,
        0.0302038230_f32,
        0.0794201568_f32,
        0.0366000235_f32,
        0.0025032023_f32,
        0.0543186963_f32,
        -0.0651494265_f32,
        0.0817086250_f32,
        -0.0108483238_f32,
        -0.0451175272_f32,
        0.0730452687_f32,
        0.0554438420_f32,
        -0.0447329581_f32,
        0.0430107042_f32,
        -0.0655275583_f32,
        0.0504876077_f32,
        -0.0915234014_f32,
        -0.0174817834_f32,
        0.0591751896_f32,
        -0.0211535767_f32,
        0.0012959609_f32,
        -0.0553262942_f32,
        0.0157942344_f32,
        -0.0330143087_f32,
        0.0396910384_f32,
        0.0640995726_f32,
        -0.0113766650_f32,
        0.0072555691_f32,
        -0.0132946270_f32,
        0.0304037910_f32,
        -0.0206521396_f32,
        -0.0224719755_f32,
        0.0613088012_f32,
        -0.0507167690_f32,
        0.0104317870_f32,
        0.0225156248_f32,
        0.0069939047_f32,
        -0.0407054350_f32,
        -0.0114555750_f32,
        -0.0369476527_f32,
        -0.1236943305_f32,
    ];

    /// Flip to `true` in the same commit that populates REFERENCE_EMBEDDING with real values.
    /// See TASK-1.14.2.1.
    const REFERENCE_EMBEDDING_POPULATED: bool = true;

    /// Reference embedding for "title: My Note | text: hello world"
    /// (document-text prefix path).
    ///
    /// TODO: Populate from a TEI run.
    #[allow(clippy::excessive_precision)]
    const DOC_REFERENCE_EMBEDDING: &[f32] = &[
        -0.0232019797_f32,
        0.0271538738_f32,
        0.0020510182_f32,
        0.0349158421_f32,
        -0.0548576452_f32,
        -0.0073120221_f32,
        -0.0464903116_f32,
        -0.0205965135_f32,
        -0.0317338258_f32,
        -0.0280351155_f32,
        -0.0038790053_f32,
        -0.1003270820_f32,
        -0.0608734973_f32,
        -0.0544943102_f32,
        0.0254183300_f32,
        -0.0050556841_f32,
        -0.0493353568_f32,
        -0.0083367610_f32,
        -0.0433397964_f32,
        -0.0477738976_f32,
        0.0248342082_f32,
        -0.0028988228_f32,
        -0.0016341388_f32,
        -0.0528560542_f32,
        0.0161726102_f32,
        0.0137891378_f32,
        0.0079497015_f32,
        0.0269431509_f32,
        0.0416246206_f32,
        0.0511197746_f32,
        0.0520196967_f32,
        0.0733741224_f32,
        0.0265845824_f32,
        0.0077883457_f32,
        0.0066973604_f32,
        -0.0064519383_f32,
        -0.0791375786_f32,
        -0.0438932180_f32,
        -0.0150166275_f32,
        0.0006069895_f32,
        -0.0296164993_f32,
        0.0377353765_f32,
        0.0329653136_f32,
        0.0198783036_f32,
        0.0014170725_f32,
        -0.0778129026_f32,
        -0.0840978175_f32,
        -0.0399077572_f32,
        -0.0087603144_f32,
        -0.0620794110_f32,
        0.0530286767_f32,
        0.0075750523_f32,
        0.0440522134_f32,
        0.0173724126_f32,
        -0.0379395746_f32,
        0.0216358583_f32,
        0.0393492915_f32,
        0.0054821139_f32,
        0.0143334428_f32,
        -0.0324877463_f32,
        -0.0064072926_f32,
        -0.0397672541_f32,
        0.0300326794_f32,
        0.0201414414_f32,
        -0.0042619030_f32,
        0.0443973616_f32,
        0.0179312006_f32,
        -0.0007570540_f32,
        0.0552098975_f32,
        0.0724076927_f32,
        -0.0270000175_f32,
        0.0568500049_f32,
        0.0039898809_f32,
        -0.0566894785_f32,
        0.0166889969_f32,
        0.0611167327_f32,
        0.0005435630_f32,
        0.0100953309_f32,
        -0.0292637255_f32,
        -0.0096379276_f32,
        0.0245051999_f32,
        -0.0097864587_f32,
        0.0023549169_f32,
        -0.0522622056_f32,
        0.0309971385_f32,
        0.0116389245_f32,
        -0.0132437311_f32,
        -0.0399484709_f32,
        -0.0021781384_f32,
        -0.0378793851_f32,
        -0.0052129417_f32,
        0.0279580466_f32,
        0.0143193016_f32,
        -0.0155666620_f32,
        -0.0093467468_f32,
        0.0519325584_f32,
        -0.0259567089_f32,
        0.0567906611_f32,
        -0.0302062780_f32,
        0.0073904013_f32,
        -0.0018882041_f32,
        0.0704462305_f32,
        0.0273968093_f32,
        -0.0209686887_f32,
        -0.0025666279_f32,
        0.0439368486_f32,
        -0.0033659895_f32,
        0.0238654595_f32,
        -0.0065620490_f32,
        0.0276693292_f32,
        -0.0039622867_f32,
        -0.0286706984_f32,
        -0.0681793988_f32,
        0.0195434745_f32,
        0.0396968760_f32,
        -0.0436872877_f32,
        -0.0614142232_f32,
        0.0180019084_f32,
        0.0322383679_f32,
        0.0195962153_f32,
        0.0065629883_f32,
        0.0011457802_f32,
        -0.0825045854_f32,
        0.0299528260_f32,
        -0.0577547029_f32,
        0.0772492811_f32,
        0.0223404802_f32,
        -0.0533228405_f32,
        0.0322454534_f32,
        0.0359617621_f32,
        -0.0579607598_f32,
        -0.0009010420_f32,
        0.0322269239_f32,
        -0.0216491856_f32,
        -0.0395862721_f32,
        -0.0156639535_f32,
        0.0406109095_f32,
        0.0249807090_f32,
        -0.0130931269_f32,
        0.0631284192_f32,
        0.0186761394_f32,
        -0.0829481333_f32,
        -0.0516539514_f32,
        0.0705519095_f32,
        0.0420778245_f32,
        0.0219432153_f32,
        -0.0040723803_f32,
        -0.0615978055_f32,
        0.0122472411_f32,
        -0.0337149575_f32,
        0.0005104025_f32,
        -0.0072870450_f32,
        0.0349914134_f32,
        0.0204845704_f32,
        -0.0523702092_f32,
        -0.0036847135_f32,
        0.0610579848_f32,
        -0.0183705520_f32,
        -0.0102639208_f32,
        -0.0503158830_f32,
        0.0307058748_f32,
        -0.0435955040_f32,
        0.0116310259_f32,
        0.0431781560_f32,
        0.0094706081_f32,
        0.0669119507_f32,
        0.0166303385_f32,
        -0.0117661757_f32,
        -0.0064485334_f32,
        -0.0119006801_f32,
        -0.0071713743_f32,
        -0.0316368267_f32,
        0.0185558796_f32,
        0.0282466114_f32,
        0.0242362544_f32,
        0.0501263291_f32,
        0.0023184035_f32,
        -0.0623159297_f32,
        -0.0114719262_f32,
        -0.0157069694_f32,
        -0.0188478641_f32,
        0.0203821324_f32,
        0.0033291893_f32,
        0.0165081322_f32,
        -0.0210751742_f32,
        0.0186041500_f32,
        0.0748612359_f32,
        -0.0134049347_f32,
        0.0405943282_f32,
        -0.0514801517_f32,
        -0.0876804814_f32,
        0.0290724598_f32,
        -0.0508380495_f32,
        -0.0205897614_f32,
        -0.0144881643_f32,
        0.0261121262_f32,
        0.0612594709_f32,
        -0.0201888606_f32,
        0.0363974236_f32,
        0.0377018340_f32,
        0.0400325544_f32,
        -0.0101067983_f32,
        0.0092186416_f32,
        0.0656371489_f32,
        -0.0043271272_f32,
        -0.0372663438_f32,
        -0.0030136169_f32,
        0.0179732256_f32,
        -0.0624040812_f32,
        -0.0032373220_f32,
        0.0541809686_f32,
        -0.0029642934_f32,
        0.0154494736_f32,
        0.0118078142_f32,
        0.0342433676_f32,
        -0.0041980124_f32,
        -0.0444727130_f32,
        -0.0104668271_f32,
        -0.0427311286_f32,
        -0.0222971458_f32,
        0.0289038327_f32,
        -0.0309199896_f32,
        -0.0066980557_f32,
        -0.0790187493_f32,
        -0.0060427049_f32,
        0.0309803076_f32,
        -0.0529392287_f32,
        0.0047053015_f32,
        0.0112646613_f32,
        -0.0382907130_f32,
        0.0179422088_f32,
        -0.0174007025_f32,
        -0.0347030684_f32,
        0.0123951556_f32,
        -0.0697232783_f32,
        -0.0002798545_f32,
        0.0167765990_f32,
        -0.0190601312_f32,
        0.0116205774_f32,
        -0.0613930859_f32,
        -0.0603465550_f32,
        0.0563468710_f32,
        -0.0684550330_f32,
        -0.0055265957_f32,
        0.0107513666_f32,
        -0.0210899655_f32,
        0.0352863446_f32,
        -0.0177360978_f32,
        -0.0019518610_f32,
        -0.0165409222_f32,
        -0.0436665975_f32,
        0.0118507491_f32,
        -0.0116420630_f32,
        -0.0852299631_f32,
        0.0122249220_f32,
        0.0653364956_f32,
        -0.0093396613_f32,
        -0.0035199188_f32,
        -0.0327358842_f32,
        0.0207323693_f32,
        0.0023152898_f32,
        -0.0557320490_f32,
        -0.0395888314_f32,
        -0.0251897518_f32,
        -0.0132825747_f32,
        -0.0124528725_f32,
        0.0258471537_f32,
        0.0140378885_f32,
        -0.0087412028_f32,
        -0.0488211550_f32,
        0.0437809601_f32,
        -0.0247572251_f32,
        -0.0063129566_f32,
        -0.0474744849_f32,
        0.0117232250_f32,
        0.0154090263_f32,
        -0.0131737385_f32,
        0.0173936319_f32,
        -0.0572218858_f32,
        0.0344957300_f32,
        -0.0216700062_f32,
        0.0308493115_f32,
        -0.0065424126_f32,
        0.0343831293_f32,
        0.0185811911_f32,
        0.0068767141_f32,
        0.0249609239_f32,
        0.0412660353_f32,
        -0.0116225323_f32,
        -0.0259699002_f32,
        0.0206255689_f32,
        -0.0270097423_f32,
        -0.0056549036_f32,
        -0.0390801206_f32,
        -0.0405718498_f32,
        0.0137586007_f32,
        -0.0519951321_f32,
        0.0354694799_f32,
        0.0391143076_f32,
        -0.0010788258_f32,
        -0.0011020501_f32,
        -0.0290064998_f32,
        -0.0401471555_f32,
        0.0078720888_f32,
        -0.0550649725_f32,
        0.0333396904_f32,
        -0.0108824773_f32,
        -0.0109510338_f32,
        0.0397231877_f32,
        0.0083369985_f32,
        -0.0108747808_f32,
        0.0071907807_f32,
        -0.0086324196_f32,
        -0.0369156748_f32,
        0.0612906478_f32,
        -0.0018915223_f32,
        0.0385500565_f32,
        -0.0446332805_f32,
        -0.0351708196_f32,
        -0.0279752649_f32,
        0.0054018199_f32,
        -0.0057235253_f32,
        0.0195085984_f32,
        0.0421885923_f32,
        0.0000423958_f32,
        -0.0195831787_f32,
        0.0102244532_f32,
        0.0571847409_f32,
        -0.0259760730_f32,
        0.0295138843_f32,
        -0.0153622786_f32,
        -0.0077271671_f32,
        0.0313061886_f32,
        0.0467751883_f32,
        -0.0193835739_f32,
        0.0416923426_f32,
        0.0275301356_f32,
        0.0307814777_f32,
        0.0045230552_f32,
        0.0041383049_f32,
        0.0208784752_f32,
        -0.0012029837_f32,
        -0.0443958864_f32,
        0.0233979411_f32,
        0.0501711965_f32,
        0.0338188298_f32,
        0.0178568345_f32,
        0.0013898745_f32,
        0.0052047535_f32,
        -0.0558661558_f32,
        -0.0252277013_f32,
        0.0515839420_f32,
        -0.0451264642_f32,
        -0.0365451612_f32,
        -0.0329367071_f32,
        0.0141478758_f32,
        0.0950441062_f32,
        0.0126095479_f32,
        0.0234590974_f32,
        -0.0134304231_f32,
        -0.0741023496_f32,
        -0.0814688504_f32,
        0.0318291187_f32,
        -0.0013971514_f32,
        -0.0028411371_f32,
        -0.0265494548_f32,
        -0.0022464227_f32,
        0.1046569347_f32,
        0.0237788390_f32,
        0.0524426214_f32,
        -0.0442640446_f32,
        0.0430316627_f32,
        0.0165063255_f32,
        -0.0017553294_f32,
        -0.0042480007_f32,
        -0.0218351278_f32,
        0.0536996908_f32,
        -0.0330546163_f32,
        -0.0348458551_f32,
        0.0583651550_f32,
        0.0077706990_f32,
        -0.0121885100_f32,
        -0.0936391801_f32,
        -0.0501868315_f32,
        0.0376795158_f32,
        0.0484665595_f32,
        0.0117591713_f32,
        0.0082642557_f32,
        0.0034672872_f32,
        0.0373109542_f32,
        -0.0035247321_f32,
        -0.0243354142_f32,
        0.0160434544_f32,
        -0.0315834843_f32,
        -0.0223951582_f32,
        0.0647642389_f32,
        0.0666939020_f32,
        0.0526669733_f32,
        0.0196223669_f32,
        -0.0170642696_f32,
        0.0289506745_f32,
        -0.0093188602_f32,
        -0.0353710055_f32,
        0.0813130885_f32,
        -0.0029282314_f32,
        -0.0017046690_f32,
        -0.0002407116_f32,
        0.0304148588_f32,
        0.0187808126_f32,
        0.0560825802_f32,
        0.0203695055_f32,
        -0.0211434439_f32,
        -0.0053686691_f32,
        0.0065609124_f32,
        -0.0800422132_f32,
        -0.0453421883_f32,
        -0.0347995721_f32,
        0.0304125026_f32,
        -0.0096997684_f32,
        0.0048060226_f32,
        -0.0169816259_f32,
        -0.0071039763_f32,
        0.0571166538_f32,
        0.0647602379_f32,
        0.0276449211_f32,
        -0.0282668825_f32,
        -0.0103718610_f32,
        -0.0119420951_f32,
        -0.0242404863_f32,
        -0.1171171814_f32,
        -0.0315988474_f32,
        -0.0140225254_f32,
        -0.0057676076_f32,
        -0.0603364594_f32,
        0.0309729688_f32,
        0.0388718806_f32,
        -0.0193189122_f32,
        -0.0084467148_f32,
        -0.0095646363_f32,
        -0.0261842087_f32,
        0.0403973944_f32,
        -0.0037309420_f32,
        0.0043406002_f32,
        -0.0360655524_f32,
        -0.0368092507_f32,
        -0.0005705639_f32,
        -0.0121975075_f32,
        0.0335097089_f32,
        0.0216561351_f32,
        -0.0508602187_f32,
        0.0141279260_f32,
        -0.0136921937_f32,
        -0.0283461977_f32,
        -0.0306766089_f32,
        -0.0164569207_f32,
        -0.0149321901_f32,
        -0.0038134928_f32,
        0.0007168623_f32,
        -0.0426164754_f32,
        0.0438801534_f32,
        0.0708057210_f32,
        -0.0098242732_f32,
        -0.0544609427_f32,
        -0.0710762218_f32,
        -0.0127210310_f32,
        -0.0630207509_f32,
        -0.0639330149_f32,
        -0.0307161473_f32,
        -0.0179537069_f32,
        -0.0668733343_f32,
        -0.0120313810_f32,
        0.0088354526_f32,
        0.0126882093_f32,
        -0.0089736180_f32,
        0.0108480109_f32,
        0.0256611798_f32,
        0.0389979184_f32,
        0.0238422453_f32,
        -0.0020766330_f32,
        -0.0525760464_f32,
        0.0227198601_f32,
        0.0496796258_f32,
        0.0425307713_f32,
        0.0073482790_f32,
        0.0452583469_f32,
        -0.0163488258_f32,
        -0.0189230312_f32,
        0.0335791148_f32,
        -0.0192263424_f32,
        0.0400293879_f32,
        -0.0211833008_f32,
        0.0340786651_f32,
        -0.0034159208_f32,
        0.0211050548_f32,
        0.0172344800_f32,
        0.0346142165_f32,
        0.0225436352_f32,
        0.0703434572_f32,
        -0.0380113199_f32,
        -0.0042673964_f32,
        -0.0209812783_f32,
        -0.0043536816_f32,
        0.0609596223_f32,
        0.0672737658_f32,
        -0.0464234948_f32,
        -0.0179886594_f32,
        0.0153868943_f32,
        0.0000911963_f32,
        -0.0062587666_f32,
        -0.0334434807_f32,
        -0.0209111422_f32,
        -0.0200576093_f32,
        0.0578173660_f32,
        0.0051891198_f32,
        0.0114627536_f32,
        0.0001187448_f32,
        0.0880648270_f32,
        0.0077517116_f32,
        0.0271183886_f32,
        0.0073681027_f32,
        -0.0262809228_f32,
        -0.0066821994_f32,
        0.0034998646_f32,
        0.0045400597_f32,
        -0.0057278085_f32,
        0.0029020247_f32,
        0.0101336772_f32,
        -0.0070087584_f32,
        -0.0065972023_f32,
        0.0763970241_f32,
        -0.0360361934_f32,
        -0.0076653850_f32,
        0.0214485116_f32,
        0.0160965063_f32,
        -0.0311948638_f32,
        -0.0094292825_f32,
        -0.0526522174_f32,
        0.0425017513_f32,
        0.0221658126_f32,
        -0.0629199669_f32,
        -0.0303492695_f32,
        0.0466585271_f32,
        0.0388836823_f32,
        0.0181745645_f32,
        0.0371281244_f32,
        0.0174077023_f32,
        0.0447816029_f32,
        -0.0370219871_f32,
        -0.0359658487_f32,
        -0.0185491368_f32,
        -0.0412500761_f32,
        0.0063349381_f32,
        0.0427848622_f32,
        0.0295833126_f32,
        0.0197014287_f32,
        0.0052501792_f32,
        0.0436316915_f32,
        -0.0756947622_f32,
        0.0179658718_f32,
        -0.0394487903_f32,
        0.0072921910_f32,
        0.0533360094_f32,
        -0.0340586826_f32,
        -0.0112496754_f32,
        0.0151719358_f32,
        -0.0251951907_f32,
        0.0022928996_f32,
        -0.0290537700_f32,
        -0.0012770068_f32,
        -0.0051372452_f32,
        -0.0325374492_f32,
        0.0433720984_f32,
        -0.0252139401_f32,
        -0.0154475244_f32,
        0.0223646164_f32,
        0.0187732577_f32,
        0.0398150682_f32,
        0.0235319529_f32,
        -0.0302434713_f32,
        -0.0266864169_f32,
        0.0052925628_f32,
        -0.0050906953_f32,
        0.0018470233_f32,
        -0.0366911925_f32,
        -0.0301496796_f32,
        -0.0131648351_f32,
        -0.0009706095_f32,
        -0.0435727946_f32,
        -0.0249792524_f32,
        0.0804109126_f32,
        -0.0381668359_f32,
        0.0098564653_f32,
        0.0318424739_f32,
        0.0146656251_f32,
        0.0344300680_f32,
        0.0413681045_f32,
        0.0568229370_f32,
        -0.0061033913_f32,
        0.0059281327_f32,
        -0.0351187997_f32,
        -0.0323671512_f32,
        0.0297416337_f32,
        -0.0051261541_f32,
        -0.0330150016_f32,
        0.0356224850_f32,
        0.0307823904_f32,
        0.0577314757_f32,
        -0.0217848439_f32,
        -0.0545450784_f32,
        -0.0367976278_f32,
        0.0671850517_f32,
        -0.0508064069_f32,
        0.0551236160_f32,
        -0.0001699143_f32,
        -0.0608908720_f32,
        -0.0545773841_f32,
        0.0682890415_f32,
        -0.0153779406_f32,
        0.0320642740_f32,
        0.0020752859_f32,
        0.0418952294_f32,
        0.0102206748_f32,
        -0.0818082616_f32,
        0.0267092511_f32,
        -0.0403671525_f32,
        0.0368185490_f32,
        0.0372849330_f32,
        -0.0342242606_f32,
        -0.0232296214_f32,
        0.0254651438_f32,
        -0.0217820331_f32,
        0.0110614793_f32,
        0.0327017158_f32,
        0.0112188859_f32,
        0.0027090563_f32,
        -0.0459778756_f32,
        -0.0456320271_f32,
        0.0087764421_f32,
        -0.0257891063_f32,
        -0.0209431928_f32,
        -0.0271354932_f32,
        0.0178186614_f32,
        -0.0104626734_f32,
        0.0045214882_f32,
        -0.0517822132_f32,
        0.0326162875_f32,
        -0.0347503051_f32,
        -0.0226543508_f32,
        -0.0467755608_f32,
        -0.0002184615_f32,
        0.0240168739_f32,
        0.0032695238_f32,
        0.0244333521_f32,
        0.0710149556_f32,
        -0.0086221360_f32,
        0.0296954680_f32,
        -0.0439739637_f32,
        -0.0236670412_f32,
        -0.0068820291_f32,
        -0.0537502021_f32,
        0.0458157100_f32,
        0.0196417961_f32,
        0.0410779826_f32,
        -0.0454083271_f32,
        -0.0299954731_f32,
        -0.0196166728_f32,
        -0.0378240719_f32,
        0.0298517663_f32,
        -0.0280734971_f32,
        0.0461666211_f32,
        -0.0216104854_f32,
        0.0238105431_f32,
        -0.0064651021_f32,
        -0.0405022092_f32,
        0.0038920550_f32,
        -0.0353369564_f32,
        0.0205433872_f32,
        0.0440057144_f32,
        0.0451249480_f32,
        0.0741508901_f32,
        -0.0074251634_f32,
        -0.0344496295_f32,
        0.0818637535_f32,
        -0.0048967344_f32,
        0.0731848031_f32,
        -0.0473797321_f32,
        -0.0052646450_f32,
        -0.0601153336_f32,
        0.0451815799_f32,
        0.0031517900_f32,
        -0.0350292213_f32,
        -0.0159121659_f32,
        0.0583671965_f32,
        -0.0371118039_f32,
        -0.0606880598_f32,
        0.0832339972_f32,
        0.0316735432_f32,
        0.0528854206_f32,
        -0.0060036699_f32,
        0.0036519414_f32,
        0.0446038395_f32,
        0.0032855207_f32,
        -0.0319204666_f32,
        0.0065348689_f32,
        0.0034444691_f32,
        -0.0408955850_f32,
        -0.0047828350_f32,
        -0.0085702287_f32,
        0.0484830663_f32,
        0.0616147555_f32,
        -0.0356446914_f32,
        0.0781186819_f32,
        -0.0226362273_f32,
        0.0355184190_f32,
        -0.0047524050_f32,
        -0.0157734845_f32,
        -0.0508445464_f32,
        -0.0413894132_f32,
        0.0033911157_f32,
        -0.0042604902_f32,
        0.0723273084_f32,
        0.0118826758_f32,
        0.0091063464_f32,
        0.0352638476_f32,
        0.0095249955_f32,
        -0.0114119258_f32,
        -0.0340403989_f32,
        -0.0343567021_f32,
        0.0087953275_f32,
        0.0605385415_f32,
        -0.0095503833_f32,
        -0.0049547115_f32,
        0.0974564403_f32,
        -0.0180110317_f32,
        0.0499243215_f32,
        -0.0001960109_f32,
        -0.0194298811_f32,
        0.0292168446_f32,
        0.0347519666_f32,
        -0.0218126420_f32,
        -0.0031604662_f32,
        -0.0128528848_f32,
        0.0246979948_f32,
        -0.0687308833_f32,
        -0.0578336120_f32,
        0.0389770754_f32,
        -0.0357174017_f32,
        0.0279909298_f32,
        -0.0456140265_f32,
        0.0288375095_f32,
        -0.0194925535_f32,
        0.0198612232_f32,
        0.0423892699_f32,
        0.0510963388_f32,
        0.0096547864_f32,
        0.0085514467_f32,
        -0.0309314840_f32,
        -0.0395637825_f32,
        -0.0397309884_f32,
        0.0656967834_f32,
        -0.0064054234_f32,
        -0.0016942231_f32,
        0.0329678692_f32,
        0.0113510098_f32,
        -0.0117815807_f32,
        -0.0020475187_f32,
        0.0015729807_f32,
        -0.0952866226_f32,
    ];

    /// Flip to `true` in the same commit that populates DOC_REFERENCE_EMBEDDING with real values.
    /// See TASK-1.14.2.1.
    const DOC_REFERENCE_EMBEDDING_POPULATED: bool = true;

    const QUERY_TEST_INPUT: &str = "task: search result | query: hello world";
    const DOC_TEST_INPUT: &str = "title: My Note | text: hello world";

    /// Mirrors embed.rs::inner_embed_text (tokenize → pad → forward → mean_pooling → projection → normalize_embedding).
    /// Keep the two in sync — this duplication is what let the missing normalize_embedding call go unnoticed.
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
        let padded = truncate_and_pad(&token_ids, max_len, pad_id);

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

        // Mean pool + project — use pad_tensor so padding positions are excluded from the average.
        let pooled = mean_pooling(&hidden_states, &pad_tensor).expect("Mean pooling failed");
        let projected = loaded
            .projection_head
            .forward(&pooled)
            .expect("Projection failed");

        let embedding = projected.squeeze(0).unwrap();
        let raw: Vec<f32> = embedding.to_dtype(DType::F32).unwrap().to_vec1().unwrap();
        normalize_embedding(&raw, 768)
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
        if REFERENCE_EMBEDDING_POPULATED {
            assert_matches_reference(&embedding, REFERENCE_EMBEDDING, "Query embedding");
        } else {
            eprintln!(
                "Query embedding: shape/dim/norm verified, but REFERENCE_EMBEDDING not populated. \
                 Populate REFERENCE_EMBEDDING and set REFERENCE_EMBEDDING_POPULATED = true to enable numerical validation."
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
        if DOC_REFERENCE_EMBEDDING_POPULATED {
            assert_matches_reference(&embedding, DOC_REFERENCE_EMBEDDING, "Document embedding");
        } else {
            eprintln!(
                "Document embedding: shape/dim/norm verified, but DOC_REFERENCE_EMBEDDING not populated. \
                 Populate DOC_REFERENCE_EMBEDDING and set DOC_REFERENCE_EMBEDDING_POPULATED = true to enable numerical validation."
            );
        }
    }
}
