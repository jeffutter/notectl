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
    const REFERENCE_EMBEDDING: &[f32] = &[
        -0.009387283_f32,
        0.1003272_f32,
        -0.001408969_f32,
        -0.04727069_f32,
        -0.05269144_f32,
        -0.01000305_f32,
        -0.05631662_f32,
        -0.002653417_f32,
        -0.002911898_f32,
        -0.04407301_f32,
        0.01510749_f32,
        -0.03767685_f32,
        -0.01016652_f32,
        0.04248362_f32,
        -0.03516319_f32,
        0.001342048_f32,
        -0.04869314_f32,
        0.0441334_f32,
        -0.01886183_f32,
        -0.04666124_f32,
        0.0190265_f32,
        0.05557097_f32,
        -0.005080205_f32,
        0.0476488_f32,
        0.00940904_f32,
        0.004354999_f32,
        -0.008523094_f32,
        0.01323501_f32,
        0.03083036_f32,
        0.08348936_f32,
        -0.02452639_f32,
        0.009247088_f32,
        0.04012721_f32,
        -0.04488796_f32,
        0.0190335_f32,
        -0.006356229_f32,
        -0.02670317_f32,
        0.008009323_f32,
        -0.03520135_f32,
        0.0177848_f32,
        -0.007311206_f32,
        -0.008497435_f32,
        -0.03943941_f32,
        -0.007971776_f32,
        0.01315568_f32,
        -0.05204571_f32,
        -0.08467446_f32,
        0.03844177_f32,
        -0.0890376_f32,
        -0.05552482_f32,
        0.03819631_f32,
        -0.03539012_f32,
        0.05027108_f32,
        0.02271776_f32,
        0.001999309_f32,
        0.01559908_f32,
        -0.02999822_f32,
        -0.01421265_f32,
        0.009348596_f32,
        -0.02281335_f32,
        0.02259722_f32,
        0.02291621_f32,
        0.03141183_f32,
        0.003535946_f32,
        -0.003923822_f32,
        -0.0254006_f32,
        0.02664657_f32,
        0.001755284_f32,
        0.03186652_f32,
        0.01548636_f32,
        -0.03021394_f32,
        0.02123079_f32,
        0.02648736_f32,
        -0.03986008_f32,
        0.04005218_f32,
        0.002427758_f32,
        -05_f32,
        -0.02800699_f32,
        -0.0203195_f32,
        -0.007833839_f32,
        -0.03215616_f32,
        0.004976645_f32,
        0.03573656_f32,
        -0.006348455_f32,
        -0.002777244_f32,
        0.005861097_f32,
        0.04710305_f32,
        0.005138915_f32,
        -0.00458277_f32,
        0.02526001_f32,
        -0.0860272_f32,
        0.01713298_f32,
        -0.03998084_f32,
        -0.003453042_f32,
        -0.08325364_f32,
        0.02516209_f32,
        -0.0570096_f32,
        0.01025121_f32,
        -0.05183744_f32,
        -0.01124397_f32,
        0.03748974_f32,
        -0.01322172_f32,
        0.04283975_f32,
        0.004291963_f32,
        0.03365878_f32,
        0.02296804_f32,
        -0.007440132_f32,
        0.002261767_f32,
        0.01146755_f32,
        0.02808949_f32,
        0.001308312_f32,
        0.03916999_f32,
        -0.00722249_f32,
        0.008836751_f32,
        0.02535329_f32,
        -0.05432541_f32,
        -0.01001924_f32,
        0.02062872_f32,
        0.0329476_f32,
        0.04799073_f32,
        0.00602118_f32,
        -0.03496703_f32,
        -0.03300706_f32,
        0.01619542_f32,
        0.0562228_f32,
        0.06433208_f32,
        -0.04050802_f32,
        0.02567454_f32,
        0.05006218_f32,
        -0.08578398_f32,
        -0.01507849_f32,
        -0.001034846_f32,
        0.04169912_f32,
        0.0491619_f32,
        0.02502102_f32,
        0.003000085_f32,
        -0.05771391_f32,
        0.00636932_f32,
        -0.01659252_f32,
        -0.01109723_f32,
        -0.01363681_f32,
        0.05389859_f32,
        0.05425407_f32,
        0.02112505_f32,
        -0.01650783_f32,
        0.05480887_f32,
        -0.004456712_f32,
        -0.04262421_f32,
        -0.008857931_f32,
        -0.02287555_f32,
        -0.01466425_f32,
        -0.006790562_f32,
        0.05493652_f32,
        0.001510634_f32,
        -0.05922988_f32,
        0.0212409_f32,
        -0.05005839_f32,
        -0.01834513_f32,
        0.01308339_f32,
        -0.04637529_f32,
        0.04565703_f32,
        0.005777786_f32,
        0.06089145_f32,
        -0.009863731_f32,
        0.01015094_f32,
        0.03551321_f32,
        -0.03627618_f32,
        0.006192591_f32,
        -0.03343144_f32,
        0.0846231_f32,
        -0.01258147_f32,
        0.03659784_f32,
        0.01126527_f32,
        -0.04305734_f32,
        0.0613214_f32,
        -0.008356406_f32,
        0.007045729_f32,
        -0.06386783_f32,
        0.02903442_f32,
        0.005035806_f32,
        0.0075853_f32,
        0.01025847_f32,
        0.01559734_f32,
        0.002194874_f32,
        0.01222157_f32,
        0.09620754_f32,
        0.04091284_f32,
        -0.02218596_f32,
        -0.01204612_f32,
        -0.02296321_f32,
        -0.008914375_f32,
        -0.009692644_f32,
        -0.009796167_f32,
        -0.05658529_f32,
        0.04895222_f32,
        0.03759798_f32,
        0.001992113_f32,
        0.05413611_f32,
        -0.01052497_f32,
        -0.02423978_f32,
        -0.04008347_f32,
        0.03937738_f32,
        0.02615542_f32,
        0.04206095_f32,
        -0.02218717_f32,
        0.06852622_f32,
        0.04168055_f32,
        -0.006186845_f32,
        -0.009094304_f32,
        0.02757176_f32,
        0.02185493_f32,
        0.02517505_f32,
        -0.03943041_f32,
        0.04792105_f32,
        0.04422121_f32,
        0.02525596_f32,
        -0.04072668_f32,
        -0.002124009_f32,
        0.07273915_f32,
        0.003236448_f32,
        0.01983036_f32,
        0.01034498_f32,
        0.007772848_f32,
        0.05960505_f32,
        0.01606082_f32,
        -0.02561813_f32,
        -0.02006378_f32,
        -0.007392524_f32,
        -0.06387596_f32,
        -0.03386296_f32,
        0.02215689_f32,
        -0.0423396_f32,
        -0.0134949_f32,
        0.04102491_f32,
        -0.06265254_f32,
        0.005409502_f32,
        -0.00597223_f32,
        -0.01800115_f32,
        -0.04190419_f32,
        -0.05680849_f32,
        -0.005618312_f32,
        0.02433626_f32,
        -0.01183822_f32,
        0.01957027_f32,
        0.001571455_f32,
        -0.03453865_f32,
        -0.04502232_f32,
        0.03483636_f32,
        -0.0006951341_f32,
        -0.0007558682_f32,
        0.03062775_f32,
        0.03167564_f32,
        0.04061257_f32,
        -0.03289988_f32,
        0.0147487_f32,
        0.02482375_f32,
        -0.08034187_f32,
        -0.01542332_f32,
        -0.02948867_f32,
        0.01511244_f32,
        -0.05524963_f32,
        0.005663176_f32,
        -0.03219634_f32,
        0.03340004_f32,
        0.0009710671_f32,
        -0.03688116_f32,
        0.01950261_f32,
        0.01576511_f32,
        0.01579075_f32,
        -0.03265431_f32,
        0.0640065_f32,
        0.02357869_f32,
        0.04473493_f32,
        -0.0301307_f32,
        -0.01046954_f32,
        0.05919404_f32,
        -0.06266709_f32,
        0.01849922_f32,
        -0.0004986649_f32,
        0.01708979_f32,
        0.04204587_f32,
        0.06551727_f32,
        -0.05682182_f32,
        -0.007531557_f32,
        0.01336985_f32,
        0.04965443_f32,
        -0.003137861_f32,
        0.04877146_f32,
        0.01044274_f32,
        -0.01012869_f32,
        0.03384365_f32,
        -0.03554305_f32,
        0.01752439_f32,
        0.03794385_f32,
        0.00303679_f32,
        -0.02737666_f32,
        0.0223214_f32,
        -0.02542709_f32,
        0.03552176_f32,
        0.08046725_f32,
        -0.001560916_f32,
        -0.02257202_f32,
        -0.04628078_f32,
        0.01742916_f32,
        -0.04927779_f32,
        -0.006694384_f32,
        -0.01448947_f32,
        0.03475615_f32,
        -0.07233226_f32,
        0.0003400468_f32,
        0.007017854_f32,
        0.0004581976_f32,
        0.04910825_f32,
        -0.01001583_f32,
        -0.01418375_f32,
        0.05372908_f32,
        0.0475738_f32,
        -0.04052911_f32,
        -0.02799263_f32,
        0.04758968_f32,
        0.04023708_f32,
        -0.02330609_f32,
        -0.01053969_f32,
        -0.02214227_f32,
        -0.02020463_f32,
        0.01242026_f32,
        -0.02119241_f32,
        -0.007695303_f32,
        -0.01269762_f32,
        0.07534089_f32,
        -0.05309258_f32,
        0.02898297_f32,
        -0.01041325_f32,
        0.01682643_f32,
        0.06742418_f32,
        -0.001612294_f32,
        -0.03827522_f32,
        -0.007644562_f32,
        0.0132303_f32,
        -0.01544807_f32,
        -0.01129217_f32,
        0.05820474_f32,
        -0.02006635_f32,
        0.03126169_f32,
        -0.0004017536_f32,
        0.01921813_f32,
        -0.002353925_f32,
        -0.01127999_f32,
        0.02042528_f32,
        -0.04859063_f32,
        0.04712617_f32,
        0.02013305_f32,
        0.02682396_f32,
        -0.03368358_f32,
        0.01971501_f32,
        -0.05208495_f32,
        0.04452707_f32,
        0.04593318_f32,
        -0.01948297_f32,
        0.09028567_f32,
        -0.01060565_f32,
        -0.01254193_f32,
        0.07766731_f32,
        0.001033503_f32,
        0.02561118_f32,
        -0.01322506_f32,
        -0.004310323_f32,
        0.01251161_f32,
        -0.02833728_f32,
        0.003444256_f32,
        0.04189542_f32,
        -0.01048527_f32,
        0.008742715_f32,
        0.01414933_f32,
        0.0155014_f32,
        -0.0004144317_f32,
        0.01922764_f32,
        -0.03584737_f32,
        0.0174394_f32,
        0.008810137_f32,
        -0.06838254_f32,
        -0.03373274_f32,
        -0.04283938_f32,
        0.0480486_f32,
        -0.01794283_f32,
        0.01277815_f32,
        0.002916954_f32,
        -0.04101056_f32,
        -0.04871727_f32,
        0.0002292767_f32,
        -0.01267805_f32,
        0.01265486_f32,
        -0.00164007_f32,
        0.01222269_f32,
        0.03468375_f32,
        0.03253278_f32,
        -0.02560527_f32,
        -0.05760404_f32,
        -0.009129452_f32,
        0.01737183_f32,
        0.08365452_f32,
        -0.01953901_f32,
        0.01389643_f32,
        0.001547613_f32,
        0.04555926_f32,
        -0.01497141_f32,
        0.002554983_f32,
        0.04407789_f32,
        0.003136573_f32,
        -0.00540879_f32,
        -0.03102072_f32,
        -0.003303679_f32,
        -0.01927494_f32,
        0.02525933_f32,
        0.01832384_f32,
        -0.04030666_f32,
        -0.0484204_f32,
        0.03751879_f32,
        -0.01080793_f32,
        -0.01863428_f32,
        0.009679255_f32,
        0.002686211_f32,
        0.05944125_f32,
        0.01026107_f32,
        -0.003177096_f32,
        0.06080708_f32,
        -0.02909586_f32,
        -0.03073768_f32,
        -0.04297707_f32,
        -0.01487972_f32,
        -0.04842283_f32,
        0.0001231405_f32,
        -0.02509807_f32,
        -0.02233247_f32,
        0.01846197_f32,
        0.0582523_f32,
        -0.04633661_f32,
        -0.01817959_f32,
        0.006631931_f32,
        -0.01299172_f32,
        0.0507915_f32,
        -0.04465953_f32,
        0.007621598_f32,
        -0.0120023_f32,
        -0.09901508_f32,
        -0.008387623_f32,
        -0.006772518_f32,
        -0.03302187_f32,
        0.0005826238_f32,
        -0.03776729_f32,
        -0.05150512_f32,
        -0.07621162_f32,
        0.01761834_f32,
        0.003312294_f32,
        0.03591512_f32,
        -0.02398071_f32,
        -0.03638955_f32,
        0.03820057_f32,
        -0.04253893_f32,
        0.01920278_f32,
        0.003371009_f32,
        0.00236763_f32,
        -0.02806307_f32,
        0.04747556_f32,
        0.02465886_f32,
        -0.05493936_f32,
        -0.02110324_f32,
        0.05288035_f32,
        0.003240043_f32,
        -0.02514898_f32,
        -0.008218847_f32,
        0.01048863_f32,
        0.01129923_f32,
        0.02586563_f32,
        -0.03234525_f32,
        0.07001255_f32,
        -0.0271834_f32,
        0.01633561_f32,
        -0.06453326_f32,
        -0.01649022_f32,
        0.03842201_f32,
        -0.05755334_f32,
        -0.003540752_f32,
        0.004291493_f32,
        0.04708901_f32,
        -0.01803974_f32,
        0.01957802_f32,
        -0.01900386_f32,
        -0.0349564_f32,
        0.05923767_f32,
        0.04704468_f32,
        -0.05645272_f32,
        -0.0171329_f32,
        0.07997366_f32,
        0.005112482_f32,
        -0.01422484_f32,
        0.01633163_f32,
        0.0009352049_f32,
        0.002308554_f32,
        0.06619319_f32,
        0.03878155_f32,
        0.03500711_f32,
        0.0621733_f32,
        0.0142348_f32,
        -0.01945717_f32,
        -0.02776596_f32,
        0.01540505_f32,
        -0.004360685_f32,
        0.0209587_f32,
        -0.06188614_f32,
        0.01635042_f32,
        -0.007201702_f32,
        -0.006444342_f32,
        -0.0391016_f32,
        0.03284902_f32,
        0.01526555_f32,
        -0.02566709_f32,
        0.009855236_f32,
        -0.01149652_f32,
        -0.03064354_f32,
        0.05781186_f32,
        0.03810487_f32,
        0.001008726_f32,
        -0.02794946_f32,
        -0.01260433_f32,
        0.001919381_f32,
        -0.07101204_f32,
        -0.07785653_f32,
        0.01949517_f32,
        0.01687733_f32,
        0.05755402_f32,
        0.05144483_f32,
        -0.04259902_f32,
        0.0260778_f32,
        0.01833866_f32,
        -0.003258558_f32,
        -0.02531791_f32,
        -0.03848374_f32,
        0.09359705_f32,
        0.01939352_f32,
        0.05923245_f32,
        -0.0206724_f32,
        0.041793_f32,
        0.03218723_f32,
        -0.04904022_f32,
        -0.03172_f32,
        -0.04975377_f32,
        -0.01548924_f32,
        0.008381644_f32,
        0.06784344_f32,
        0.004844185_f32,
        0.08215449_f32,
        0.04813126_f32,
        0.01935386_f32,
        -0.0001561924_f32,
        0.01379765_f32,
        0.03407921_f32,
        0.02641715_f32,
        -0.03139948_f32,
        0.05202304_f32,
        0.01272613_f32,
        0.07582947_f32,
        0.006810555_f32,
        0.02504929_f32,
        -0.01795899_f32,
        -0.002851255_f32,
        0.01798614_f32,
        -0.05759474_f32,
        -0.0007895152_f32,
        0.03492074_f32,
        0.03447358_f32,
        -0.03546233_f32,
        0.03744078_f32,
        0.02984242_f32,
        0.03147831_f32,
        -0.02451405_f32,
        0.06302737_f32,
        0.01224983_f32,
        -0.06533242_f32,
        0.01203248_f32,
        -0.04813792_f32,
        -0.01525185_f32,
        -0.003410567_f32,
        0.05118878_f32,
        -0.01916118_f32,
        -0.03908219_f32,
        -0.03133529_f32,
        -0.05749057_f32,
        0.02743761_f32,
        0.03430684_f32,
        -0.02500325_f32,
        0.009173826_f32,
        -0.02333258_f32,
        0.02455272_f32,
        0.03257999_f32,
        -0.04536623_f32,
        0.08435061_f32,
        -0.005646556_f32,
        0.03636803_f32,
        -0.05885506_f32,
        -0.04601568_f32,
        0.009981089_f32,
        -0.02156383_f32,
        -0.01400608_f32,
        0.06492942_f32,
        -0.04561555_f32,
        -0.04083668_f32,
        0.05888156_f32,
        0.008585599_f32,
        0.04529525_f32,
        0.05683379_f32,
        -0.010185_f32,
        -0.02547612_f32,
        0.01701299_f32,
        -0.04786827_f32,
        -0.02202107_f32,
        0.05039497_f32,
        -0.04382799_f32,
        0.0337567_f32,
        0.01620026_f32,
        -0.007302741_f32,
        -0.001006381_f32,
        0.006833003_f32,
        0.08845851_f32,
        0.03427429_f32,
        0.02070984_f32,
        -0.03347035_f32,
        -0.03052402_f32,
        -0.04258522_f32,
        -0.02368085_f32,
        0.01973131_f32,
        -0.02201618_f32,
        0.05791451_f32,
        0.02493518_f32,
        0.01477322_f32,
        0.04544881_f32,
        0.006245792_f32,
        0.03188115_f32,
        -0.06563079_f32,
        -0.01167709_f32,
        0.01850458_f32,
        0.01484279_f32,
        0.0220631_f32,
        0.0007905056_f32,
        0.07661795_f32,
        -0.0329558_f32,
        0.008593341_f32,
        0.01263858_f32,
        -0.03983457_f32,
        0.01241168_f32,
        -0.03343256_f32,
        -0.03264391_f32,
        -0.0008958308_f32,
        -0.04687797_f32,
        -0.06814839_f32,
        -0.01063062_f32,
        -0.05754738_f32,
        -0.0548398_f32,
        0.07666486_f32,
        -0.01810761_f32,
        0.03223491_f32,
        0.00571495_f32,
        0.0230375_f32,
        -0.07379456_f32,
        -0.04851037_f32,
        0.01029158_f32,
        -0.03237773_f32,
        -0.005991394_f32,
        0.01826949_f32,
        0.002167833_f32,
        -0.01161504_f32,
        0.04108559_f32,
        0.002842698_f32,
        0.002848625_f32,
        0.00814566_f32,
        -0.01779242_f32,
        0.02401812_f32,
        0.01915985_f32,
        0.03583634_f32,
        0.04926412_f32,
        0.03454027_f32,
        0.04963868_f32,
        -0.01377969_f32,
        0.03527489_f32,
        -0.004074452_f32,
        -0.05153957_f32,
        0.001059314_f32,
        -0.0257748_f32,
        -0.004154548_f32,
        -0.037404_f32,
        0.03596321_f32,
        -0.0115447_f32,
        -0.07120298_f32,
        0.03222203_f32,
        0.003559698_f32,
        0.1206615_f32,
        -0.04074612_f32,
        0.001747087_f32,
        -0.01867156_f32,
        0.02405265_f32,
        -0.005579028_f32,
        -0.03268676_f32,
        0.09789545_f32,
        -0.01342189_f32,
        0.02805263_f32,
        0.003188239_f32,
        -0.03075006_f32,
        -0.07125178_f32,
        -0.00685726_f32,
        0.005497853_f32,
        0.03799636_f32,
        -0.0587412_f32,
        0.01451614_f32,
        0.006789378_f32,
        -0.03820381_f32,
        -0.01240052_f32,
        -0.02322431_f32,
        -0.008415996_f32,
        0.01134339_f32,
        -0.02187479_f32,
        -0.01938058_f32,
        0.03945347_f32,
        -0.02980585_f32,
        0.002958944_f32,
        0.01295486_f32,
        -0.01792146_f32,
        -0.1217296_f32,
        -0.05911661_f32,
        0.03519091_f32,
        0.01878627_f32,
        0.02290091_f32,
        -0.03642184_f32,
        -0.007394471_f32,
        0.05303485_f32,
        0.07848324_f32,
        -0.07232959_f32,
        0.005616345_f32,
        0.03411628_f32,
        -0.01754348_f32,
        -0.002313461_f32,
        -0.008912158_f32,
        -0.07196799_f32,
        -0.01863975_f32,
        -0.08024_f32,
        0.0267427_f32,
        0.03982915_f32,
        -0.02729116_f32,
        0.007318754_f32,
        -0.005750649_f32,
        -0.02774237_f32,
        0.03502136_f32,
        -0.02021024_f32,
        -0.01818373_f32,
        -0.04123318_f32,
        0.02830666_f32,
        0.001837337_f32,
        0.05185633_f32,
        0.02080581_f32,
        0.03019164_f32,
        0.05363796_f32,
        -0.04450494_f32,
        0.05350474_f32,
        -0.006361876_f32,
        0.03852979_f32,
        -0.03834278_f32,
        -0.01958576_f32,
        -0.04274444_f32,
        -0.07383515_f32,
    ];

    /// Flip to `true` in the same commit that populates REFERENCE_EMBEDDING with real values.
    /// See TASK-1.14.2.1.
    const REFERENCE_EMBEDDING_POPULATED: bool = true;

    /// Reference embedding for "title: My Note | text: hello world"
    /// (document-text prefix path).
    ///
    /// TODO: Populate from a TEI run.
    const DOC_REFERENCE_EMBEDDING: &[f32] = &[
        0.03298885_f32,
        0.08504353_f32,
        0.005409304_f32,
        -0.05360641_f32,
        -0.04760937_f32,
        -0.02072966_f32,
        -0.07264145_f32,
        0.0157651_f32,
        -0.02390459_f32,
        -0.03738891_f32,
        0.02131799_f32,
        -0.04600137_f32,
        -0.02601154_f32,
        0.0308832_f32,
        -0.02408395_f32,
        0.01558921_f32,
        -0.06899521_f32,
        0.05591991_f32,
        -0.03011991_f32,
        -0.03461309_f32,
        0.02313043_f32,
        0.01651142_f32,
        0.002448228_f32,
        0.05220399_f32,
        0.009674582_f32,
        0.009432918_f32,
        -0.03370695_f32,
        0.03488103_f32,
        0.03207203_f32,
        0.05383925_f32,
        -0.009722991_f32,
        0.01370787_f32,
        0.067292_f32,
        -0.01144961_f32,
        0.02776294_f32,
        -0.02451046_f32,
        -0.005837444_f32,
        -0.005772641_f32,
        -0.01641321_f32,
        0.03479034_f32,
        -0.02202185_f32,
        0.03327386_f32,
        -0.01364823_f32,
        -0.0001484411_f32,
        0.03714941_f32,
        -0.04730876_f32,
        -0.03906802_f32,
        0.00324663_f32,
        -0.05253753_f32,
        -0.04580182_f32,
        0.03408548_f32,
        -0.03989154_f32,
        0.009432903_f32,
        -0.009899512_f32,
        0.008497538_f32,
        0.02153162_f32,
        -0.006831984_f32,
        -0.0388381_f32,
        0.01108945_f32,
        -0.01573782_f32,
        0.004191261_f32,
        0.01526762_f32,
        0.03728921_f32,
        -0.02418102_f32,
        -0.01887091_f32,
        -0.0680586_f32,
        0.03589946_f32,
        0.01821474_f32,
        0.0368877_f32,
        -0.005825495_f32,
        0.02217907_f32,
        0.03427742_f32,
        -0.008869444_f32,
        -0.04024231_f32,
        0.02781928_f32,
        0.004948122_f32,
        0.0008644161_f32,
        -0.005897592_f32,
        -0.02463327_f32,
        -0.01605748_f32,
        -0.04011896_f32,
        0.006733921_f32,
        0.04579773_f32,
        0.0292575_f32,
        -0.0209749_f32,
        -0.02192856_f32,
        0.01030511_f32,
        0.002194279_f32,
        -0.01606515_f32,
        0.004328684_f32,
        -0.02547293_f32,
        0.01562355_f32,
        -0.01858272_f32,
        0.02330624_f32,
        -0.07410858_f32,
        -0.00892182_f32,
        -0.01682564_f32,
        -0.01424975_f32,
        -0.02274676_f32,
        -0.05154802_f32,
        0.02116986_f32,
        -0.05073296_f32,
        0.03620975_f32,
        0.02075029_f32,
        -0.04386406_f32,
        0.02658922_f32,
        -0.02579461_f32,
        -0.02273862_f32,
        0.003048506_f32,
        0.008311112_f32,
        -0.0005412212_f32,
        -0.007099692_f32,
        -0.02807399_f32,
        0.005134471_f32,
        0.08291606_f32,
        -0.03462512_f32,
        -0.02722277_f32,
        -0.0002341978_f32,
        0.05813951_f32,
        0.02100356_f32,
        -0.01773348_f32,
        0.002783264_f32,
        -0.02610185_f32,
        0.00334409_f32,
        0.02690807_f32,
        0.02583344_f32,
        -0.01039945_f32,
        -0.01584259_f32,
        0.028298_f32,
        -0.04600917_f32,
        -0.01827722_f32,
        -0.003531159_f32,
        0.02106665_f32,
        0.02389204_f32,
        0.04243384_f32,
        0.02769245_f32,
        -0.05279172_f32,
        0.009030899_f32,
        -0.03652508_f32,
        -0.01397733_f32,
        -0.03387657_f32,
        0.1034506_f32,
        0.04594946_f32,
        -0.005467275_f32,
        0.03363848_f32,
        0.02994567_f32,
        0.01300338_f32,
        -0.02576393_f32,
        0.01348759_f32,
        -0.002181191_f32,
        -0.031888_f32,
        0.0002798506_f32,
        0.06344722_f32,
        -0.03282828_f32,
        -0.03274037_f32,
        -0.005355858_f32,
        -0.03293418_f32,
        -0.02273037_f32,
        -0.003915818_f32,
        -0.07272469_f32,
        0.004124108_f32,
        -0.02518968_f32,
        0.01355497_f32,
        -0.0001933437_f32,
        0.01490262_f32,
        0.06688868_f32,
        -0.04276709_f32,
        0.01414825_f32,
        -0.02510134_f32,
        0.08405921_f32,
        -0.008034158_f32,
        0.06454333_f32,
        -0.02204489_f32,
        -0.03725078_f32,
        0.07031247_f32,
        -0.04773731_f32,
        0.03449696_f32,
        -0.06824727_f32,
        -0.002454224_f32,
        0.00107219_f32,
        -0.02916795_f32,
        0.02657233_f32,
        0.004126751_f32,
        0.02380842_f32,
        -0.0005891703_f32,
        0.1027814_f32,
        0.008136224_f32,
        0.001048349_f32,
        -0.02771226_f32,
        -0.04847239_f32,
        -0.03278174_f32,
        0.01835285_f32,
        -0.01775024_f32,
        -0.06788906_f32,
        0.08578081_f32,
        0.05272317_f32,
        -0.0359959_f32,
        -0.0067078_f32,
        -0.05432285_f32,
        -0.00625039_f32,
        -0.01891092_f32,
        0.01579144_f32,
        0.0240446_f32,
        0.01385024_f32,
        0.001473954_f32,
        0.01713974_f32,
        0.01713937_f32,
        -0.01922088_f32,
        -0.01525876_f32,
        0.03028041_f32,
        0.03592471_f32,
        0.02992153_f32,
        -0.01730832_f32,
        0.02475132_f32,
        0.01545614_f32,
        0.007742585_f32,
        -0.05281989_f32,
        -0.03576122_f32,
        0.05243232_f32,
        0.05442016_f32,
        0.008071058_f32,
        -0.01021884_f32,
        0.0473177_f32,
        0.05320034_f32,
        0.00980734_f32,
        0.005606453_f32,
        -0.01408571_f32,
        0.007635403_f32,
        -0.01113958_f32,
        0.004025795_f32,
        0.03844445_f32,
        -0.02641692_f32,
        -0.001341406_f32,
        0.05406936_f32,
        -0.04466586_f32,
        0.01013299_f32,
        -0.0009219837_f32,
        -0.000199123_f32,
        -0.01710797_f32,
        -0.06930275_f32,
        0.02269485_f32,
        -0.01916883_f32,
        -0.05573319_f32,
        0.04522685_f32,
        0.03743002_f32,
        0.008199292_f32,
        -0.02202498_f32,
        0.01226868_f32,
        0.03345314_f32,
        0.03202842_f32,
        0.03876692_f32,
        0.009159213_f32,
        0.02797752_f32,
        -0.007473356_f32,
        -0.005466012_f32,
        0.03391986_f32,
        -0.05841483_f32,
        -0.05509263_f32,
        0.001048154_f32,
        0.02685495_f32,
        -0.06011143_f32,
        0.007439707_f32,
        -0.04566043_f32,
        0.02379112_f32,
        0.02357075_f32,
        -0.009135831_f32,
        0.02338986_f32,
        -0.004771918_f32,
        -0.001172057_f32,
        -0.05481772_f32,
        0.04506938_f32,
        0.05557577_f32,
        0.07161909_f32,
        -0.05584173_f32,
        -0.01955809_f32,
        0.02730392_f32,
        -0.04714322_f32,
        -0.03203394_f32,
        0.05135498_f32,
        0.0143656_f32,
        0.00257775_f32,
        0.0640774_f32,
        -0.09092195_f32,
        -0.0317947_f32,
        -0.01749204_f32,
        0.06167226_f32,
        0.01996922_f32,
        0.05486933_f32,
        -0.005548591_f32,
        0.0037908_f32,
        0.02616359_f32,
        -0.06324884_f32,
        -0.0001258438_f32,
        0.0420137_f32,
        0.001743984_f32,
        -0.03881981_f32,
        0.04192012_f32,
        0.008630056_f32,
        0.01812473_f32,
        0.04738048_f32,
        -0.004929057_f32,
        -0.0293318_f32,
        -0.002651773_f32,
        0.03818765_f32,
        -0.02748846_f32,
        -0.002544736_f32,
        -0.02100482_f32,
        0.02093425_f32,
        -0.04682283_f32,
        -0.0004075473_f32,
        -0.007130792_f32,
        -0.02761012_f32,
        0.04246717_f32,
        -0.01825016_f32,
        0.005675658_f32,
        0.09810246_f32,
        0.03536212_f32,
        -0.01745332_f32,
        -0.02190833_f32,
        0.06023136_f32,
        0.03223424_f32,
        -0.03236619_f32,
        -0.005433657_f32,
        -0.02559867_f32,
        -0.007763698_f32,
        0.02312704_f32,
        -0.004020681_f32,
        -0.01167369_f32,
        0.02131401_f32,
        0.07735661_f32,
        -0.02852925_f32,
        0.02820079_f32,
        -0.04182177_f32,
        0.001401619_f32,
        0.0591036_f32,
        -0.02803239_f32,
        -0.01644254_f32,
        -0.01407815_f32,
        -0.02178466_f32,
        -0.01800454_f32,
        0.02077806_f32,
        0.07820977_f32,
        -0.005526855_f32,
        0.003796975_f32,
        -0.01196261_f32,
        0.02057824_f32,
        -0.02298411_f32,
        0.002738106_f32,
        0.01793129_f32,
        -0.02483341_f32,
        0.02725892_f32,
        0.03082409_f32,
        0.02261373_f32,
        0.001985615_f32,
        0.03599403_f32,
        -0.08837572_f32,
        0.05355164_f32,
        0.03421442_f32,
        -0.02613307_f32,
        0.07581998_f32,
        0.00944469_f32,
        -0.03364836_f32,
        0.0598348_f32,
        -0.01757596_f32,
        0.004410585_f32,
        -0.005715308_f32,
        0.009859758_f32,
        -0.02001326_f32,
        -0.02109681_f32,
        0.007583643_f32,
        0.03232938_f32,
        -0.001986641_f32,
        0.02051379_f32,
        0.008039946_f32,
        0.04216485_f32,
        -0.004126173_f32,
        0.02821107_f32,
        -0.03507994_f32,
        -0.03599053_f32,
        0.03746654_f32,
        -0.01298901_f32,
        -0.02011521_f32,
        -0.01449763_f32,
        0.02336443_f32,
        -0.03825847_f32,
        0.002850929_f32,
        -0.01416655_f32,
        -0.04809424_f32,
        -0.04445614_f32,
        0.002454222_f32,
        -0.02436261_f32,
        -0.01978828_f32,
        -0.006788662_f32,
        -0.03390494_f32,
        0.02485782_f32,
        0.01388015_f32,
        0.03152552_f32,
        -0.0167366_f32,
        0.006720541_f32,
        0.0175158_f32,
        0.08686531_f32,
        -0.05788055_f32,
        0.003744387_f32,
        0.01130499_f32,
        0.0651242_f32,
        -0.0328577_f32,
        0.005972113_f32,
        0.006866844_f32,
        -0.007076013_f32,
        -0.02327882_f32,
        -0.02845873_f32,
        0.01317367_f32,
        -0.01857987_f32,
        0.05674567_f32,
        -0.01006965_f32,
        -0.06941085_f32,
        -0.01051877_f32,
        0.02827524_f32,
        0.01340816_f32,
        0.008484357_f32,
        -0.0002541694_f32,
        0.0297343_f32,
        0.06159651_f32,
        -0.01070618_f32,
        0.02804515_f32,
        0.01356912_f32,
        -0.04139289_f32,
        -0.04896668_f32,
        -0.01632161_f32,
        -0.03485602_f32,
        -0.07273488_f32,
        -0.002749879_f32,
        -0.0008499315_f32,
        -0.03438474_f32,
        0.05341367_f32,
        0.02308686_f32,
        -0.01148613_f32,
        -0.01285087_f32,
        0.00856942_f32,
        0.005132678_f32,
        0.05687413_f32,
        -0.04581695_f32,
        -0.007104096_f32,
        -0.02191187_f32,
        -0.0686515_f32,
        0.01319693_f32,
        0.01130799_f32,
        -0.03923808_f32,
        -0.01142467_f32,
        -0.01814188_f32,
        0.006588583_f32,
        -0.08326674_f32,
        -0.02798979_f32,
        -0.01001761_f32,
        0.02281427_f32,
        -0.01784558_f32,
        -0.05056562_f32,
        -0.02041186_f32,
        -0.0633845_f32,
        0.001167652_f32,
        0.02418204_f32,
        0.007755084_f32,
        -0.02153495_f32,
        0.04031768_f32,
        0.03770394_f32,
        -0.0994082_f32,
        -0.03034175_f32,
        0.09335733_f32,
        0.01659378_f32,
        -0.006266862_f32,
        -0.03255982_f32,
        0.02425046_f32,
        -0.03101374_f32,
        0.03919958_f32,
        -0.005436325_f32,
        0.01741483_f32,
        -0.06894474_f32,
        0.006958311_f32,
        -0.06065217_f32,
        0.02419171_f32,
        0.02990455_f32,
        -0.07779552_f32,
        0.02358434_f32,
        0.008501627_f32,
        0.04467615_f32,
        -0.02159071_f32,
        -0.02505391_f32,
        -0.001520254_f32,
        -0.01424038_f32,
        0.02308021_f32,
        0.02838735_f32,
        -0.01282342_f32,
        -0.01160214_f32,
        0.07253762_f32,
        0.02887766_f32,
        -0.02861997_f32,
        0.01852645_f32,
        -0.0244507_f32,
        -0.004522051_f32,
        0.06590468_f32,
        0.02469974_f32,
        0.02413962_f32,
        0.01259578_f32,
        0.01902038_f32,
        -0.02441409_f32,
        0.001079535_f32,
        0.02418116_f32,
        0.006608669_f32,
        -0.002668378_f32,
        -0.06938452_f32,
        0.03746971_f32,
        -0.009006792_f32,
        -0.008880321_f32,
        -0.0714215_f32,
        0.01694067_f32,
        0.01723768_f32,
        0.01084139_f32,
        -0.0532022_f32,
        -0.02478068_f32,
        -0.0398073_f32,
        0.04159571_f32,
        0.05336065_f32,
        -0.02970705_f32,
        -0.01062347_f32,
        -0.0164829_f32,
        -0.00439719_f32,
        -0.07883947_f32,
        -0.04653291_f32,
        -0.008185971_f32,
        0.04694235_f32,
        0.03421021_f32,
        0.04258105_f32,
        -0.05527602_f32,
        0.05906404_f32,
        0.03552713_f32,
        -0.0008703191_f32,
        -0.07005636_f32,
        -0.02602216_f32,
        0.08315278_f32,
        0.009991324_f32,
        0.05486335_f32,
        -0.05558323_f32,
        0.01973966_f32,
        0.02664637_f32,
        -0.05020294_f32,
        0.05156484_f32,
        -0.03336992_f32,
        -0.04957372_f32,
        0.01555644_f32,
        0.05841689_f32,
        -0.02355838_f32,
        0.07507984_f32,
        0.06236517_f32,
        -0.0162931_f32,
        0.03556701_f32,
        0.06210328_f32,
        0.03251264_f32,
        -0.008472504_f32,
        0.009207274_f32,
        0.03611444_f32,
        0.006532723_f32,
        0.0567883_f32,
        -0.004354754_f32,
        0.00832341_f32,
        -0.02590567_f32,
        0.01287133_f32,
        0.04349147_f32,
        0.006226789_f32,
        -0.03131353_f32,
        0.03694969_f32,
        0.01368257_f32,
        -0.01780164_f32,
        0.09064665_f32,
        0.05421563_f32,
        -0.003554848_f32,
        -0.01629371_f32,
        0.04832613_f32,
        0.007112015_f32,
        -0.0466647_f32,
        0.03338393_f32,
        -0.005703985_f32,
        0.006781015_f32,
        0.02736677_f32,
        -0.01195893_f32,
        -0.008394252_f32,
        -0.01532076_f32,
        -0.0222978_f32,
        -0.01684594_f32,
        0.01743487_f32,
        0.06310833_f32,
        -0.001563059_f32,
        -0.02873456_f32,
        -0.02348063_f32,
        0.02355779_f32,
        0.02624376_f32,
        -0.01923935_f32,
        0.06016189_f32,
        -0.02616152_f32,
        0.004860053_f32,
        -0.09725253_f32,
        -0.0559187_f32,
        -0.03662929_f32,
        -0.009851489_f32,
        -0.02011828_f32,
        0.0562719_f32,
        -0.01298767_f32,
        -0.005504512_f32,
        0.04328026_f32,
        0.04084288_f32,
        0.01007504_f32,
        0.05129951_f32,
        -0.03663692_f32,
        -0.02914479_f32,
        0.02111489_f32,
        -0.006682743_f32,
        -0.01046759_f32,
        0.05830533_f32,
        0.01289094_f32,
        0.01216604_f32,
        0.03658019_f32,
        -0.008877174_f32,
        0.000489618_f32,
        -0.01237273_f32,
        0.06205193_f32,
        0.03388489_f32,
        -0.01994094_f32,
        0.01886608_f32,
        -0.02899295_f32,
        -0.05226934_f32,
        -0.01171591_f32,
        -0.01605975_f32,
        -0.06342228_f32,
        0.1050829_f32,
        0.01574123_f32,
        0.006751997_f32,
        0.06382489_f32,
        -0.01644593_f32,
        0.05047016_f32,
        -0.07857664_f32,
        -0.009163646_f32,
        -0.02173698_f32,
        0.05675134_f32,
        -0.008190345_f32,
        0.002783958_f32,
        0.01167404_f32,
        0.004338539_f32,
        0.008092747_f32,
        -0.0007446093_f32,
        -0.0360525_f32,
        0.01525615_f32,
        -0.006707134_f32,
        -0.01782565_f32,
        0.001321073_f32,
        -0.1052575_f32,
        -0.05698406_f32,
        -0.0354586_f32,
        -0.0438649_f32,
        -0.001585358_f32,
        0.06724697_f32,
        -0.01278089_f32,
        0.09059221_f32,
        0.01387244_f32,
        0.0157786_f32,
        -0.06221282_f32,
        -0.07301908_f32,
        -0.002688392_f32,
        -0.01559082_f32,
        -0.0417248_f32,
        0.02844391_f32,
        0.04183552_f32,
        -0.001469187_f32,
        0.05065957_f32,
        -0.006435158_f32,
        -0.03156701_f32,
        -0.03833957_f32,
        -0.02871311_f32,
        0.04148466_f32,
        0.002995606_f32,
        0.01613118_f32,
        0.06732765_f32,
        0.04938618_f32,
        0.05581886_f32,
        -0.01954652_f32,
        -0.005628012_f32,
        0.01619538_f32,
        -0.07513627_f32,
        -0.01343188_f32,
        0.01840665_f32,
        -0.008488718_f32,
        0.01129867_f32,
        0.01708702_f32,
        0.002774787_f32,
        -0.08587893_f32,
        -0.02644327_f32,
        0.01697917_f32,
        0.0539012_f32,
        -0.0419999_f32,
        0.04018815_f32,
        -0.02784589_f32,
        -0.0301916_f32,
        -0.01268079_f32,
        -0.08029023_f32,
        0.07221172_f32,
        -0.002345596_f32,
        0.03846241_f32,
        0.02357568_f32,
        -0.05354495_f32,
        -0.0314189_f32,
        -0.01863187_f32,
        0.005525303_f32,
        0.08312029_f32,
        -0.05893642_f32,
        0.01488875_f32,
        0.05034208_f32,
        -0.04552784_f32,
        -0.03934201_f32,
        -0.004898705_f32,
        -0.005334772_f32,
        -0.006965932_f32,
        0.005267634_f32,
        -0.02807925_f32,
        0.05273094_f32,
        -0.05419336_f32,
        0.02567079_f32,
        -0.008187627_f32,
        -0.0002183209_f32,
        -0.08071324_f32,
        -0.06437807_f32,
        -0.005438248_f32,
        -0.006374169_f32,
        -0.005831388_f32,
        0.0370548_f32,
        0.01883661_f32,
        0.03281118_f32,
        0.04116745_f32,
        -0.05312104_f32,
        0.01620268_f32,
        0.0418276_f32,
        -0.0114277_f32,
        -0.008354982_f32,
        -0.0534432_f32,
        -0.04842054_f32,
        -0.02539288_f32,
        -0.09614572_f32,
        0.02751439_f32,
        -0.001860047_f32,
        -0.03319291_f32,
        -0.02101517_f32,
        -0.002851211_f32,
        -0.03183515_f32,
        -0.01348068_f32,
        -0.003772831_f32,
        -0.001548059_f32,
        -0.07531881_f32,
        0.01666048_f32,
        -0.006787367_f32,
        0.03293696_f32,
        0.0743764_f32,
        0.009717998_f32,
        0.01902157_f32,
        -0.01709995_f32,
        0.01134409_f32,
        -0.0268736_f32,
        0.03519798_f32,
        -0.02172873_f32,
        -0.009943735_f32,
        0.01861613_f32,
        -0.01479012_f32,
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
