//! Dense embedding support via candle + Gemma-3 backbone.
//!
//! This module provides:
//! - Weight downloading via hf-hub (with offline caching)
//! - Gemma-3 model loading with mean pooling + Dense projection layers
//! - Batch embedding with query/document prefix injection
//!
//! Gated behind the `embeddings` cargo feature.

pub mod download;
pub mod embed;
pub mod model;

pub use download::{DownloadError, download_model};
pub use embed::{Embedder, EmbeddingConfig};
pub use model::ModelLoader;
