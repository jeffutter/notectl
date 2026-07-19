//! Harness to generate reference embeddings for integration tests.
//!
//! Produces REFERENCE_EMBEDDING and DOC_REFERENCE_EMBEDDING constants
//! from real EmbeddingGemma-300M inference output. Run with:
//!
//! ```bash
//! cargo run --features embeddings --example print_embedding
//! ```
//!
//! Requires HF_TOKEN with accepted license for google/embeddinggemma-300m.

use candle_core::{DType, Device, Tensor};
use notectl_search::embeddings::{
    download,
    model::{EmbeddingModelConfig, load_model, mean_pooling, normalize_embedding},
};
use tokenizers::Tokenizer;

fn get_embedding(
    input: &str,
    loaded: &mut notectl_search::embeddings::model::LoadedModel,
    tokenizer: &Tokenizer,
    embedding_config: &EmbeddingModelConfig,
) -> Vec<f32> {
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

    let input_ids = Tensor::new(padded.as_slice(), &loaded.device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let pad_tensor = Tensor::new(attention_mask.as_slice(), &loaded.device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    let hidden_states = loaded
        .model
        .forward(&input_ids, Some(&pad_tensor))
        .expect("Encoder forward failed");

    let pooling_mask = pad_tensor.clone();
    let pooled = mean_pooling(&hidden_states, &pooling_mask).expect("Mean pooling failed");
    let projected = loaded
        .projection_head
        .forward(&pooled)
        .expect("Projection failed");

    let embedding = projected.squeeze(0).unwrap();
    let raw: Vec<f32> = embedding.to_dtype(DType::F32).unwrap().to_vec1().unwrap();
    normalize_embedding(&raw, 768)
}

fn format_rust_array(values: &[f32], name: &str) -> String {
    let formatted: Vec<String> = values.iter().map(|v| format!("{:.10}_f32", v)).collect();
    // Group into lines of ~10 values for readability
    let mut lines = Vec::new();
    let chunk_size = 10;
    for chunk in formatted.chunks(chunk_size) {
        lines.push(format!("    {},", chunk.join(", ")));
    }
    format!("const {}: &[f32] = &[\n{}\n];", name, lines.join("\n"))
}

#[tokio::main]
async fn main() {
    let cache_dir = download::default_cache_dir();
    println!("Cache dir: {}", cache_dir.display());

    if !download::is_model_ready(&cache_dir) {
        println!("Model not found. Downloading...");
        match download::download_model(&cache_dir).await {
            Ok(_) => println!("Download complete."),
            Err(e) => {
                eprintln!("Download failed: {e}");
                std::process::exit(1);
            }
        }
    } else {
        println!("Model already downloaded.");
    }

    let device = Device::Cpu;
    let embedding_config = EmbeddingModelConfig {
        output_dim: 768,
        max_seq_len: 2048,
        dtype: DType::F32,
    };

    let mut loaded =
        load_model(&cache_dir, &device, &embedding_config).expect("Failed to load encoder model");

    let tokenizer_path = cache_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");

    let query_input = "task: search result | query: hello world";
    let doc_input = "title: My Note | text: hello world";

    println!("\nGenerating QUERY embedding...");
    let query_embedding = get_embedding(query_input, &mut loaded, &tokenizer, &embedding_config);
    println!("Query embedding dim: {}", query_embedding.len());
    println!("First 5 values: {:?}", &query_embedding[..5]);

    println!("\nGenerating DOC embedding...");
    let doc_embedding = get_embedding(doc_input, &mut loaded, &tokenizer, &embedding_config);
    println!("Doc embedding dim: {}", doc_embedding.len());
    println!("First 5 values: {:?}", &doc_embedding[..5]);

    // Output Rust source code ready to paste into model.rs
    println!("\n// === Paste into notectl-search/src/embeddings/model.rs ===\n");
    println!(
        "{}",
        format_rust_array(&query_embedding, "REFERENCE_EMBEDDING")
    );
    println!();
    println!(
        "/// Flip to `true` when REFERENCE_EMBEDDING is populated with real values.\n\
         const REFERENCE_EMBEDDING_POPULATED: bool = true;"
    );
    println!();
    println!(
        "{}",
        format_rust_array(&doc_embedding, "DOC_REFERENCE_EMBEDDING")
    );
    println!();
    println!(
        "/// Flip to `true` when DOC_REFERENCE_EMBEDDING is populated with real values.\n\
         const DOC_REFERENCE_EMBEDDING_POPULATED: bool = true;"
    );
}
