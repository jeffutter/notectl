//! Harness to generate reference embeddings for integration tests.
//!
//! Produces embeddings from the configured model via fastembed. Run with:
//!
//! ```bash
//! cargo run --example print_embedding
//! ```
//!
//! The model downloads automatically on first run.

use notectl_search::embeddings::{Embedder, EmbeddingConfig, TaskType};

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
    let config = EmbeddingConfig::default();
    println!("Model: {}", config.model_id);
    println!("Dimension: {}", config.embedding_dim);

    let mut embedder = Embedder::new(config);

    let query_input = "search result";
    let doc_input = "hello world";

    println!("\nGenerating QUERY embedding...");
    match embedder
        .embed_single(query_input, None, TaskType::RetrievalQuery)
        .await
    {
        Ok(query_embedding) => {
            println!("Query embedding dim: {}", query_embedding.len());
            println!(
                "First 5 values: {:?}",
                &query_embedding[..5.min(query_embedding.len())]
            );

            // Output Rust source code ready to paste
            println!("\n// === Generated embedding reference ===\n");
            println!(
                "{}",
                format_rust_array(&query_embedding, "REFERENCE_EMBEDDING")
            );
        }
        Err(e) => {
            eprintln!("Query embedding failed: {e}");
            std::process::exit(1);
        }
    }

    println!("\nGenerating DOC embedding...");
    match embedder
        .embed_single(doc_input, Some("My Note"), TaskType::RetrievalDocument)
        .await
    {
        Ok(doc_embedding) => {
            println!("Doc embedding dim: {}", doc_embedding.len());
            println!(
                "First 5 values: {:?}",
                &doc_embedding[..5.min(doc_embedding.len())]
            );

            println!();
            println!(
                "{}",
                format_rust_array(&doc_embedding, "DOC_REFERENCE_EMBEDDING")
            );
        }
        Err(e) => {
            eprintln!("Doc embedding failed: {e}");
            std::process::exit(1);
        }
    }
}
