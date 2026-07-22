//! One-off diagnostic: chunk a real file with the default config and report
//! the resulting chunk count / max token count / max byte size, to verify
//! the chunker's oversized-word fix actually bounds real pathological
//! content (minified JSON pastes, Excalidraw scene data) before re-enabling
//! previously-excluded files.

use notectl_search::chunker::{Chunker, ChunkerConfig};
use notectl_search::tokenize;
use std::path::Path;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: check_chunk_sizes <file>");
    let content = std::fs::read_to_string(&path).expect("failed to read file");

    let chunker = Chunker::new(ChunkerConfig::default());
    let chunks = chunker.chunk_file(Path::new(&path), &content);

    let mut max_tokens = 0;
    let mut max_bytes = 0;
    for c in &chunks {
        let tokens = tokenize::count_tokens(&c.text);
        max_tokens = max_tokens.max(tokens);
        max_bytes = max_bytes.max(c.text.len());
    }

    println!("file: {path}");
    println!("chunks: {}", chunks.len());
    println!(
        "max_tokens_per_chunk: {max_tokens} (budget: {})",
        ChunkerConfig::default().max_tokens
    );
    println!("max_bytes_per_chunk: {max_bytes}");
}
