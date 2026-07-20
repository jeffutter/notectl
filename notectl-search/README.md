# notectl-search

Semantic search over markdown notes using hybrid BM25 + dense vector retrieval.

## Features

- **Sparse retrieval**: BM25 scoring via lightweight in-memory indexer (no external dependencies)
- **Dense retrieval**: Qwen3-Embedding-0.6B via candle (default), with ONNX Runtime fallback for BGE/EmbeddingGemma models
- **Hybrid ranking**: Weighted Reciprocal Rank Fusion (RRF) merges sparse + dense results
- **Incremental indexing**: Only re-processes changed files on subsequent index builds
- **Matryoshka embeddings**: Supports truncating 1024-dim vectors to 512, 256, or 128 dimensions

## Cargo Features

| Feature | Description |
|---------|-------------|
| `qwen3` | Enable Qwen3 embedding models via candle backend (enabled by default) |
| `integration` | Run integration tests that require model download + inference |

## Smoke Test

```bash
# Build and index a vault
cargo run --bin notectl --features search -- index /path/to/vault

# Search the indexed vault
cargo run --bin notectl --features search -- search /path/to/vault "your query here"

# Verify JSON output structure
cargo run --bin notectl --features search -- search /path/to/vault "query" | jq '.results[0]'
```

## Running Tests

```bash
# Unit tests (no model required)
cargo test -p notectl-search

# Unit tests (all features)
cargo test -p notectl-search

# Doc-tests
cargo test -p notectl-search --doc

# Integration test (requires HF_TOKEN + network access)
HF_TOKEN=<token> cargo test -p notectl-search --features integration
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                    CLI / MCP                     │
├─────────────────────────────────────────────────┤
│               capability.rs                      │
│          (SearchCapability + operations)         │
├──────────┬───────────┬──────────┬───────────────┤
│  index   │  search   │  chunker │   storage     │
│  (build) │  (query)  │ (split)  │ (persist)     │
├──────────┼───────────┼──────────┴───────────────┤
│  bm25    │  sparse   │  fusion                  │
│ (scoring)│ (wrapper) │(cosine + RRF)            │
├──────────┴───────────┴──────────────────────────┤
│              embeddings/                        │
│  ┌────────────┬────────────┬─────────────────┐ │
│  │ candle     │  onnx      │    config       │ │
│  │ (qwen3*)   │ (bge, etc) │  (model_id)     │ │
│  └────────────┴────────────┴─────────────────┘ │
└─────────────────────────────────────────────────┘
```

## Index Format

Indexed data is stored in `<cache_dir>/notectl/search/` by default:

- `manifest.json` — build metadata (timestamp, config hash, dimension)
- `chunks.json` — extracted text chunks with source file and line info
- `vectors.bin` — binary-packed embedding vectors (only with `embeddings` feature)
