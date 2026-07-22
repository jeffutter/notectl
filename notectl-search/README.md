# notectl-search

Semantic search over markdown notes using hybrid BM25 + dense vector retrieval.

## Features

- **Sparse retrieval**: BM25 scoring via lightweight in-memory indexer (no external dependencies)
- **Dense retrieval**: HTTP client to any OpenAI-compatible `/v1/embeddings` endpoint (llama.cpp/llama-swap, vLLM, Ollama, OpenAI, etc.) — no model runs in-process
- **Hybrid ranking**: Weighted Reciprocal Rank Fusion (RRF) merges sparse + dense results
- **Incremental indexing**: Only re-processes changed files on subsequent index builds
- **Matryoshka embeddings**: Optionally truncate a model's native embedding dimension for storage savings

## Smoke Test

```bash
# Build and index a vault
cargo run --bin notectl -- index /path/to/vault

# Search the indexed vault
cargo run --bin notectl -- search /path/to/vault "your query here"

# Verify JSON output structure
cargo run --bin notectl -- search /path/to/vault "query" | jq '.results[0]'
```

## Running Tests

```bash
# Unit tests — fully offline, nothing touches the network
cargo test -p notectl-search

# Doc-tests
cargo test -p notectl-search --doc

# Sanity-check connectivity to the configured embedding endpoint
cargo run --example print_embedding -p notectl-search
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
│  reqwest client → POST {api_base}/embeddings     │
└─────────────────────────────────────────────────┘
```

## Index Format

Indexed data is stored in `<cache_dir>/notectl/search/` by default:

- `manifest.json` — build metadata (timestamp, config hash, dimension)
- `chunks/` — extracted text chunks, one file per chunk
- `vectors.bin` — binary-packed embedding vectors (only when an embedding endpoint is configured)
