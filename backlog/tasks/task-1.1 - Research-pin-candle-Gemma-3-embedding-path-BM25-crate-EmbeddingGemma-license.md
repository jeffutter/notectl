---
id: TASK-1.1
title: >-
  Research: pin candle Gemma-3 embedding path, BM25 crate, EmbeddingGemma
  license
status: Done
assignee:
  - '@ralph'
created_date: '2026-07-14 02:21'
updated_date: '2026-07-14 04:04'
labels: []
dependencies: []
parent_task_id: TASK-1
priority: high
type: spike
ordinal: 2000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Confirm candle-transformers exposes a Gemma-3 model usable for EmbeddingGemma (sliding+full attention). Pick a BM25 crate (or evaluate tantivy's built-in scoring as an alternative). Verify EmbeddingGemma's HF gating/license terms and what the first-run auth flow needs to look like. Study the text-embeddings-inference Rust reference implementation for the exact pooling + 2_Dense projection head and the query/document prompt prefixes EmbeddingGemma expects.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
## Research Findings

### 1. Candle Transformers Gemma-3 Support ✓ CONFIRMED
- **Location**:  (main branch)
- **Attention Pattern**: Supports both sliding window and full attention via  config
  - Layers alternate:  → SlidingAttention, else FullAttention
  - Default pattern: 1 full attention layer followed by N sliding attention layers
- **Architecture**: GQA (Grouped Query Attention), RMSNorm, RoPE with local frequencies for sliding window
- **KV Cache**: Normal cache for full attention, RotatingKvCache for sliding window
- **Two Model Variants**:
  -  - For text generation (inference)
  -  - For embeddings (in TEI)

### 2. BM25 Crate Recommendation: bm25x v0.3.1 ✓
- **License**: Apache-2.0 (permissive)
- **Key Features**:
  - Lazy scoring (no rebuild for add/delete/update operations)
  - Persistent indices with mmap support (low RAM usage)
  - Pre-filtered search: O(|subset| × |query_terms| × log n) vs full scan
  - Batch search with rayon parallelism (2.6x faster on CPU)
  - Optional GPU acceleration via CUDA (up to 815x faster on large datasets)
  - 5 BM25 variants: lucene, robertson, atire, bm25l, bm25+
  - Built-in English stopwords removal
  - Streaming mutations (add, delete, update) with auto-persistence
- **Performance**: 3.5-6x faster than bm25s on CPU indexing/search
- **API Design**: Clean, well-documented, Python bindings available
- **Recommended for notectl**: Use  with default k1=1.5, b=0.75

### 3. EmbeddingGemma License & Auth Flow ✓
- **License Type**: Gemma Terms of Use (gated model)
- **Access Requirements**:
  1. User must be logged into Hugging Face account
  2. Must acknowledge Google's usage license via gated portal ( + )
  3. Requires valid HF access token with read permissions
  4. Subject to Gemma Prohibited Use Policy
- **First-Run Auth Flow**:
  
- **Implication for notectl**: Need HF token configuration, license acknowledgment UI/CLI flag

### 4. Text-Embeddings-Inference Implementation Details ✓

**Pooling Strategy**:
- **Type**: Mean Pooling (NOT last-token or CLS)
- **Config**:  with 
- **TEI Support**: ,  (CLS/SPLADE not supported for Gemma3)
- **Implementation**: Average hidden states across sequence dimension, excluding padding

**Dense Projection Layers**:
- **Architecture**: Two sequential Dense layers
  - : 768 → 3072 dimensions (linear + activation)
  - : 3072 → 768 dimensions (final output)
- **Definition**: In  as  modules
- **File Structure**: 
  
- **TEI Parsing**: Reads , applies Dense modules sequentially after pooling
- **CLI Override**:  arg (only for single Dense, not multiple sequential)

**Prompt Prefixes** (from model card):
- **Query (Retrieval)**: 
- **Document (Retrieval)**: 
- **Question Answering**: 
- **Fact Verification**: 
- **Classification**: 
- **Clustering**: 
- **Semantic Similarity**: 
- **Code Retrieval**:  (query) +  (doc)

**Model Specs**:
- Parameters: 300M
- Backbone: Gemma 3 with T5Gemma initialization
- Max Context: 2048 tokens
- Output Dimension: 768 (with MRL support for 512, 256, 128)
- Supported dtypes: float32, bfloat16 (NOT float16)
- Training: ~320B tokens, 100+ languages

## Implementation Recommendations

1. **Use ** for Gemma3 backbone (mature, well-tested)
2. **Implement custom embedding wrapper** that:
   - Applies mean pooling to hidden states
   - Loads and applies Dense projection layers from 
   - Handles prompt prefix injection based on task type
3. **Use  crate** for BM25 scoring with lazy indexing
4. **Hybrid Search**: Combine BM25 + dense embeddings with RRF (Reciprocal Rank Fusion) or weighted sum
5. **HF Token Management**: Add  CLI arg and  file support
6. **License Acknowledgment**: Add config flag or interactive prompt for first-time users

## Research Findings

### 1. Candle Transformers Gemma-3 Support ✓ CONFIRMED
- **Location**: `candle-transformers/src/models/gemma3.rs` (main branch)
- **Attention Pattern**: Supports both sliding window and full attention via `sliding_window_pattern` config
  - Layers alternate: `(layer_idx + 1) % sliding_window_pattern > 0` → SlidingAttention, else FullAttention
  - Default pattern: 1 full attention layer followed by N sliding attention layers
- **Architecture**: GQA (Grouped Query Attention), RMSNorm, RoPE with local frequencies for sliding window
- **KV Cache**: Normal cache for full attention, RotatingKvCache for sliding window
- **Two Model Variants**:
  - `candle_transformers::models::gemma3::Model` - For text generation (inference)
  - `text_embeddings_backend_candle::models::gemma3::Gemma3Model` - For embeddings (in TEI)

### 2. BM25 Crate Recommendation: bm25x v0.3.1 ✓
- **License**: Apache-2.0 (permissive)
- **Key Features**:
  - Lazy scoring (no rebuild for add/delete/update operations)
  - Persistent indices with mmap support (low RAM usage)
  - Pre-filtered search: O(|subset| × |query_terms| × log n) vs full scan
  - Batch search with rayon parallelism (2.6x faster on CPU)
  - Optional GPU acceleration via CUDA (up to 815x faster on large datasets)
  - 5 BM25 variants: lucene, robertson, atire, bm25l, bm25+
  - Built-in English stopwords removal
  - Streaming mutations (add, delete, update) with auto-persistence
- **Performance**: 3.5-6x faster than bm25s on CPU indexing/search
- **API Design**: Clean, well-documented, Python bindings available
- **Recommended for notectl**: Use `Method::Lucene` with default k1=1.5, b=0.75

### 3. EmbeddingGemma License & Auth Flow ✓
- **License Type**: Gemma Terms of Use (gated model)
- **Access Requirements**:
  1. User must be logged into Hugging Face account
  2. Must acknowledge Google's usage license via gated portal
  3. Requires valid HF access token with read permissions
  4. Subject to Gemma Prohibited Use Policy
- **First-Run Auth Flow**:
  1. User clicks model page on HuggingFace
  2. Sees "Acknowledge license" button
  3. Logs in (if not already)
  4. Clicks button → request processed immediately
  5. Gets access to download weights
  6. Token stored in $HF_HOME/token for future use
- **Implication for notectl**: Need HF token configuration, license acknowledgment UI/CLI flag

### 4. Text-Embeddings-Inference Implementation Details ✓

**Pooling Strategy**:
- **Type**: Mean Pooling (NOT last-token or CLS)
- **Config**: `1_Pooling/config.json` with `"pooling_mode": "mean"`
- **TEI Support**: Pool::Mean, Pool::LastToken (CLS/SPLADE not supported for Gemma3)
- **Implementation**: Average hidden states across sequence dimension, excluding padding

**Dense Projection Layers**:
- **Architecture**: Two sequential Dense layers
  - `2_Dense`: 768 → 3072 dimensions (linear + activation)
  - `3_Dense`: 3072 → 768 dimensions (final output)
- **Definition**: In `modules.json` as `sentence_transformers.models.Dense` modules
- **File Structure**: 
  - `2_Dense/config.json`      # {dimension: 3072, name: "Dense", activation: "tanh"}
  - `2_Dense/model.safetensors`
  - `3_Dense/config.json`      # {dimension: 768, name: "Dense", activation: "linear"}
  - `3_Dense/model.safetensors`
- **TEI Parsing**: Reads `modules.json`, applies Dense modules sequentially after pooling
- **CLI Override**: `--dense-path` arg (only for single Dense, not multiple sequential)

**Prompt Prefixes** (from model card):
- **Query (Retrieval)**: `task: search result | query: {content}`
- **Document (Retrieval)**: `title: {title | "none"} | text: {content}`
- **Question Answering**: `task: question answering | query: {content}`
- **Fact Verification**: `task: fact checking | query: {content}`
- **Classification**: `task: classification | query: {content}`
- **Clustering**: `task: clustering | query: {content}`
- **Semantic Similarity**: `task: sentence similarity | query: {content}`
- **Code Retrieval**: `task: code retrieval | query: {content}` (query) + `retrieval_document` (doc)

**Model Specs**:
- Parameters: 300M
- Backbone: Gemma 3 with T5Gemma initialization
- Max Context: 2048 tokens
- Output Dimension: 768 (with MRL support for 512, 256, 128)
- Supported dtypes: float32, bfloat16 (NOT float16)
- Training: ~320B tokens, 100+ languages

## Implementation Recommendations

1. **Use `candle_transformers::models::gemma3`** for Gemma3 backbone (mature, well-tested)
2. **Implement custom embedding wrapper** that:
   - Applies mean pooling to hidden states
   - Loads and applies Dense projection layers from `modules.json`
   - Handles prompt prefix injection based on task type
3. **Use `bm25x` crate** for BM25 scoring with lazy indexing
4. **Hybrid Search**: Combine BM25 + dense embeddings with RRF (Reciprocal Rank Fusion) or weighted sum
5. **HF Token Management**: Add `--hf-token` CLI arg and `$HF_HOME/token` file support
6. **License Acknowledgment**: Add config flag or interactive prompt for first-time users
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Research spike completed. All four objectives confirmed:

1. **Candle Gemma-3 Support**: Confirmed in candle-transformers/src/models/gemma3.rs with full sliding+full attention support via sliding_window_pattern config. Two model variants available (inference and embedding).

2. **BM25 Crate**: Recommended bm25x v0.3.1 (Apache-2.0 license). Features lazy scoring, persistent mmap indices, pre-filtered search, batch parallelism, optional GPU acceleration. 3.5-6x faster than alternatives. Use Method::Lucene with default params.

3. **EmbeddingGemma License**: Gemma Terms of Use (gated model). Requires HF login + license acknowledgment via gated portal + valid HF token. First-run flow: user acknowledges on HF website, token stored in /token.

4. **TEI Implementation Details**: 
   - Pooling: Mean pooling (config: 1_Pooling/config.json)
   - Dense Layers: Two sequential projections (2_Dense: 768→3072, 3_Dense: 3072→768) defined in modules.json
   - Prompts: Query='task: search result | query:', Document='title: {title} | text:'
   - Model: 300M params, Gemma 3 backbone, 2048 context, 768 output dims (MRL supported)

Implementation path clear: use candle_transformers for backbone, implement custom embedding wrapper with mean pooling + Dense layers, integrate bm25x for hybrid search.
<!-- SECTION:FINAL_SUMMARY:END -->
