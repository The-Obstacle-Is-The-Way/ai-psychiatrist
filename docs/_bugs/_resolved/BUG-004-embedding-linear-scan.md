# BUG-004: O(N*M) Linear Scan in Embedding Retrieval

**Severity**: P3 (Performance)
**Status**: Resolved (Already Fixed)
**Resolved**: 2026-01-03
**Resolution**: Code already uses vectorized numpy at `embedding.py:223-247` with `matrix @ query_vec`
**Created**: 2026-01-03
**File**: `src/ai_psychiatrist/services/embedding.py`

## Description

The `EmbeddingService._compute_similarities` method performs a brute-force linear scan over all pre-computed reference embeddings to calculate cosine similarity.

```python
        for participant_id, chunks in all_refs.items():
            # ...
            for idx, (chunk_text, embedding) in enumerate(chunks):
                 # ...
                 raw_cos = float(cosine_similarity(query_array, ref_array)[0][0])
```

**Complexity**: $O(N \cdot M)$ where $N$ is the number of participants and $M$ is the average chunks per participant.
While acceptable for the current DAIC-WOZ dataset (100-200 participants), this will not scale to larger datasets (e.g., thousands of interviews).

## Impact

-   **Latency**: Retrieval time grows linearly with dataset size.
-   **CPU Usage**: High CPU consumption during the retrieval phase of "Few-Shot" assessment.

## Recommended Fix

Use a vector index or approximate nearest neighbor (ANN) search library for efficient retrieval (e.g., FAISS, ChromaDB, or simply `scikit-learn`'s `NearestNeighbors` with a ball tree if data fits in memory).

For a quick in-memory fix, pre-stack all embeddings into a single numpy matrix $X$ ($K \times D$) and compute similarities via matrix multiplication:
$$ \text{sims} = X \cdot q^T $$
This allows using BLAS optimizations rather than Python loops.

```
