# Chunk-Level Scoring (Spec 35) — Schema, Workflow, and Gotchas

**Audience**: Researchers running few-shot experiments
**Last Updated**: 2026-01-02

Chunk-level scores are a **new artifact** used when `EMBEDDING_REFERENCE_SCORE_SOURCE=chunk`. This replaces “participant-level score attached to every chunk” with a per-chunk estimated label.

SSOT implementations:
- `scripts/score_reference_chunks.py`
- `src/ai_psychiatrist/services/chunk_scoring.py` (prompt template + prompt hash)
- `src/ai_psychiatrist/services/reference_store.py` (load/validate chunk scores)

---

## When You Need This

You need chunk scoring if you want **few-shot examples whose labels match the chunk content**.

Without chunk scores (`reference_score_source=participant`), retrieval can surface a chunk unrelated to the target symptom but still show it with the participant’s PHQ-8 item score. That is label noise by construction.

---

## Output Artifacts

For an embeddings artifact `{emb}.npz` (with `{emb}.json` sidecar), chunk scoring generates:

- `{emb}.chunk_scores.json`
- `{emb}.chunk_scores.meta.json`

Where `{emb}` is the resolved embeddings path without suffix (e.g., `data/embeddings/huggingface_qwen3_8b_paper_train_participant_only`).

---

## Schema (Exact)

### `{emb}.chunk_scores.json`

Top-level JSON object:
- keys: participant id strings (e.g., `"303"`)
- values: list of chunk score objects aligned with `{emb}.json`

Example:

```json
{
  "303": [
    {
      "PHQ8_NoInterest": null,
      "PHQ8_Depressed": 2,
      "PHQ8_Sleep": 3,
      "PHQ8_Tired": null,
      "PHQ8_Appetite": null,
      "PHQ8_Failure": null,
      "PHQ8_Concentrating": null,
      "PHQ8_Moving": null
    }
  ]
}
```

Constraints (enforced at load time):
- Participant IDs must match `{emb}.json`.
- Per-participant list length must equal the number of chunks for that participant in `{emb}.json`.
- Keys must be exactly the 8 `PHQ8_*` strings.
- Values must be `0..3` or `null`.

### `{emb}.chunk_scores.meta.json`

Example:

```json
{
  "scorer_model": "gemma3:27b-it-qat",
  "scorer_backend": "ollama",
  "temperature": 0.0,
  "prompt_hash": "...",
  "generated_at": "2025-12-31T00:00:00Z",
  "source_embeddings": "huggingface_qwen3_8b_paper_train_participant_only.npz",
  "total_chunks": 6837
}
```

The `prompt_hash` is a protocol lock:
- If the prompt template changes, chunk scores must be regenerated.
- Loading mismatched scores is blocked unless explicitly overridden (unsafe).

---

## How To Generate Chunk Scores

Run the scorer script:

```bash
uv run python scripts/score_reference_chunks.py \
  --embeddings-file huggingface_qwen3_8b_paper_train_participant_only \
  --scorer-backend ollama \
  --scorer-model gemma3:27b-it-qat
```

### Circularity Guard (`--allow-same-model`)

By default, the script blocks using the same model as the quantitative assessment model:
- if `--scorer-model == MODEL_QUANTITATIVE_MODEL` it exits `2`
- override with `--allow-same-model`

This is about **research defensibility** (correlated bias), not "state leakage". There is no training here; treat scorer choice as an ablation.

### Scorer Model Recommendations

| Priority | Scorer Choice | Notes |
|----------|---------------|-------|
| 1 (ideal) | MedGemma via HuggingFace | Medical tuning, most defensible (requires HF deps) |
| 2 (practical) | Different model family | e.g., `qwen2.5:7b-instruct-q4_K_M`, `llama3.1:8b-instruct-q4_K_M` |
| 3 (baseline) | Same model with `--allow-same-model` | Explicit opt-in, ablate against disjoint |

**MedGemma example** (if HuggingFace deps installed):
```bash
uv run python scripts/score_reference_chunks.py \
  --embeddings-file huggingface_qwen3_8b_paper_train_participant_only \
  --scorer-backend huggingface \
  --scorer-model medgemma:27b
```

**Disjoint model example** (practical default):
```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
uv run python scripts/score_reference_chunks.py \
  --embeddings-file huggingface_qwen3_8b_paper_train_participant_only \
  --scorer-backend ollama \
  --scorer-model qwen2.5:7b-instruct-q4_K_M
```

**Same model baseline** (for comparison):
```bash
uv run python scripts/score_reference_chunks.py \
  --embeddings-file huggingface_qwen3_8b_paper_train_participant_only \
  --scorer-backend ollama \
  --scorer-model gemma3:27b-it-qat \
  --allow-same-model
```

---

## How To Enable Chunk Scores at Runtime

Set:

```bash
EMBEDDING_REFERENCE_SCORE_SOURCE=chunk
```

If `{emb}.chunk_scores.json` or `{emb}.chunk_scores.meta.json` is missing, the system will fail fast.

Unsafe override (do not use for primary results):

```bash
EMBEDDING_ALLOW_CHUNK_SCORES_PROMPT_HASH_MISMATCH=true
```

---

## Important Implementation Detail (Format Interaction)

References whose `reference_score` is `null` are omitted from the few-shot prompt formatting. This can reduce the number of examples surfaced for some items.

See: `docs/embeddings/few-shot-prompt-format.md`.

---

## Failure Semantics (Be Honest)

`scripts/score_reference_chunks.py` currently does **not** fail-fast per chunk:
- if a chunk scoring call fails or returns invalid JSON, the script logs a warning and writes **all-null scores for that chunk**.

That means:
- A “successful” scoring run can still produce a low-quality artifact (many `null`s).
- Treat warnings as signals to rerun or adjust timeouts/model choice.

If you need strict fail-fast chunk scoring, add it as a dedicated spec (mirroring Spec 40’s design).

---

## Related Docs

- Embedding generation: `docs/embeddings/embedding-generation.md`
- Item tags: `docs/embeddings/item-tagging-setup.md`
- CRAG validation: `docs/embeddings/crag-validation.md`
