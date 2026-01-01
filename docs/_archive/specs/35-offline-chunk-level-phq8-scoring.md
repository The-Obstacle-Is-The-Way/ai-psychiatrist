# Spec 35: Offline Chunk-Level PHQ-8 Scoring (New Method)

> **STATUS: ✅ IMPLEMENTED (2025-12-30)**
>
> This replaces participant-level score attachment with per-chunk score estimates.
>
> **Not paper-parity**: runs with `reference_score_source="chunk"` must be labeled as an experiment,
> not a paper reproduction baseline.

## Problem

The paper’s approach attaches participant-level PHQ-8 scores to arbitrary retrieved chunks. If a retrieved chunk does not reflect the severity implied by the participant’s overall score, few-shot examples become contradictory.

Chunk-level “ground truth” does not exist in the dataset, so any chunk-level score is necessarily an **estimate**.

## Goals (Acceptance Criteria)

1. Produce per-chunk per-item score estimates (`0..3` or `null`) for reference chunks.
2. Retrieval must attach **chunk-level score** for the retrieved chunk when enabled.
3. Must include explicit controls to avoid “label leakage” and circular evaluation.

## Non-goals

- This is not paper-parity (do not use for reproduction baseline).
- This does not change evidence extraction.

## Artifact Format (Backwards Compatible)

Add new sidecar:

- `{name}.chunk_scores.json`

Schema:

```json
{
  "303": [
    {
      "PHQ8_Sleep": 2,
      "PHQ8_Tired": null,
      "...": null
    },
    ...
  ]
}
```

Constraints:

- For each participant id, list length equals number of chunks in `{name}.json`.
- Keys must be the 8 `PHQ8_*` strings.
- Values must be `0..3` or `null`.

## Scoring Pipeline (Index-Time)

Add a new script:

- `scripts/score_reference_chunks.py`

Inputs:

- `--embeddings-file <name-or-path>` (same semantics as `EMBEDDING_EMBEDDINGS_FILE`)
- `--scorer-backend <ollama|huggingface|...>` (reuse existing LLM client infra)
- `--scorer-model <model>` (must be configurable)
- `--allow-same-model` (override; **unsafe** — allows scorer model to equal the assessment model)

Outputs:

- Writes `<embeddings_path>.chunk_scores.json` (where `<embeddings_path>` is the resolved `.npz` path)
- Writes `<embeddings_path>.chunk_scores.meta.json` with provenance:
  - scorer model
  - temperature
  - prompt hash
  - timestamp

### Scoring Prompt (Exact Requirements)

The scorer must:

- Use `temperature=0`.
- Only score what is supported **by the chunk text itself** (no extrapolation).
- Return strict JSON only (no prose).

Prompt template (single chunk → all 8 items at once):

**SSOT**: `src/ai_psychiatrist/services/chunk_scoring.py`
(`CHUNK_SCORING_PROMPT_TEMPLATE` + `chunk_scoring_prompt_hash()`).

```text
You are labeling a single transcript chunk for PHQ-8 item frequency evidence.

Task:
- For each PHQ-8 item key below, output an integer 0-3 if the chunk explicitly supports that frequency.
- If the chunk does not mention the symptom or frequency is unclear, output null.
- Do not guess or infer beyond the text.

Keys (must be present exactly):
PHQ8_NoInterest, PHQ8_Depressed, PHQ8_Sleep, PHQ8_Tired,
PHQ8_Appetite, PHQ8_Failure, PHQ8_Concentrating, PHQ8_Moving

Chunk:
<<<CHUNK_TEXT>>>

Return JSON only in this exact shape:
{
  "PHQ8_NoInterest": 0|1|2|3|null,
  "PHQ8_Depressed": 0|1|2|3|null,
  ...
}
```

Parsing:

- Reject any output that is not valid JSON.
- Reject any integer not in `0..3`.
- Treat missing keys as schema error.

## Circularity Controls (Required)

1. **Disjoint model**: scorer model must not equal the assessment model (enforced by default;
   override requires `--allow-same-model`).
2. **Temperature=0** (determinism).
3. **Protocol lock**: store prompt hash; refuse to load scores if prompt hash differs (unless override flag set).
4. **Reporting**: outputs must clearly label runs as “chunk-score method”, not reproduction.

### Scorer Model Selection (Practical Guidance)

This “disjoint model” rule is about **defensibility**, not “state leakage” (there is no training here).
The actual risk is correlated bias: the same model may find its own labels more “natural” to use as few-shot examples.

For research clarity, treat the scorer choice as an **ablation**:

1. **Same-model baseline** (explicit opt-in):
   - `--scorer-model gemma3:27b-it-qat --allow-same-model`
2. **Disjoint scorer** (recommended sensitivity check):
   - `--scorer-model qwen2.5:7b-instruct-q4_K_M` (or `llama3.1:8b-instruct-q4_K_M`)
3. **MedGemma scorer** (optional, if feasible):
   - `--scorer-backend huggingface --scorer-model medgemma:27b`
   - Uses canonical `medgemma:27b` → resolves to `google/medgemma-27b-text-it`
   - **Do not** use `google/medgemma-27b-it` / `google/medgemma-4b-it` directly here; those are multimodal
     (`Gemma3ForConditionalGeneration`) and are not supported by our current HuggingFace chat client.

Operational note: `scripts/score_reference_chunks.py` writes to a fixed filename
(`<embeddings>.chunk_scores.json`). If you run multiple scorers, copy/rename the outputs (including
`.chunk_scores.meta.json`) and swap them back in before each evaluation run.

## Runtime Integration

Add setting:

- `EmbeddingSettings.reference_score_source: Literal[\"participant\", \"chunk\"] = \"participant\"` (env: `EMBEDDING_REFERENCE_SCORE_SOURCE`)
- `EmbeddingSettings.allow_chunk_scores_prompt_hash_mismatch: bool = False`
  - Env: `EMBEDDING_ALLOW_CHUNK_SCORES_PROMPT_HASH_MISMATCH`
  - **Unsafe**: allows loading chunk scores when `.chunk_scores.meta.json` is missing or the prompt hash mismatches.

### Files to Change

- `src/ai_psychiatrist/config.py` (`EmbeddingSettings`)
- `src/ai_psychiatrist/services/chunk_scoring.py` (prompt SSOT + prompt hash)
- `src/ai_psychiatrist/services/reference_store.py` (load/validate `.chunk_scores.json` + protocol lock via `.chunk_scores.meta.json`)
- `src/ai_psychiatrist/services/embedding.py` (use chunk score when enabled; requires chunk index)
- `scripts/score_reference_chunks.py` (index-time scorer; protocol lock metadata)
- `scripts/reproduce_results.py` (prints `reference_score_source` and labels non-paper runs as experiments)
- `tests/unit/services/test_embedding.py` + `tests/unit/services/test_reference_store.py`

### ReferenceStore API (Exact)

Add to `src/ai_psychiatrist/services/reference_store.py`:

- Sidecar path:

```python
def _get_chunk_scores_path(self) -> Path:
    return self._embeddings_path.with_suffix(".chunk_scores.json")
```

- Loader + validator that enforces:
  - same participant IDs as texts sidecar
  - per-participant list length equals chunk count
  - keys are exactly the 8 `PHQ8_*` strings, values are `0..3` or `None`
  - `.chunk_scores.meta.json` exists and `prompt_hash` matches `chunk_scoring_prompt_hash()` unless override enabled

- Public method:

```python
def has_chunk_scores(self) -> bool: ...

def get_chunk_score(self, participant_id: int, chunk_index: int, item: PHQ8Item) -> int | None: ...
```

### EmbeddingService Integration (Exact)

In `src/ai_psychiatrist/services/embedding.py`:

1. In `EmbeddingService.__init__` store:

```python
self._reference_score_source = settings.reference_score_source
if self._reference_score_source == "chunk" and not self._reference_store.has_chunk_scores():
    raise ValueError("reference_score_source='chunk' requires <embeddings>.chunk_scores.json")
```

2. In `_compute_similarities(...)`, enumerate chunks and pick score source:

```python
for chunk_index, (chunk_text, embedding) in enumerate(chunks):
    ...
    if self._reference_score_source == "chunk":
        score = self._reference_store.get_chunk_score(participant_id, chunk_index, lookup_item)
    else:
        score = self._reference_store.get_score(participant_id, lookup_item)
```

Store `reference_score=score` in the `SimilarityMatch` as today.

### Design Choice (Intentional)

- If a chunk score is `None`, `ReferenceBundle.format_for_prompt()` (Spec 31) will omit that reference.
- This keeps the “examples” pool semantically cleaner but can reduce coverage; measure the tradeoff.

## TDD: Tests (Must Exist)

1. `test_get_chunk_score_returns_expected_score`
   - Given a synthetic `{name}.chunk_scores.json`, verify lookup by `(pid, chunk_index, item)` returns the correct int or None.
2. `test_reference_score_source_participant_is_default`
   - With default settings, `_compute_similarities` must use CSV score lookup (`get_score`).
3. `test_reference_score_source_chunk_requires_sidecar`
   - With `reference_score_source="chunk"` and no sidecar, EmbeddingService init must raise.

Additional safety tests (recommended):

- Prompt hash mismatch must fail closed (unless override enabled).
- Invalid key sets and length mismatches must raise `EmbeddingArtifactMismatchError`.

## Verification

- Generate chunk scores on paper-train references.
- Run as a “new method” experiment (NOT paper reproduction); compare paired AURC deltas.
- Audit whether chunk-score examples look semantically aligned.

## Risks

- High compute cost (LLM per chunk).
- Still potentially circular (LLM-generated labels steering an LLM).
- May improve metrics without improving truth alignment.
