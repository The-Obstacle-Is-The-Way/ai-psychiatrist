# Spec 36: CRAG-Style Runtime Reference Validation (New Method)

> **STATUS: PLANNED (High runtime cost; variance controls required)**
>
> This adds a post-retrieval evaluator to reject irrelevant/contradictory references.

## Problem

Even with item tags (Spec 34) and/or similarity thresholds (Spec 33), retrieved chunks can still be irrelevant or misleading. A CRAG-style evaluator can filter or flag bad references at runtime.

## Goals (Acceptance Criteria)

1. Add an optional runtime evaluator that labels each retrieved reference as:
   - `accept` / `reject` / `unsure`
2. Only `accept` references are included in the final `ReferenceBundle` when enabled.
3. Must be configurable and default OFF.

## Non-goals

- Not paper-parity.
- Not a replacement for fixing formatting (Spec 31).

## Design (Strategy)

Introduce a `ReferenceValidator` Strategy with implementations:

- `NoOpValidator` (default)
- `LLMValidator` (CRAG-style)

Validator input:

- `item` (PHQ8Item)
- `evidence_text` (combined evidence quotes)
- `reference_chunk_text`
- `reference_score` (participant-level or chunk-level depending on score source)

Validator output:

- `Literal[\"accept\", \"reject\", \"unsure\"]` + optional rationale (not logged by default)

## Implementation

### Files to Change

- `src/ai_psychiatrist/config.py` (`EmbeddingSettings`)
- `src/ai_psychiatrist/services/embedding.py` (wire validator into `build_reference_bundle`)
- `src/ai_psychiatrist/services/reference_validation.py` (new; validator Strategy)
- `server.py` (construct and inject validator when enabled)
- `scripts/reproduce_results.py` (construct and inject validator when enabled)
- `tests/unit/services/test_embedding.py` (mock validator; no real LLM calls)

### Settings

Add to config:

- `EmbeddingSettings.enable_reference_validation: bool = False`
- `EmbeddingSettings.validation_model: str`
- `EmbeddingSettings.validation_max_refs_per_item: int = 2` (keep bounded)

Expected env vars:

- `EMBEDDING_ENABLE_REFERENCE_VALIDATION`
- `EMBEDDING_VALIDATION_MODEL`
- `EMBEDDING_VALIDATION_MAX_REFS_PER_ITEM`

### Prompt (Exact Requirement)

The validator prompt must be:

- Deterministic: `temperature=0`
- Strict JSON output: `{\"decision\": \"accept|reject|unsure\"}`

Treat `unsure` as `reject` unless a future setting explicitly changes this.

### Validator Strategy (Exact Interfaces)

Create `src/ai_psychiatrist/services/reference_validation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from ai_psychiatrist.domain.enums import PHQ8Item

Decision = Literal["accept", "reject", "unsure"]

@dataclass(frozen=True, slots=True)
class ReferenceValidationRequest:
    item: PHQ8Item
    evidence_text: str
    reference_text: str
    reference_score: int | None

class ReferenceValidator(Protocol):
    async def validate(self, request: ReferenceValidationRequest) -> Decision: ...
```

Implementations:

- `NoOpReferenceValidator`: always returns `"accept"`.
- `LLMReferenceValidator`: uses a `ChatClient` to produce the strict JSON decision.

### EmbeddingService Wiring (Dependency Injection)

Update `EmbeddingService.__init__` to accept an optional validator:

```python
def __init__(..., reference_validator: ReferenceValidator | None = None, ...) -> None:
    ...
    self._reference_validator = reference_validator or NoOpReferenceValidator()
    self._enable_reference_validation = settings.enable_reference_validation
    self._validation_max_refs_per_item = settings.validation_max_refs_per_item
```

In `build_reference_bundle`, after `top_matches` computed:

1. If validation disabled: keep `top_matches` unchanged.
2. If enabled:
   - Build `ReferenceValidationRequest` for each match
   - Drop matches where decision != `"accept"`
   - If decision is `"unsure"`, treat as reject
   - Keep at most `validation_max_refs_per_item` accepted matches

### Construction (server + scripts)

Because `LLMReferenceValidator` needs a `ChatClient` (not the embedding client), construct it in the orchestrators that already have a chat client:

- `server.py`: use `chat_client` to build `LLMReferenceValidator(...)` when `settings.embedding.enable_reference_validation` is true; pass into `EmbeddingService(...)`.
- `scripts/reproduce_results.py`: use the chat client used for quantitative scoring to build and inject the validator when enabled.

### Integration Point

In `EmbeddingService.build_reference_bundle`, after `top_matches` computed:

1. For each match, call validator.
2. Keep only `accept`.
3. If zero accepted matches, treat item as having no matches.

## TDD: Tests (Must Exist)

1. `test_validation_disabled_keeps_all_top_matches`
2. `test_validation_rejects_drops_reference`
3. `test_validation_all_rejected_results_in_empty_item`

All tests must mock the validator (no real LLM calls).

Copy/paste test guidance:

- Add a fake validator that returns `reject` for a specific `participant_id` or chunk text.
- Patch `_compute_similarities` to return deterministic matches.
- Assert that `bundle.item_references[item]` contains only accepted matches.

## Verification

- Run small manual audit on 5 participants with validation enabled; confirm rejected references are truly irrelevant.
- Compare paired AURC deltas vs baseline.

## Risks

- Runtime cost and latency.
- Additional LLM variance (must control with temperature=0 and repeated runs).
- Potential bias if validator is too strict (coverage drops).
