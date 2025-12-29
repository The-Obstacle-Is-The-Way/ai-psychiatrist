# Spec 32: Few-Shot Retrieval Diagnostics (Audit Logs)

> **STATUS: ✅ IMPLEMENTED (2025-12-28) → Archive**
>
> **Scope**: Add an auditable, structured log trail for what retrieval returned.

## Problem

The “semantic mismatch” hypothesis is currently largely untestable because we do not persist or log which chunks were retrieved (participant id, similarity, score, etc.). Without auditability:

- We can’t verify whether few-shot examples are contradictory or low-similarity.
- We can’t compute distributions (similarity vs error, by item) without re-running with ad hoc prints.

## Goals (Acceptance Criteria)

1. For every PHQ-8 item with evidence, emit an INFO log per retrieved match (`rank <= top_k`).
2. Logs must include at minimum: `item`, `evidence_key`, `rank`, `similarity`, `participant_id`, `reference_score`, `chunk_preview`, `chunk_chars`.
3. Logging must be **opt-in** and default OFF to avoid noisy reproduction artifacts.
4. Must be safe by default: log only a preview, not the full chunk.

## Non-goals

- No filtering, reranking, or quality judgments (Spec 33+).
- No changes to reference formatting (Spec 31).
- No changes to output JSON schema (future enhancement; not required here).

## Implementation

### Files to Change

- `src/ai_psychiatrist/config.py` (`EmbeddingSettings`)
- `src/ai_psychiatrist/services/embedding.py` (`EmbeddingService.build_reference_bundle`)
- `tests/unit/services/test_embedding.py` (new unit test)

### Configuration (Opt-in Toggle)

Add a new setting:

- `EmbeddingSettings.enable_retrieval_audit: bool = False`
- Env var: `EMBEDDING_ENABLE_RETRIEVAL_AUDIT=true`

### Logging Event (Exact Fields)

In `src/ai_psychiatrist/services/embedding.py`, inside `EmbeddingService.build_reference_bundle`, after:

```python
top_matches = matches[: self._top_k]
```

Add:

```python
if self._enable_retrieval_audit:
    evidence_key = f"PHQ8_{item.value}"
    for rank, match in enumerate(top_matches, start=1):
        logger.info(
            "retrieved_reference",
            item=item.value,
            evidence_key=evidence_key,
            rank=rank,
            similarity=match.similarity,
            participant_id=match.chunk.participant_id,
            reference_score=match.reference_score,
            chunk_preview=match.chunk.text[:160],
            chunk_chars=len(match.chunk.text),
        )
```

### Required Wiring

In `EmbeddingService.__init__`, store the flag on the instance:

```python
self._enable_retrieval_audit = settings.enable_retrieval_audit
```

## TDD: Unit Test (Copy/Paste)

✅ Copy/paste test code (self-safe): add this **method** inside the existing
`class TestEmbeddingService` in `tests/unit/services/test_embedding.py`.

- This is **not** a module-level function (it uses `self`).
- This uses `monkeypatch.setattr(...)` to avoid mypy `method-assign` ignores.
- No new imports are required beyond what `tests/unit/services/test_embedding.py` already has.

```python
    @pytest.mark.asyncio
    async def test_build_reference_bundle_logs_audit_when_enabled(
        self,
        mock_llm_client: MagicMock,
        mock_reference_store: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = EmbeddingSettings(
            dimension=256,
            top_k_references=2,
            min_evidence_chars=1,
            enable_retrieval_audit=True,
        )

        service = EmbeddingService(mock_llm_client, mock_reference_store, settings)

        # Force deterministic matches without depending on sklearn cosine math.
        sleep_item = PHQ8Item.SLEEP
        matches = [
            SimilarityMatch(
                chunk=TranscriptChunk(text="aaa " * 100, participant_id=111),
                similarity=0.9,
                reference_score=3,
            ),
            SimilarityMatch(
                chunk=TranscriptChunk(text="bbb " * 100, participant_id=222),
                similarity=0.8,
                reference_score=1,
            ),
        ]
        monkeypatch.setattr(service, "_compute_similarities", MagicMock(return_value=matches))

        # Patch module logger
        from ai_psychiatrist.services import embedding as embedding_module

        logger_mock = MagicMock()
        monkeypatch.setattr(embedding_module, "logger", logger_mock)

        bundle = await service.build_reference_bundle({sleep_item: ["evidence"]})

        assert sleep_item in bundle.item_references

        # Two audit log calls for two matches
        retrieved = [
            call
            for call in logger_mock.info.call_args_list
            if call.args and call.args[0] == "retrieved_reference"
        ]
        assert len(retrieved) == 2
        logger_mock.info.assert_any_call(
            "retrieved_reference",
            item="Sleep",
            evidence_key="PHQ8_Sleep",
            rank=1,
            similarity=0.9,
            participant_id=111,
            reference_score=3,
            chunk_preview=("aaa " * 100)[:160],
            chunk_chars=len("aaa " * 100),
        )
```

## Verification

- Fast unit check (skip global coverage gate): `uv run pytest tests/unit/services/test_embedding.py -q --no-cov`
- Full suite (enforces coverage): `make test`
- Run a reproduction with audit enabled:

```bash
EMBEDDING_ENABLE_RETRIEVAL_AUDIT=true uv run python scripts/reproduce_results.py --split paper-test
```

## Design Notes (SOLID / DRY / GoF)

- Opt-in logging keeps production runs clean (Single Responsibility + Open/Closed).
- If audit needs to be persisted (not just logged), add an Observer/Publisher abstraction later (out of scope).
