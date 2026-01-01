# CRAG Reference Validation (Spec 36)

**Audience**: Researchers enabling “judge the retrieval” (CRAG)
**Last Updated**: 2026-01-01

CRAG-style validation adds a second LLM step after retrieval:
- retrieve candidate reference chunks
- validate each reference against the item + evidence (`accept` / `reject`)
- include only accepted references in the few-shot prompt

SSOT implementations:
- `src/ai_psychiatrist/services/reference_validation.py`
- `src/ai_psychiatrist/services/embedding.py` (wires validation into retrieval)

---

## Enable CRAG Validation

```bash
EMBEDDING_ENABLE_REFERENCE_VALIDATION=true
```

Optional overrides:

```bash
# If unset, runners fall back to MODEL_JUDGE_MODEL
EMBEDDING_VALIDATION_MODEL=gemma3:27b-it-qat

# Keep at most N accepted refs per item after validation
EMBEDDING_VALIDATION_MAX_REFS_PER_ITEM=2
```

---

## Fail-Fast Semantics (Spec 38)

If validation is enabled, it must work or crash:
- invalid JSON responses raise `LLMResponseParseError`
- network/backend failures propagate (preserve exception type)

There is no “return unsure and continue” fallback. Silent fallbacks corrupt research.

---

## What CRAG Can and Cannot Fix

CRAG validation is a **filter**, not a relabeler:
- It can reject irrelevant or contradictory references.
- It cannot correct a wrong `reference_score` label (that’s Spec 35’s job).

Recommended layering:
1. Spec 35 (chunk scores) for label correctness
2. Spec 34 (item tags) for candidate set precision
3. Spec 33 (threshold/budget) for quality guardrails
4. Spec 36 (CRAG) for semantic validation

---

## Debugging

Use retrieval audit logs (Spec 32) to see:
- which references survived CRAG filtering
- similarities and reference scores

See: `docs/guides/debugging-retrieval-quality.md`.
