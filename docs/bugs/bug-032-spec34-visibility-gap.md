# BUG-032: Spec 34 Item Tag Filter Not Displayed in Run Configuration

**Status**: OPEN
**Severity**: Low (UX/Observability)
**Found**: 2025-12-29
**Spec**: 34 (Item-Tagged Reference Embeddings)

## Problem

When running `scripts/reproduce_results.py`, the `enable_item_tag_filter` setting is not displayed in the run configuration header. Users cannot tell whether Spec 34's item filtering is active without inspecting logs or code.

**Current output:**
```
============================================================
PAPER REPRODUCTION: Quantitative PHQ-8 Evaluation (Item-level MAE)
============================================================
  Ollama: http://127.0.0.1:11434
  Quantitative Model: gemma3:27b-it-qat
  Embedding Model: qwen3-embedding:8b
  Embeddings Artifact: data/embeddings/huggingface_qwen3_8b_paper_train.npz
  Data Directory: data
  Split: paper-test
============================================================
  Embedding Backend: huggingface
```

**Missing**: No indication of whether `EMBEDDING_ENABLE_ITEM_TAG_FILTER` is `true` or `false`.

## Root Cause

Spec 34 defined the config flag and its implementation but did not include a requirement to display the flag in the run configuration output. The verification section (lines 167-170) focused on functional testing, not observability.

## Impact

- Users may run ablations without realizing the filter is off (default: `false`)
- Difficult to verify correct configuration from output logs alone
- Reduces reproducibility confidence

## Fix

Add to `print_run_configuration()` in `scripts/reproduce_results.py`:

```python
print(f"  Item Tag Filter: {settings.embedding.enable_item_tag_filter}")
```

Also consider adding Spec 33 guardrail settings for completeness:
```python
print(f"  Min Reference Similarity: {settings.embedding.min_reference_similarity}")
```

## Verification

After fix, dry-run should show:
```
  Item Tag Filter: True
  Min Reference Similarity: 0.5
```

## Related

- Spec 34: `docs/archive/specs/34-item-tagged-reference-embeddings.md`
- Spec 33: `docs/archive/specs/33-retrieval-quality-guardrails.md`
