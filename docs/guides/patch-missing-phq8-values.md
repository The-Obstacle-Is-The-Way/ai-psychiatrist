# Patching Missing PHQ-8 Values (Deterministic, Not Imputation)

**Audience**: Researchers running reproduction splits
**Last Updated**: 2026-01-01

The AVEC2017-derived ground-truth CSVs occasionally contain a **missing PHQ-8 item cell**. This breaks paper-split evaluation because the reproduction runner requires complete per-item ground truth and will fail fast.

This repo includes a deterministic patch script:
- `scripts/patch_missing_phq8_values.py`

---

## Why This Is Valid

The dataset includes:
- `PHQ8_Score` (total score; authoritative)
- 8 item columns `PHQ8_*` (0–3 each)

For valid rows, the invariant must hold:

```text
PHQ8_Score == sum(PHQ8 item columns)
```

If exactly **one** item cell is missing and the total is present, the missing cell is uniquely determined:

```text
missing_item = PHQ8_Score - sum(known_items)
```

This is **not** statistical imputation. It is deterministic reconstruction of a single missing cell required for the invariant to hold.

---

## How To Patch

1) Preview what would change:

```bash
uv run python scripts/patch_missing_phq8_values.py --dry-run
```

2) Apply the patch:

```bash
uv run python scripts/patch_missing_phq8_values.py --apply
```

3) Regenerate paper splits (so paper CSVs reflect corrected values):

```bash
uv run python scripts/create_paper_split.py --verify
```

4) Re-run a quick validation:

```bash
uv run python scripts/reproduce_results.py --split paper --zero-shot-only --limit 3
```

---

## Failure Semantics (Expected)

If a ground-truth CSV has:
- more than one missing PHQ-8 item in a row, or
- an invariant violation (sum != total), or
- a reconstructed value outside `0..3`

the patch script will fail fast, because it cannot be corrected deterministically.
