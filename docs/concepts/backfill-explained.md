# Keyword Backfill: The Safety Net for Evidence Extraction

**Audience**: Researchers and developers wanting to understand the coverage-accuracy tradeoff
**Related**: [SPEC-003](../specs/SPEC-003-backfill-toggle.md) | [Coverage Investigation](../bugs/coverage-investigation.md) | [Extraction Mechanism](./extraction-mechanism.md)
**Last Updated**: 2025-12-23

---

## What Is Keyword Backfill?

Keyword backfill is a **rule-based safety net** that runs after LLM evidence extraction. When the LLM fails to find evidence for a PHQ-8 symptom, backfill scans the transcript for keyword matches and adds those sentences as evidence.

```
┌─────────────────────────────────────────────────────────────────┐
│                     EVIDENCE EXTRACTION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. LLM Extraction (semantic analysis)                          │
│     └── LLM reads transcript, extracts quotes per PHQ-8 item    │
│                                                                 │
│  2. Keyword Backfill (rule-based) ← THIS DOCUMENT               │
│     └── If LLM missed evidence: scan for keyword matches        │
│                                                                 │
│  3. Scoring                                                     │
│     └── Score items with evidence; N/A for items without        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Does Backfill Exist?

### The Problem: LLM Extraction Isn't Perfect

LLMs can miss obvious evidence:
- "I'm so exhausted" might not be mapped to `PHQ8_Tired`
- "Can't sleep at night" might be overlooked for `PHQ8_Sleep`
- Colloquial language might not trigger semantic matching

### The Solution: Rule-Based Backup

When the LLM returns empty evidence for a symptom, we search for known keywords:

| Domain | Example Keywords |
|--------|------------------|
| Sleep | "can't sleep", "insomnia", "wake up tired" |
| Tired | "exhausted", "no energy", "drained" |
| Depressed | "hopeless", "crying", "feeling down" |
| Appetite | "no appetite", "overeating", "lost weight" |

**Source**: `src/ai_psychiatrist/resources/phq8_keywords.yaml`

---

## How It Works

### The Algorithm

From `src/ai_psychiatrist/agents/quantitative.py` (keyword backfill helpers):

```python
def _find_keyword_hits(self, transcript: str, cap: int = 3) -> dict[str, list[str]]:
    """Find keyword-matched sentences per PHQ-8 item (substring match)."""
    parts = re.split(r"(?<=[.?!])\s+|\n+", transcript.strip())
    sentences = [p.strip() for p in parts if p and len(p.strip()) > 0]

    hits: dict[str, list[str]] = {}
    for key, keywords in DOMAIN_KEYWORDS.items():
        key_hits: list[str] = []
        for sent in sentences:
            sent_lower = sent.lower()
            if any(kw in sent_lower for kw in keywords):
                key_hits.append(sent)
            if len(key_hits) >= cap:
                break
        hits[key] = key_hits
    return hits


def _merge_evidence(
    self,
    current: dict[str, list[str]],
    hits: dict[str, list[str]],
    cap: int = 3,
) -> dict[str, list[str]]:
    """Merge keyword hits into LLM evidence, respecting a total per-item cap."""
    out = {k: list(v) for k, v in current.items()}
    for key, key_hits in hits.items():
        current_items = out.get(key, [])
        if len(current_items) >= cap:
            continue
        need = cap - len(current_items)
        existing = set(current_items)
        new_hits = [h for h in key_hits if h not in existing]
        out[key] = current_items + new_hits[:need]
    return out
```

### Key Properties

1. **Runs when evidence is insufficient**: If LLM found `cap` quotes, backfill skips that item
2. **Caps at 3 evidence items per domain** (LLM + keyword): Prevents prompt bloat
3. **Simple substring matching**: Fast, deterministic, no LLM calls
4. **Deduplicates**: Won't add the same sentence twice

---

## The Tradeoff: Coverage vs Purity

### Without Backfill (Default - Paper Parity)

| Pros | Cons |
|------|------|
| Measures pure LLM capability | Lower coverage (~50%) |
| Matches paper methodology | More N/A results |
| Cleaner research comparison | Less clinical utility |
| What the paper actually tested | Misses some obvious evidence |

### With Backfill (Higher Coverage)

| Pros | Cons |
|------|------|
| Higher coverage (~74%) | Measures "LLM + heuristics" |
| Catches LLM blind spots | May include irrelevant matches |
| More clinical utility | Diverges from paper methodology |
| More items get assessed | Harder to compare with paper |

---

## Paper's Approach

**Unknown.** The paper does not explicitly describe a keyword backfill mechanism.

What the paper *does* state is that when evidence is insufficient, the system produces no
prediction for that item (resulting in substantial missingness / “no output” for many
items):

> "If no relevant evidence was found for a given PHQ-8 item, the model produced no output."

And in Section 3.2:

> "In ~50% of cases, the model was unable to provide a prediction due to insufficient evidence."

For paper-parity experiments, the safest assumption is that the evaluation intended to
measure **pure LLM extraction/scoring behavior**, without additional heuristic evidence
injection. **Backfill is now OFF by default** per [SPEC-003](../specs/SPEC-003-backfill-toggle.md).

---

## When to Use Each Mode

### Use Backfill OFF (Default) When:
- Reproducing paper results
- Evaluating LLM capability
- Running ablation studies
- Comparing different LLM models

### Use Backfill ON When:
- Building a clinical decision support tool
- Want maximum coverage
- Prefer "some assessment" over "N/A"
- Not comparing directly with paper metrics

---

## N/A Reason Tracking

With [SPEC-003](../specs/SPEC-003-backfill-toggle.md) implemented, each N/A result can include a deterministic reason (when enabled):

| Reason | Description |
|--------|-------------|
| `NO_MENTION` | No evidence from LLM and no keyword matches found |
| `LLM_ONLY_MISSED` | LLM found no evidence, but keyword hits exist (backfill OFF) |
| `KEYWORDS_INSUFFICIENT` | Keywords matched and were provided (backfill ON), but scoring still returned N/A |
| `SCORE_NA_WITH_EVIDENCE` | LLM extracted evidence, but scoring still returned N/A |

This enables:
- Debugging extraction failures
- Understanding model behavior
- Comparing backfill ON vs OFF
- Identifying keyword list gaps

---

## Configuration

Backfill is **OFF by default** (paper parity mode):

```bash
# Default: paper parity mode (pure LLM)
# QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false  # (this is the default)

# Enable backfill for higher coverage
QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=true
```

---

## Impact on Results

### Example Run (2025-12-23)

| Mode | Coverage | Item MAE | Items Assessed |
|------|----------|----------|----------------|
| Backfill ON | 74.1% | 0.753 | 243/328 |
| Backfill OFF | ~50%* | ~0.62* | ~164/328* |

*Estimated based on paper; ablation needed to confirm.

### Per-Item Impact

Items most affected by backfill (highest keyword contribution):

| Item | With Backfill | Without (Est.) |
|------|---------------|----------------|
| Appetite | 34% coverage | <10%* |
| Moving | 44% coverage | ~30%* |
| Concentrating | 51% coverage | ~35%* |

---

## Decision Tree

```
Transcript → LLM Evidence Extraction
                    │
                    ▼
         ┌─────────────────────────┐
         │ LLM found evidence?     │
         └─────────────────────────┘
                │           │
               YES          NO
                │           │
                ▼           ▼
         Use LLM      ┌─────────────────────────┐
         evidence     │ Backfill enabled?       │
                      └─────────────────────────┘
                             │           │
                            YES          NO
                             │           │
                             ▼           ▼
                      Scan for      Return empty
                      keywords      evidence
                             │           │
                             ▼           ▼
                      ┌─────────────────────────┐
                      │ Keywords found?         │
                      └─────────────────────────┘
                             │           │
                            YES          NO
                             │           │
                             ▼           ▼
                      Add keyword    N/A (NO_MENTION
                      matches to     or LLM_ONLY_MISSED)
                      evidence
                             │
                             ▼
                      Score with
                      LLM (0-3 or N/A)
                             │
                             ▼
                   If N/A with evidence:
                 SCORE_NA_WITH_EVIDENCE
```

---

## Related Documentation

- [SPEC-003: Backfill Toggle](../specs/SPEC-003-backfill-toggle.md) - Implementation specification
- [Coverage Investigation](../bugs/coverage-investigation.md) - Why our coverage differs
- [Extraction Mechanism](./extraction-mechanism.md) - Full extraction pipeline
- [Paper Parity Guide](../guides/paper-parity-guide.md) - How to reproduce paper results
