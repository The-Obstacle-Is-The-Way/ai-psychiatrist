# Keyword Backfill: The Safety Net for Evidence Extraction

**Audience**: Researchers and developers wanting to understand the coverage-accuracy tradeoff
**Related**: [SPEC-003](../specs/SPEC-003-backfill-toggle.md) | [Coverage Investigation](../bugs/coverage-investigation.md) | [Extraction Mechanism](./extraction-mechanism.md)
**Last Updated**: 2025-12-23

---

## What Is Keyword Backfill?

Keyword backfill is a **rule-based safety net** that runs after LLM evidence extraction. When the LLM fails to find evidence for a PHQ-8 symptom, backfill scans the transcript for keyword matches and adds those sentences as evidence.

```
┌─────────────────────────────────────────────────────────────────┐
│                     EVIDENCE EXTRACTION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. LLM Extraction (semantic analysis)                          │
│     └── LLM reads transcript, extracts quotes per PHQ-8 item    │
│                                                                  │
│  2. Keyword Backfill (rule-based) ← THIS DOCUMENT               │
│     └── If LLM missed evidence: scan for keyword matches        │
│                                                                  │
│  3. Scoring                                                      │
│     └── Score items with evidence; N/A for items without        │
│                                                                  │
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

From `src/ai_psychiatrist/agents/quantitative.py`:

```python
def _keyword_backfill(
    self,
    transcript: str,
    current: dict[str, list[str]],
    cap: int = 3,
) -> dict[str, list[str]]:
    """Add keyword-matched sentences when LLM misses evidence."""

    # 1. Split transcript into sentences
    sentences = re.split(r"(?<=[.?!])\s+|\n+", transcript.strip())

    # 2. For each PHQ-8 domain with insufficient evidence
    for key, keywords in DOMAIN_KEYWORDS.items():
        need = max(0, cap - len(current.get(key, [])))
        if need == 0:
            continue  # Already have enough evidence

        # 3. Find sentences containing keywords
        hits = []
        for sent in sentences:
            if any(kw in sent.lower() for kw in keywords):
                hits.append(sent)
            if len(hits) >= need:
                break

        # 4. Merge with existing evidence (deduplicated)
        if hits:
            existing = set(current.get(key, []))
            merged = current.get(key, []) + [h for h in hits if h not in existing]
            current[key] = merged[:cap]

    return current
```

### Key Properties

1. **Only runs when LLM misses evidence**: If LLM found 3+ quotes, backfill skips that domain
2. **Caps at 3 sentences per domain**: Prevents prompt bloat
3. **Simple substring matching**: Fast, deterministic, no LLM calls
4. **Deduplicates**: Won't add the same sentence twice

---

## The Tradeoff: Coverage vs Purity

### With Backfill (Default)

| Pros | Cons |
|------|------|
| Higher coverage (~74%) | Measures "LLM + heuristics" |
| Catches LLM blind spots | May include irrelevant matches |
| More clinical utility | Diverges from paper methodology |
| More items get assessed | Harder to compare with paper |

### Without Backfill (Paper Parity)

| Pros | Cons |
|------|------|
| Measures pure LLM capability | Lower coverage (~50%) |
| Matches paper methodology | More N/A results |
| Cleaner research comparison | Less clinical utility |
| What the paper actually tested | Misses some obvious evidence |

---

## Paper's Approach

The paper does **not** use keyword backfill. Section 2.3.2:

> "If no relevant evidence was found for a given PHQ-8 item, the model produced no output."

Section 3.2 reports:

> "In ~50% of cases, the model was unable to provide a prediction due to insufficient evidence."

This is by design - the paper tests **LLM capability**, not production utility.

---

## When to Use Each Mode

### Use Backfill ON (Default) When:
- Building a clinical decision support tool
- Want maximum coverage
- Prefer "some assessment" over "N/A"
- Not comparing directly with paper metrics

### Use Backfill OFF When:
- Reproducing paper results
- Evaluating LLM capability
- Running ablation studies
- Comparing different LLM models

---

## N/A Reason Tracking (Planned)

> **⚠️ NOT YET IMPLEMENTED** - See [SPEC-003](../specs/SPEC-003-backfill-toggle.md)

Currently, when an item returns N/A, we **do not** track why. After SPEC-003 is implemented, each N/A will include a reason:

| Reason | Description |
|--------|-------------|
| `NO_MENTION` | No evidence from LLM and no keyword matches |
| `LLM_ONLY_MISSED` | LLM missed it but keywords found it (backfill OFF) |
| `KEYWORDS_INSUFFICIENT` | Keywords matched but still insufficient |
| `SCORING_REFUSED` | Evidence exists but LLM declined to score |

This will enable:
- Debugging extraction failures
- Understanding model behavior
- Comparing backfill ON vs OFF
- Identifying keyword list gaps

---

## Configuration

Currently backfill **always runs**. After SPEC-003 is implemented:

```bash
# Paper parity mode (pure LLM)
QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false

# Production mode (default)
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
```

---

## Related Documentation

- [SPEC-003: Backfill Toggle](../specs/SPEC-003-backfill-toggle.md) - Implementation specification
- [Coverage Investigation](../bugs/coverage-investigation.md) - Why our coverage differs
- [Extraction Mechanism](./extraction-mechanism.md) - Full extraction pipeline
- [Paper Parity Guide](../guides/paper-parity-guide.md) - How to reproduce paper results
