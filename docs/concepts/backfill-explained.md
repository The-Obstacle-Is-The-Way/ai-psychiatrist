# Keyword Backfill: The Safety Net for Evidence Extraction

**Audience**: Researchers and developers wanting to understand the coverage-accuracy tradeoff
**Related**: [SPEC-003](../specs/SPEC-003-backfill-toggle.md) | [Coverage Investigation](../bugs/coverage-investigation.md) | [Extraction Mechanism](./extraction-mechanism.md)
**Last Updated**: 2025-12-24

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

## Detailed Quantitative Agent Pipeline (With Code References)

This section shows exactly where backfill fits in the quantitative assessment flow,
with line number references to `src/ai_psychiatrist/agents/quantitative.py`.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    QUANTITATIVE AGENT PIPELINE                          │
│                    (quantitative.py:102-254)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STEP 1: LLM EVIDENCE EXTRACTION (Lines 256-307)                        │
│  ─────────────────────────────────────────────────────────────────────  │
│  • _extract_evidence() calls LLM with transcript                        │
│  • LLM returns JSON: {"PHQ8_Sleep": ["I can't sleep"], ...}             │
│  • Pure semantic analysis - no keywords involved                        │
│                                                                         │
│                              │                                          │
│                              ▼                                          │
│                                                                         │
│  STEP 2: KEYWORD HIT DETECTION (Lines 127-135)                          │
│  ─────────────────────────────────────────────────────────────────────  │
│  IF enable_keyword_backfill=TRUE or track_na_reasons=TRUE:              │
│    • _find_keyword_hits() scans transcript for YAML keywords            │
│    • Returns: {"PHQ8_Tired": ["I'm exhausted"], ...}                    │
│    • Uses substring matching (case-insensitive)                         │
│                                                                         │
│                              │                                          │
│                              ▼                                          │
│                                                                         │
│  STEP 3: CONDITIONAL BACKFILL MERGE (Lines 137-144)                     │
│  ─────────────────────────────────────────────────────────────────────  │
│  IF enable_keyword_backfill=TRUE:                                       │
│    • _merge_evidence() adds keyword hits to LLM evidence                │
│    • Caps at 3 quotes per item (prevents prompt bloat)                  │
│    • Deduplicates (won't add same sentence twice)                       │
│  ELSE:                                                                  │
│    • final_evidence = llm_evidence (no backfill)                        │
│                                                                         │
│                              │                                          │
│                              ▼                                          │
│                                                                         │
│  STEP 4: FEW-SHOT REFERENCE RETRIEVAL (Lines 156-165)                   │
│  ─────────────────────────────────────────────────────────────────────  │
│  IF mode=FEW_SHOT and embedding_service exists:                         │
│    • Embed the extracted evidence                                       │
│    • Find similar reference chunks from training set                    │
│    • Build reference bundle with example scores                         │
│                                                                         │
│                              │                                          │
│                              ▼                                          │
│                                                                         │
│  STEP 5: LLM SCORING (Lines 167-184)                                    │
│  ─────────────────────────────────────────────────────────────────────  │
│  • LLM receives: evidence + reference examples (if few-shot)            │
│  • Returns: {"PHQ8_Sleep": {"evidence": "...", "score": 2}, ...}        │
│  • Items without evidence → "N/A"                                       │
│                                                                         │
│                              │                                          │
│                              ▼                                          │
│                                                                         │
│  STEP 6: N/A REASON ASSIGNMENT (Lines 571-584)                          │
│  ─────────────────────────────────────────────────────────────────────  │
│  IF score is N/A and track_na_reasons=TRUE:                             │
│    • _determine_na_reason() classifies why                              │
│    • NO_MENTION: no LLM evidence AND no keyword hits                    │
│    • LLM_ONLY_MISSED: keywords existed but backfill was OFF             │
│    • KEYWORDS_INSUFFICIENT: backfill ON but scorer still said N/A       │
│    • SCORE_NA_WITH_EVIDENCE: LLM had evidence but still said N/A        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Key Code (assess() method, lines 102-254)

```python
# Step 1: LLM extraction (pure semantic)
llm_evidence = await self._extract_evidence(transcript.text)
llm_counts = {k: len(v) for k, v in llm_evidence.items()}

# Step 2: Find keyword hits (always computed if backfill OR tracking enabled)
keyword_hits: dict[str, list[str]] = {}
if self._settings.enable_keyword_backfill or self._settings.track_na_reasons:
    keyword_hits = self._find_keyword_hits(
        transcript.text,
        cap=self._settings.keyword_backfill_cap,
    )

# Step 3: Conditional merge (THE BACKFILL DECISION)
if self._settings.enable_keyword_backfill:
    final_evidence = self._merge_evidence(llm_evidence, keyword_hits, cap=3)
else:
    final_evidence = llm_evidence  # Pure LLM only - paper parity mode
```

### File Reference Table

| Component | File | Lines | What It Does |
|-----------|------|-------|--------------|
| Backfill toggle | `config.py` | 257-260 | `enable_keyword_backfill` setting |
| Main pipeline | `quantitative.py` | 102-254 | `assess()` orchestrates all steps |
| LLM extraction | `quantitative.py` | 256-307 | `_extract_evidence()` |
| Keyword search | `quantitative.py` | 309-339 | `_find_keyword_hits()` |
| Evidence merge | `quantitative.py` | 341-374 | `_merge_evidence()` |
| N/A classification | `quantitative.py` | 571-584 | `_determine_na_reason()` |
| Keyword YAML | `resources/phq8_keywords.yaml` | all | 200+ curated keyword phrases |
| YAML loader | `prompts/quantitative.py` | ~50-70 | Loads YAML as packaged resource |

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

### What the Paper TEXT Says

The paper does not explicitly describe a keyword backfill mechanism. What it states:

> "If no relevant evidence was found for a given PHQ-8 item, the model produced no output."

And in Section 3.2:

> "In ~50% of cases, the model was unable to provide a prediction due to insufficient evidence."

### What the Paper CODE Does (Discovery: 2025-12-24)

**The public repository has backfill ALWAYS ON.** In `_reference/agents/quantitative_assessor_f.py`,
the few-shot agent unconditionally calls `_keyword_backfill()` inside `extract_evidence()` (~line 478).

This means:
- The paper TEXT implies pure LLM extraction
- The paper CODE uses LLM + keyword heuristics

We don't know which approach produced the reported MAE of 0.619.

### Our Decision: Paper-Text Parity by Default

We chose to measure **pure LLM capability** (backfill OFF) because:
1. The paper doesn't describe keyword backfill as part of the methodology
2. Scientific reproducibility should match documented methodology
3. We can always enable backfill for clinical utility

**Backfill is OFF by default** per [SPEC-003](../specs/SPEC-003-backfill-toggle.md).

### Asking the Authors

To resolve this ambiguity, consider asking:

> "When the LLM couldn't find evidence for a PHQ-8 item, did you use keyword-based
> fallback to recover missed evidence before scoring? The public repo has
> `_keyword_backfill()` in the few-shot agent but the paper doesn't mention it.
> Was that used in the reported 0.619 MAE evaluation?"

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
