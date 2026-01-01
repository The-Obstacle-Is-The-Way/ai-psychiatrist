# Keyword Backfill: The Safety Net for Evidence Extraction

> **DEPRECATED FEATURE - DO NOT ENABLE**
>
> Keyword backfill is a flawed heuristic that inflates coverage metrics without
> improving clinical validity. The feature matches keywords like "sleep" or "tired"
> via simple substring matching, leading to false positives and misleading results.
>
> **This feature is retained for historical comparison only. Keep it OFF.**
>
> The original paper's methodology has fundamental issues that make "paper parity"
> meaningless. See `HYPOTHESIS-FEWSHOT-DESIGN-FLAW.md` and `POST-ABLATION-DEFAULTS.md` for details.

**Audience**: Researchers and developers wanting to understand the coverage-accuracy tradeoff
**Related**: [Extraction Mechanism](./extraction-mechanism.md) | [Configuration Reference](../configs/configuration.md) | [Coverage Explained](./coverage-explained.md)
**Last Updated**: 2026-01-01

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
2. **Caps at `cap` evidence items per domain** (LLM + keyword; default cap=3): Prevents prompt bloat
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
│  STEP 2: KEYWORD HIT DETECTION (Lines 127-136)                          │
│  ─────────────────────────────────────────────────────────────────────  │
│  IF enable_keyword_backfill=TRUE or track_na_reasons=TRUE:              │
│    • _find_keyword_hits() scans transcript for YAML keywords            │
│    • Returns: {"PHQ8_Tired": ["I'm exhausted"], ...}                    │
│    • Uses substring matching (case-insensitive)                         │
│                                                                         │
│                              │                                          │
│                              ▼                                          │
│                                                                         │
│  STEP 3: CONDITIONAL BACKFILL MERGE (Lines 137-145)                     │
│  ─────────────────────────────────────────────────────────────────────  │
│  IF enable_keyword_backfill=TRUE:                                       │
│    • _merge_evidence() adds keyword hits to LLM evidence                │
│    • Caps at keyword_backfill_cap (default 3) (prevents prompt bloat)   │
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
    final_evidence = self._merge_evidence(
        llm_evidence,
        keyword_hits,
        cap=self._settings.keyword_backfill_cap,
    )
else:
    final_evidence = llm_evidence  # Pure LLM only - paper-text parity mode
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

### Without Backfill (Default - Paper-Text Parity)

| Pros | Cons |
|------|------|
| Measures pure LLM capability | Coverage depends on model/runtime; paper reports “~50% of cases unable to provide a prediction” (denominator unclear), while our observed item-level coverage was 69.2% over evaluated subjects |
| Matches paper-text methodology (as written) | More N/A results than enabling backfill (typical) |
| Cleaner research comparison | Less clinical utility |
| What the paper text describes | Misses some obvious evidence |

### With Backfill (Higher Coverage)

| Pros | Cons |
|------|------|
| Often higher coverage (e.g., 74.1% in one historical run recorded in `docs/results/reproduction-results.md`) | Measures "LLM + heuristics" |
| Catches LLM blind spots | May include irrelevant matches |
| More clinical utility | Diverges from paper-text methodology; closer to paper-repo behavior |
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
3. Backfill can still be enabled for historical ablations (not recommended for new work)

**Backfill is OFF by default** (`QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false`).

### Asking the Authors

To resolve this ambiguity, consider asking:

> "When the LLM couldn't find evidence for a PHQ-8 item, did you use keyword-based
> fallback to recover missed evidence before scoring? The public repo has
> `_keyword_backfill()` in the few-shot agent but the paper doesn't mention it.
> Was that used in the reported 0.619 MAE evaluation?"

---

## When to Use Each Mode

### Use Backfill OFF (Default) — ALWAYS

**This is the only recommended mode.** Backfill is deprecated.

- Evaluating LLM capability
- Running ablation studies
- Comparing different LLM models
- Production use

### Use Backfill ON — NEVER (Historical Only)

**Do not enable backfill.** The feature is flawed:

- Simple substring matching has no semantic understanding
- Matches "I sleep great" for `PHQ8_Sleep` (false positive)
- Inflates coverage without improving clinical validity
- The original paper's methodology is not reproducible anyway

If you need to enable backfill for a specific historical comparison, document
the reason explicitly. This is not recommended for any new work.

---

## N/A Reason Tracking

When N/A reason tracking is enabled (`QUANTITATIVE_TRACK_NA_REASONS=true`), each N/A result can include a deterministic reason (independent of backfill):

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

Backfill is **OFF by default** (paper-text parity mode):

```bash
# Default: paper-text parity mode (pure LLM)
# QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false  # (this is the default)

# Historical ablation only (deprecated):
# QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=true
```

> ⚠️ **Deprecated**: Enabling backfill is not recommended. It was used for historical ablation studies only. Keep it OFF for all new runs.

---

## Impact on Results

### Concrete Runs (Local Output + Historical Notes)

| Run | Mode | Coverage | Item MAE | Notes |
|-----|------|----------|---------|-------|
| 2025-12-24 (paper split, backfill OFF) | Paper-text parity | 69.2% (216/312) | 0.778 | Historical run notes only; the JSON artifact is not tracked in this repo snapshot (older runners used different output naming) |
| 2025-12-23 (historical, backfill ON) | Heuristic-augmented | 74.1% (243/328) | 0.757 | Recorded in `docs/results/reproduction-results.md` (no JSON artifact stored under `data/outputs/` in this repo snapshot) |

### Per-Item Impact

In the 2025-12-24 paper-text-parity run (backfill OFF), these items had the lowest coverage:

| Item | Coverage (backfill OFF) |
|------|--------------------------|
| Appetite | 25.6% |
| Moving | 41.0% |
| Concentrating | 43.6% |

Backfill is expected to increase some of these, but that requires a controlled ablation (same split,
same model/backend, backfill ON vs OFF) to attribute causality.

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

- [Extraction Mechanism](./extraction-mechanism.md) - Full extraction pipeline
- [Paper Parity Guide](../guides/paper-parity-guide.md) - How to reproduce paper results
- [Configuration Reference](../configs/configuration.md) - All settings
