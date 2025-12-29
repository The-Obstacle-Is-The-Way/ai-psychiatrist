# BUG-031: Few-Shot Retrieval Issues

**Status**: OPEN - Awaiting Senior Review
**Severity**: HIGH - Potential contributor to zero-shot outperforming few-shot
**Discovered**: 2025-12-28
**Related**: [Investigation Document](../brainstorming/investigation-zero-shot-beats-few-shot.md)

---

## Executive Summary

Investigation into why zero-shot (AURC 0.134) outperforms few-shot (AURC 0.214) in our runs identified paper-parity divergences in the embedding-based retrieval mechanism:

### Issues Found

| Issue | Type | Impact |
|-------|------|--------|
| **Score-Chunk Mismatch** | Paper methodology (correctly implemented) | Participant-level scores for chunk-level matches |
| **Format Mismatch** | OUR DIVERGENCE | 8 separate sections vs paper's 1 unified block |
| **Missing Domain Labels** | OUR DIVERGENCE | `(Score: 2)` vs paper's `(PHQ8_Sleep Score: 2)` |

### Critical Distinction

**Issue 1 is NOT a bug in our code** - it's the paper's methodology. From Section 2.4.2:
> "For each chunk, we identified its associated participant ID in the dataset and **attached its ground-truth PHQ-8 score**."

The paper intentionally uses participant-level scores. We correctly implemented this.

**Issues 2 and 3 ARE divergences** - our format differs from the paper's notebook implementation.

### Epistemic Status

**HYPOTHESIS, NOT PROVEN.** We have not yet run ablations to prove these divergences *caused* the performance inversion. Correlation ≠ causation.

---

## Verified Divergences

### Paper Methodology (Section 2.4.2)

> "For each chunk, we identified its associated participant ID in the dataset and **attached its ground-truth PHQ-8 score**."

### Notebook Implementation (embedding_quantitative_analysis.ipynb)

From cell `49f51ff5`, the `process_evidence_for_references` function:

```python
for chunk_info in similar_chunks:
    participant_id = chunk_info['participant_id']
    raw_text = chunk_info['raw_text']

    # Look up PARTICIPANT-LEVEL ground truth (matches paper Section 2.4.2)
    participant_data = phq8_ground_truths.loc[
        phq8_ground_truths['Participant_ID'] == participant_id
    ]

    if not participant_data.empty:
        # Get the PARTICIPANT's overall score - this is the paper's methodology
        score = int(participant_data[evidence_key].values[0])
        reference_entry = f"({evidence_key} Score: {score})\n{raw_text}"
```

**Verified**: Both paper text and notebook code attach participant-level scores to chunk matches. We correctly implement this.

### Format Divergence (Same Notebook Cell)

```python
# Paper's format: SINGLE unified block, same tag opens and closes
reference_evidence = "<Reference Examples>\n\n" + "\n\n".join(all_references) + "\n\n<Reference Examples>"
```

**Note**: Paper uses `<Reference Examples>` for BOTH opening AND closing (not `</Reference Examples>`). This unusual format may be intentional as a delimiter.

**Verified divergences**:
1. Paper uses single `<Reference Examples>` block, we use 8 separate blocks
2. Paper uses same tag to open and close, we use XML-style `</Reference Examples>`
3. Paper omits items with no evidence/matches, we emit per-item `"No valid evidence found"` blocks

---

## Data Structure Analysis

The current data structure only supports participant-level scoring.

### Embeddings JSON (`data/embeddings/paper_reference_embeddings.json`)

```json
{
  "303": [
    "Ellie: hi i'm ellie thanks for coming...",   // Just text, NO score
    "Participant: okay how 'bout yourself...",    // Just text, NO score
    ...
  ],
  "304": [...]
}
```

**Chunks are PLAIN TEXT.** No scores embedded. No chunk identifiers.

### Ground Truth CSV (`data/train_split_Depression_AVEC2017.csv`)

```
Participant_ID,PHQ8_NoInterest,PHQ8_Depressed,PHQ8_Sleep,...
303,0,0,0,...
304,0,1,1,...
321,2,3,3,...  ← Severe depression
```

**One row per participant.** Scores are participant-level only.

### Score Lookup Code (`reference_store.py:563-589`)

```python
def get_score(self, participant_id: int, item: PHQ8Item) -> int | None:
    df = self._load_scores()  # Load CSV
    row = df[df["Participant_ID"] == participant_id]
    return int(row[col_name].iloc[0])  # Participant-level lookup
```

**Scores are keyed by `(participant_id, item)` — NOT by chunk.**

### Concrete Example: Participant 321 (PHQ8_Sleep=3)

Participant 321 has **115+ chunks** from their interview:
- **~7% discuss sleep** (severe insomnia, waking every 1-3 hours)
- **~93% discuss other topics** (work, family, PTSD history, hobbies)

**ALL 115 chunks get attached Score 3 (PHQ8_Sleep)** when retrieved for sleep queries.

A chunk like:
> "I'm proud of my children and grandchildren"

Gets attached: `(PHQ8_Sleep Score: 3)` ← **Makes no sense**

### Architectural Constraint

| Component | What It Contains | Chunk-Level Scores? |
|-----------|------------------|---------------------|
| `paper_reference_embeddings.json` | Text chunks only | **NO** |
| `train_split_Depression_AVEC2017.csv` | Participant-level PHQ-8 | **NO** |
| Score lookup | `get_score(participant_id, item)` | **NO** |

**Chunk-level scoring would require**:
1. New data structure with chunk IDs and per-chunk scores
2. LLM annotation of each chunk during embedding generation
3. Architectural changes

This is not supported by the current data structure. The paper's methodology uses participant-level scores, which we correctly implement.

---

## Hypothesis: How Score-Chunk Mismatch May Cause Confusion

### The Potential Problem (Hypothesis)

Imagine a participant who says in their interview:

- **Minute 5**: "I slept fine last week"
- **Minute 15**: "I've been having terrible insomnia for months"
- **Minute 30**: "Yeah the sleep thing is really bad, every single night"

Their **overall PHQ8_Sleep score**: 3 (nearly every day) - because the WHOLE interview reveals severe sleep problems.

**The paper's approach**:
1. Chunk the transcript into 8-line windows
2. Find a chunk that's similar to your query
3. Attach the participant's **OVERALL** score to that tiny chunk

**So the LLM might see**:

```
(Score: 3)
"I slept fine last week, just had one bad night after coffee"
```

**The LLM thinks**: "Wait... 'slept fine' with 'one bad night' = Score 3 (nearly every day)??"

**That makes no sense.** The score doesn't match the chunk.

### Why This May Break Few-Shot Learning (Hypothesis)

Few-shot works by showing the LLM: "Here's an example, here's the score, learn the pattern."

**If** the examples are semantically contradictory:
- "Occasional coffee-related sleep issue" → Score 3
- "I can't sleep at all every night" → Score 1

The LLM may learn inconsistent patterns.

**Caveat**: We have not empirically verified that retrieved chunks are actually contradictory. This requires retrieval audits (logging retrieved chunks + manual review).

---

## Paper vs Our Results

### Paper's Claim

| Mode | MAE | Improvement |
|------|-----|-------------|
| Zero-shot | 0.796 | - |
| Few-shot | 0.619 | 22% |

### Different Metrics Approaches

| Metric | What It Measures | When Valid |
|--------|------------------|------------|
| MAE (paper) | Error on non-N/A predictions | Valid if coverages are similar |
| AURC/AUGRC (ours) | Integrated risk over all coverage levels | Better when coverages differ |

**Note**: MAE at Cmax is the "selective risk" at maximum coverage. It's not invalid, but incomplete when comparing systems with different coverage rates.

### Our Findings

| Mode | AURC | Coverage |
|------|------|----------|
| Zero-shot | **0.134** | 55.5% |
| Few-shot | 0.214 | 71.9% |

**In our runs**: Zero-shot has lower AURC (better). The 95% CIs do not overlap, suggesting significance.

**Caveats**:
1. Bootstrap CIs capture participant sampling uncertainty, not LLM stochasticity
2. We have not run multiple runs to assess LLM variance
3. Paired deltas on same participants would be more rigorous than non-overlapping CIs

---

## Issue 1: Score-Chunk Mismatch (Paper Methodology)

### Status: CORRECTLY IMPLEMENTED

This is **NOT a bug in our code**. The paper explicitly describes this behavior in Section 2.4.2:

> "For each chunk, we identified its associated participant ID in the dataset and **attached its ground-truth PHQ-8 score**."

We correctly implemented this. However, the design itself may be flawed.

### Location
`src/ai_psychiatrist/services/embedding.py:199`

### The Design

```python
# In _compute_similarities() method
for participant_id, chunks in all_refs.items():
    for chunk_text, embedding in chunks:
        # ... compute similarity ...

        # This follows paper Section 2.4.2: "attached its ground-truth PHQ-8 score"
        score = self._reference_store.get_score(participant_id, lookup_item)

        matches.append(
            SimilarityMatch(
                chunk=TranscriptChunk(text=chunk_text, ...),
                similarity=sim,
                reference_score=score,  # PARTICIPANT-LEVEL per paper design
            )
        )
```

### Why This Design May Be Flawed

When we retrieve a chunk from Participant 789 that is semantically similar to "I can't sleep":

| What we get (per paper) | What might be better |
|-------------------------|----------------------|
| Participant 789's overall PHQ8_Sleep score (e.g., 3) | Score specific to THIS chunk's content |
| May come from OTHER parts of their interview | Should reflect severity described IN this chunk |

### Example

```
Query Evidence: "I wake up at 3am every night" (clearly severe)

Retrieved Chunk (Participant 789):
  "I have trouble falling asleep after drinking coffee"
  (Situational, implies occasional/mild)

Attached Score: 3 (Participant 789's overall PHQ8_Sleep - per paper design)

LLM sees:
  "(Score: 3)
   I have trouble falling asleep after drinking coffee"

LLM interprets: "Occasional coffee-related sleep issues = Score 3??"
Result: CONFUSION
```

### Recommendation

The paper's design creates a semantic mismatch. However:
- **For paper parity**: Keep as-is (we correctly follow the paper)
- **For best practices**: Consider chunk-level scoring (but this diverges from paper)

---

## Bug 2: Reference Format Mismatch

### Location
`src/ai_psychiatrist/services/embedding.py:40-70` (`ReferenceBundle.format_for_prompt`)

### Paper's Format (Source: Notebook cell 49f51ff5)

Single unified block with inline domain labels. Note: Paper uses same tag to open and close:
```text
<Reference Examples>

(PHQ8_Sleep Score: 2)
Patient: I've been having trouble sleeping lately.
Therapist: How many nights a week?
Patient: Maybe 3 or 4 nights.

(PHQ8_Tired Score: 1)
Patient: I feel tired sometimes but I can still function.

(PHQ8_Depressed Score: 3)
Patient: I feel hopeless every single day.

<Reference Examples>
```

### Our Format

8 separate blocks, domain label in section header:
```text
[Sleep]
<Reference Examples>

(Score: 2)
Patient: I've been having trouble sleeping lately.
Therapist: How many nights a week?
Patient: Maybe 3 or 4 nights.

</Reference Examples>

[Tired]
<Reference Examples>

(Score: 1)
Patient: I feel tired sometimes but I can still function.

</Reference Examples>

[Depressed]
<Reference Examples>

(Score: 3)
Patient: I feel hopeless every single day.

</Reference Examples>
```

### Why This Is Wrong

| Aspect | Paper | Ours |
|--------|-------|------|
| LLM sees | All domains together, can cross-reference | 8 isolated problems |
| Symptom patterns | Can recognize co-occurrence | Compartmentalized |
| Context integration | Holistic psychiatric view | Fragmented assessment |

### Impact

- LLM can't recognize that Sleep + Tired + Depressed often co-occur
- Each domain assessed in isolation, losing clinical context
- Zero-shot (holistic direct analysis) outperforms fragmented few-shot

---

## Bug 3: Missing Domain Labels in Score Tags

### Location
`src/ai_psychiatrist/services/embedding.py:58-62`

### Paper's Format
```python
# Paper notebook:
reference_entry = f"({evidence_key} Score: {score})\n{raw_text}"
# Example: "(PHQ8_Sleep Score: 2)"
```

### Our Format
```python
# Our code:
score_text = f"(Score: {match.reference_score})"
# Example: "(Score: 2)"
```

### Why This Is Wrong

- Paper's inline label: `(PHQ8_Sleep Score: 2)` - domain embedded in tag
- Our label: `(Score: 2)` - domain only in section header
- LLM can't easily map score back to domain within the chunk context

---

## Root Cause Analysis

### Architectural Issue

The reference embedding system has a **fundamental mismatch**:

| Component | What We Store | What We Query | What We Attach |
|-----------|---------------|---------------|----------------|
| Chunks | Generic 8-line windows | Item-specific evidence | Participant-level scores |

Chunks are NOT tagged with PHQ-8 items at generation time. The score lookup happens at retrieval time and uses participant-level ground truth, not chunk-level analysis.

### Why Zero-Shot Wins

| Zero-Shot | Few-Shot |
|-----------|----------|
| LLM analyzes evidence directly | LLM sees conflicting references |
| No mismatched score signals | Scores don't match chunk content |
| Holistic transcript analysis | Fragmented 8-domain structure |
| Works as designed | Confused by reference quality |

---

## Evidence

### Statistical Evidence

| Mode | AURC | 95% CI | Coverage |
|------|------|--------|----------|
| Zero-shot | **0.134** | [0.094, 0.176] | 55.5% |
| Few-shot | 0.214 | [0.160, 0.278] | 71.9% |

- Non-overlapping CIs suggest significant difference (but paired deltas would be more rigorous)
- Few-shot predicts MORE (higher coverage) but with higher risk per unit coverage
- This pattern is *consistent with* overconfidence, but not proven to be caused by our divergences

### Code Evidence

1. `embedding.py:199` - `get_score(participant_id, lookup_item)` returns participant-level score
2. `reference_store.py:563-589` - `get_score()` queries ground truth CSV, not chunk content
3. `generate_embeddings.py:94-135` - Chunks created as generic windows, no item tagging

### Research Evidence

2025 RAG best practices ([LlamaIndex](https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5)):
> "Vital information might not be among the top retrieved chunks, especially if the similarity_top_k setting is as restrictive as 2."

---

## Proposed Fixes (For Senior Review)

### Fix 1: Format Alignment (REQUIRED - Fixes Bugs 2 & 3)

**Priority**: HIGH - This is a real divergence from paper's implementation.

Update `ReferenceBundle.format_for_prompt()` in `embedding.py:40-70`:

**Current (incorrect)**:
```text
[Sleep]
<Reference Examples>

(Score: 2)
{chunk}

</Reference Examples>

[Tired]
<Reference Examples>

(Score: 1)
{chunk}

</Reference Examples>
```

**Target (paper's format)**:
```text
<Reference Examples>

(PHQ8_Sleep Score: 2)
{chunk about sleep}

(PHQ8_Tired Score: 1)
{chunk about fatigue}

<Reference Examples>
```

**Note**: Paper uses `<Reference Examples>` for both opening AND closing (not XML-style `</...>`).

**Changes needed**:
1. Single unified `<Reference Examples>` block
2. Inline domain labels: `(PHQ8_Sleep Score: X)` not `(Score: X)`
3. Remove per-item section headers (e.g., `[Sleep]`)

---

## ✅ ADDED (Senior Review): Implementation-Ready Spec for Fix 1 (Paper Parity)

**Canonical spec**: `docs/specs/31-paper-parity-reference-examples-format.md` (this section should match it).

This section is intentionally **copy/paste-able** and contains the exact behavior required to match the paper notebook.

### Scope (what changes, what doesn’t)

- **Change**: `ReferenceBundle.format_for_prompt()` formatting only.
- **Do NOT change**: retrieval logic, score lookup logic, evidence extraction, `top_k`, or embeddings.
- **Why**: Fix 1 is about **paper-parity prompt formatting**, not redesigning the method.

### Files + exact locations

- **Production code**: `src/ai_psychiatrist/services/embedding.py:40` (`ReferenceBundle.format_for_prompt`)
- **Unit tests**: `tests/unit/services/test_embedding.py:55` (`TestReferenceBundle` expectations)

### Ground truth from paper notebook (cell `49f51ff5`)

The notebook builds the reference string exactly like this:

```python
if all_references:
    reference_evidence = "<Reference Examples>\n\n" + "\n\n".join(all_references) + "\n\n<Reference Examples>"
else:
    reference_evidence = "<Reference Examples>\nNo valid evidence found\n<Reference Examples>"
```

And each `reference_entry` is exactly:

```python
reference_entry = f"({evidence_key} Score: {score})\n{raw_text}"
```

Where `evidence_key` is one of:
`PHQ8_NoInterest, PHQ8_Depressed, PHQ8_Sleep, PHQ8_Tired, PHQ8_Appetite, PHQ8_Failure, PHQ8_Concentrating, PHQ8_Moving`.

### Exact output specification (character-by-character)

Define `entries` as a list of strings. Each entry is:

```text
({EVIDENCE_KEY} Score: {SCORE})
{CHUNK_TEXT}
```

Where:
- `{EVIDENCE_KEY}` must be exactly `PHQ8_{item.value}` (e.g. `PHQ8_Sleep`, `PHQ8_NoInterest`).
- `{SCORE}` must be an **integer 0–3**.
- `{CHUNK_TEXT}` must be the chunk text exactly as stored (including internal newlines).

Then `format_for_prompt()` must return:

- **If at least 1 entry exists**:

```text
<Reference Examples>\n\n
{entry_1}\n\n
{entry_2}\n\n
...\n\n
<Reference Examples>
```

Equivalently (exact):

```python
"<Reference Examples>\n\n" + "\n\n".join(entries) + "\n\n<Reference Examples>"
```

- **If no entries exist**:

```python
"<Reference Examples>\nNo valid evidence found\n<Reference Examples>"
```

### Exact “before” vs “after” code (copy/paste)

**Before** (current behavior; paper-divergent): `src/ai_psychiatrist/services/embedding.py:40`

- Emits **8 sections** (one per PHQ-8 item).
- Emits `(Score: X)` without domain labels.
- Emits XML-style closing tag `</Reference Examples>`.
- Emits `"No valid evidence found"` **inside each empty item block**.

**After** (paper-parity target): replace `ReferenceBundle.format_for_prompt()` with:

```python
def format_for_prompt(self) -> str:
    """Format references as prompt text (paper-parity).

    Paper notebook behavior (cell 49f51ff5):
    - Single unified <Reference Examples> block.
    - Each reference entry is labeled like: (PHQ8_Sleep Score: 2)
    - Items with no matches are omitted (no empty per-item blocks).
    - Uses the same literal tag to open and close: <Reference Examples>
    """
    entries: list[str] = []

    for item in PHQ8Item.all_items():
        evidence_key = f"PHQ8_{item.value}"
        for match in self.item_references.get(item, []):
            # Match notebook behavior: only include references with available ground truth.
            if match.reference_score is None:
                continue
            entries.append(f"({evidence_key} Score: {match.reference_score})\n{match.chunk.text}")

    if entries:
        return "<Reference Examples>\n\n" + "\n\n".join(entries) + "\n\n<Reference Examples>"

    return "<Reference Examples>\nNo valid evidence found\n<Reference Examples>"
```

### Edge cases (explicit behavior)

- **No evidence for an item**: omit that item entirely from the reference block (no empty section).
- **No entries at all**: return exactly `<Reference Examples>\nNo valid evidence found\n<Reference Examples>`.
- **`reference_score is None`**: omit that match (paper notebook only appends entries when ground truth exists).
- **Very low similarity**: no filtering in Fix 1; low-similarity chunks are still included if retrieved.
- **Empty chunk text**: should be impossible in production because `TranscriptChunk` rejects empty text; if it occurs due to bad artifacts, this will raise earlier when the chunk is constructed.

### Verification criteria (how to prove parity)

1. **Unit tests**
   - Update `tests/unit/services/test_embedding.py:55` to reflect the new format (see below).
   - Run: `uv run pytest tests/unit/services/test_embedding.py -q`

2. **Golden-string checks**
   - Add a test that builds a small `ReferenceBundle` and asserts exact string equality including newlines.
   - The test must assert:
     - output starts with `<Reference Examples>\n\n`
     - output ends with `\n\n<Reference Examples>`
     - output contains `"(PHQ8_Sleep Score: 2)\n..."` (not `(Score: 2)`)
     - output does **not** contain any `[` section headers
     - output does **not** contain `</Reference Examples>`

3. **Ablation rerun**
   - Re-run reproduction: `uv run python scripts/reproduce_results.py --split paper-test`
   - Compute paired deltas:
     `uv run python scripts/evaluate_selective_prediction.py --input data/outputs/<RUN>.json --mode zero_shot --input data/outputs/<RUN>.json --mode few_shot --intersection-only`

### Implementation order + dependencies (so you don’t get stuck)

1. **Update production code first** (`src/ai_psychiatrist/services/embedding.py`).
2. **Immediately update unit tests** (`tests/unit/services/test_embedding.py`) so CI goes green.
3. **Run unit tests** (fast feedback): `uv run pytest tests/unit/services/test_embedding.py -q`
4. **Only then run expensive ablations** (`scripts/reproduce_results.py` + `scripts/evaluate_selective_prediction.py`).

Parallelizable work:
- Retrieval diagnostics logging can be implemented in parallel with Fix 1, but interpret logs **only after** Fix 1 lands (otherwise you’re auditing a non-parity prompt).

### Test updates required (exact expectations)

Update `tests/unit/services/test_embedding.py`:

- `test_format_empty_bundle`: must now assert the output is exactly:

```text
<Reference Examples>
No valid evidence found
<Reference Examples>
```

- `test_format_with_matches`: must now assert:
  - output contains `<Reference Examples>`
  - output contains `(PHQ8_NoInterest Score: 2)` for a `PHQ8Item.NO_INTEREST` match
  - output does **not** contain `[NoInterest]`
  - output does **not** contain `</Reference Examples>`

- Add a new test: **skips empty items** (i.e., bundle has `Sleep` matches but `Depressed` has none → output contains only `PHQ8_Sleep` entries).

✅ ADDED (Senior Review): Copy/paste test code (drop-in replacement for `TestReferenceBundle`)

```python
class TestReferenceBundle:
    """Tests for ReferenceBundle."""

    def test_format_empty_bundle(self) -> None:
        """Should format empty bundle as notebook 'no valid evidence' sentinel."""
        bundle = ReferenceBundle(item_references={})
        formatted = bundle.format_for_prompt()
        assert formatted == "<Reference Examples>\nNo valid evidence found\n<Reference Examples>"

    def test_format_with_matches(self) -> None:
        """Should format bundle with labeled references correctly."""
        match = SimilarityMatch(
            chunk=TranscriptChunk(text="I can't enjoy anything anymore", participant_id=123),
            similarity=0.95,
            reference_score=2,
        )

        bundle = ReferenceBundle(item_references={PHQ8Item.NO_INTEREST: [match]})
        formatted = bundle.format_for_prompt()

        assert formatted.startswith("<Reference Examples>\n\n")
        assert formatted.endswith("\n\n<Reference Examples>")
        assert "(PHQ8_NoInterest Score: 2)\nI can't enjoy anything anymore" in formatted
        assert "[NoInterest]" not in formatted
        assert "</Reference Examples>" not in formatted

    def test_format_skips_none_score(self) -> None:
        """Notebook behavior: skip references without available ground truth."""
        match = SimilarityMatch(
            chunk=TranscriptChunk(text="Some text", participant_id=123),
            similarity=0.8,
            reference_score=None,
        )
        bundle = ReferenceBundle(item_references={PHQ8Item.SLEEP: [match]})
        formatted = bundle.format_for_prompt()
        assert formatted == "<Reference Examples>\nNo valid evidence found\n<Reference Examples>"

    def test_format_multiple_matches_and_items(self) -> None:
        """Should include multiple references in a single unified block."""
        sleep_match = SimilarityMatch(
            chunk=TranscriptChunk(text="sleep ref", participant_id=100),
            similarity=0.9,
            reference_score=3,
        )
        tired_match = SimilarityMatch(
            chunk=TranscriptChunk(text="tired ref", participant_id=101),
            similarity=0.85,
            reference_score=1,
        )
        bundle = ReferenceBundle(
            item_references={
                PHQ8Item.SLEEP: [sleep_match],
                PHQ8Item.TIRED: [tired_match],
            }
        )
        formatted = bundle.format_for_prompt()
        assert "(PHQ8_Sleep Score: 3)\nsleep ref" in formatted
        assert "(PHQ8_Tired Score: 1)\ntired ref" in formatted
        assert "[Sleep]" not in formatted
```

---

### Future Work (separate spec required): Item-Tagged Chunks

**Priority**: DEFER (not needed for paper-parity reproduction)

**Status**: NOT IMPLEMENTATION-READY in this document.

This is a legitimate research direction, but it requires a separate design spec (new artifact formats + new indexing pipeline + evaluation protocol). Keeping it here as a “Fix” is misleading.

**Action**: Implement only from a dedicated spec (now tracked as `docs/specs/34-item-tagged-reference-embeddings.md`).

**Why**: A developer should not attempt this based only on BUG-031.

When generating embeddings, tag chunks with which PHQ-8 items they address:

1. **During embedding generation** (`generate_embeddings.py`):
   - For each chunk, use LLM to identify which PHQ-8 items it discusses
   - Store: `{chunk_text, embedding, item_tags: [PHQ8_Sleep, PHQ8_Tired]}`

2. **During retrieval** (`embedding.py`):
   - Only retrieve chunks tagged with the relevant item
   - Ensures retrieved chunk is actually about the queried symptom

3. **Benefits**:
   - Reduces semantic mismatch
   - Chunks are guaranteed to be about the right topic
   - Still uses participant-level scores (paper parity) but with better relevance

---

### Future Work (separate spec required): Chunk-Level Scoring

**Priority**: DEFER (new-method research; high circularity risk)

**Status**: NOT IMPLEMENTATION-READY in this document.

**Tracking spec**: `docs/specs/35-offline-chunk-level-phq8-scoring.md`

This is not a “bug fix” or “paper parity” change. It requires:
- a new labeling pipeline,
- new artifact formats,
- explicit controls against leakage/circularity,
- and separate reporting.

The most correct approach, but diverges significantly from paper:

1. **During embedding generation**:
   - For each chunk, use LLM to assign chunk-specific PHQ-8 scores
   - Store: `{chunk_text, embedding, chunk_scores: {sleep: 2, tired: 1, ...}}`

2. **During retrieval**:
   - Use chunk's own score, not participant's overall score
   - Perfect semantic alignment between chunk content and score

3. **Tradeoffs**:
   - Expensive: requires LLM call per chunk during embedding generation
   - Diverges from paper methodology
   - But: most semantically correct approach

---

### NOT Recommended: Disable Few-Shot

While zero-shot currently outperforms few-shot, this is likely due to the bugs above. After fixing format issues (Fix 1), few-shot may work as intended. Only consider disabling few-shot if fixes don't improve performance.

---

## Alternative Explanations (Not Yet Ruled Out)

The performance inversion may have causes beyond our identified divergences:

| Alternative | Description | How to Test |
|-------------|-------------|-------------|
| **Prompt length/context dilution** | Few-shot adds many tokens; LLM may attend less to transcript | Compare attention patterns or test with shorter references |
| **Scoring prompt mismatch** | Our scoring prompt may differ from notebook beyond reference format | Diff entire prompt structure against notebook |
| **Model/quantization mismatch** | Paper's Gemma3 27B variant/precision may differ | Test with paper's exact model config |
| **Failure rate correlation** | Few-shot had 1 participant failure (390); may correlate with prompt length | Analyze failure patterns |
| **LLM stochasticity** | Single run; LLM variance not captured | Run multiple times, compute variance |

**Until these are ruled out, we cannot claim our divergences are the root cause.**

---

## Action Items

- [ ] **Fix format divergences** - Match paper's unified block format
- [ ] **Run ablation** - Does fixing format improve few-shot?
- [ ] **Add retrieval diagnostics** - Log retrieved chunks + similarity scores
- [ ] **Manual audit** - Review stratified sample of retrieved chunks
- [ ] **Multiple runs** - Assess LLM variance
- [ ] **Paired evaluation** - Use same-participant deltas for significance

---

---

## 2025 State-of-the-Art Solutions

This is a **known problem in RAG** with established solutions.

### The Problem (Literature Terms)

| Our Finding | Literature Term |
|-------------|-----------------|
| Chunks get participant-level scores | Label misalignment |
| Chunk content may not match attached score | Semantic mismatch |
| Retrieval finds topic, not severity | Context loss |

### Solution 1: CRAG (Corrective RAG)

**Tracking spec**: `docs/specs/36-crag-reference-validation.md`

From [LangChain CRAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/):
> "The evaluator is a language model responsible for classifying a retrieved text as correct, incorrect, or ambiguous."

**Architecture**: Add LLM judge to validate chunks before using them.

```
Retrieve chunks → LLM JUDGE → Filter misaligned → Use good chunks
                     ↓
         "Does this chunk match Score 3?"
```

**Performance**: Self-CRAG delivers 320% improvement on PopQA ([Source](https://medium.com/data-science-collective/rag-architectures-a-complete-guide-for-2025-daf98a2ede8c))

### Solution 2: Contextual Retrieval (Anthropic)

From [Anthropic](https://www.anthropic.com/news/contextual-retrieval):
> "Contextual retrieval fixes the problem of lost context by generating and adding a short, context-specific explanation to each chunk before embedding."

**Result**: 49% reduction in retrieval errors, 67% reduction in top-20 failure rate.

### Solution 3: Pre-Compute Chunk Scores

Score each chunk at embedding generation time, not at runtime.

```python
# During embedding generation (one-time cost)
for chunk in all_chunks:
    embedding = embed(chunk.text)
    chunk_scores = llm.score_chunk(chunk.text)  # "What severity does this describe?"
    store(chunk, embedding, chunk_scores)

# During retrieval (zero runtime cost)
for chunk in retrieved_chunks:
    score = chunk.chunk_scores[item]  # Semantically aligned!
```

**Potential benefit**: LLM-estimated chunk scores may better match chunk content than participant-level scores.

**⚠️ WARNING - Circularity Risk**: Using an LLM to fabricate labels that steer another LLM is potentially circular. This could improve apparent performance without improving truth alignment, and diverges significantly from the paper. If implemented, treat as a **new method**, not reproduction.

### Solution 4: Hybrid (Pre-Compute + CRAG)

Best of both worlds:
1. Pre-compute chunk scores at index time (handles 95% of cases)
2. Optional CRAG validation at runtime (safety net for edge cases)

---

## Recommended Implementation

### Priority 1: Fix Format Divergences (REQUIRED for Paper Parity)

**Effort**: Low | **Impact**: Establishes baseline

- Match paper's unified `<Reference Examples>` block format
- Use same tag for open/close: `<Reference Examples>` not `</Reference Examples>`
- Add inline domain labels: `(PHQ8_Sleep Score: 2)` not `(Score: 2)`
- Skip items with no evidence (paper does this)

**After fixing**: Re-run evaluation to see if few-shot improves. This is the only clean way to determine if divergences caused the inversion.

### Priority 2: Add Retrieval Diagnostics (REQUIRED for Causality Claims)

**Effort**: Low | **Impact**: Enables empirical verification

✅ ADDED (Senior Review): Implementation spec is now canonicalized in `docs/specs/32-few-shot-retrieval-diagnostics.md`.

- **File**: `src/ai_psychiatrist/services/embedding.py`
- **Function**: `EmbeddingService.build_reference_bundle` (around `src/ai_psychiatrist/services/embedding.py:238`)
- **Where to add**: immediately after `top_matches = matches[: self._top_k]` (around `src/ai_psychiatrist/services/embedding.py:285`)

If you still want the quick “copy/paste” version: inside the `for item in PHQ8Item.all_items():` loop, after `top_matches` is computed:

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

**Logging behavior / safety**:
- Uses `chunk_preview` only (no full chunk) to reduce accidental data leakage.
- Uses `logger.info(...)` so it will show in `scripts/reproduce_results.py` runs (that script sets log level INFO by default).
- Is **opt-in** via `EMBEDDING_ENABLE_RETRIEVAL_AUDIT=true` (see Spec 32).

Then manually audit a stratified sample to verify if retrieved chunks are actually misaligned.

### Future Work (separate spec required): Relevance Filtering

**Priority**: DEFER (not needed for parity ablation)

**Related specs**:
- `docs/specs/33-retrieval-quality-guardrails.md` (similarity threshold + context budget)
- `docs/specs/34-item-tagged-reference-embeddings.md` (index-time item tags)

This could be a non-circular improvement, but it is **not implementation-ready** here because it requires explicit decisions:
- which keyword source (LLM evidence vs `DOMAIN_KEYWORDS` vs curated list),
- whether to apply as hard filter vs reranking,
- and how to evaluate without inducing bias.

Do not implement this from BUG-031; write a dedicated spec first.

---

## References

- Investigation document: `docs/brainstorming/investigation-zero-shot-beats-few-shot.md`
- Paper notebook (source of truth): `_reference/ai_psychiatrist/quantitative_assessment/embedding_quantitative_analysis.ipynb`
- 2025 RAG research:
  - [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
  - [LangChain CRAG Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/)
  - [RAG Architectures 2025](https://medium.com/data-science-collective/rag-architectures-a-complete-guide-for-2025-daf98a2ede8c)
  - [Google Sufficient Context (ICLR 2025)](https://research.google/blog/deeper-insights-into-retrieval-augmented-generation-the-role-of-sufficient-context/)
  - [Voyage-Context-3](https://blog.voyageai.com/2025/07/23/voyage-context-3/)
