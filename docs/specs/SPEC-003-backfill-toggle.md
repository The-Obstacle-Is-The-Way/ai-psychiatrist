# SPEC-003: Make Keyword Backfill Optional + N/A Reason Tracking

**GitHub Issue**: [#49](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/49)
**Status**: Draft
**Author**: Claude Code
**Created**: 2025-12-23

---

## Problem Statement

### 1. Backfill Divergence from Paper

Our implementation includes a **keyword backfill mechanism** (`quantitative.py:228-268`) that always runs after LLM evidence extraction. This causes our results to diverge from the paper:

| Metric | Paper | Us (backfill ON) | Hypothesis (backfill OFF) |
|--------|-------|------------------|-------------------------|
| Coverage | ~50% | 74.1% | ~50% |
| Item MAE | 0.619 | 0.753 | ~0.62 |

*Note*: The “backfill OFF” column is a **paper-parity hypothesis** and must be validated by an
ablation run after this spec is implemented.

### What the Paper Says (and Doesn’t Say)

The paper describes an evidence-first scoring approach, and states that items without
relevant evidence produce no output:

> "If no relevant evidence was found for a given PHQ-8 item, the model produced no output."

However, the paper does **not** explicitly describe any rule-based fallback (keyword
matching) step. For **paper parity** experiments, the safest interpretation is to
evaluate *pure LLM extraction + scoring*, without heuristic evidence injection.

**Our approach:**
When the LLM misses evidence, we scan the transcript for keyword matches and append them to the evidence dict. This increases coverage but measures something fundamentally different: "LLM + rule-based heuristics" vs "pure LLM capability."

### 2. No N/A Reason Tracking

Currently, when an item returns N/A, we don't know **why**:
- Was the symptom never discussed?
- Did the LLM miss it but keywords found it (and still returned N/A)?
- Was evidence found but deemed insufficient?

This makes debugging and paper comparison difficult.

---

## Goals

1. **Paper parity**: Allow running without backfill to reproduce paper's methodology
2. **Observability**: Track why items return N/A for debugging and analysis
3. **Backwards compatibility**: Default behavior unchanged (backfill ON)

---

## Design

### N/A Reason Taxonomy

We define **deterministic** N/A categories based on observable pipeline state (LLM
evidence extraction output, keyword hits, and whether backfill is enabled).

| Reason | Description | Computed From |
|--------|-------------|---------------|
| `NO_MENTION` | No evidence from LLM and no keyword matches found | `llm_count=0, keyword_hits=0` |
| `LLM_ONLY_MISSED` | LLM found no evidence, but keyword hits exist (backfill was OFF) | `llm_count=0, keyword_hits>0, backfill=false` |
| `SCORE_NA_WITH_EVIDENCE` | Evidence was provided to the scorer, but the LLM still returned N/A | `final_evidence_count>0, score=null` |

Note: `LLM_ONLY_MISSED` only appears when backfill is OFF. When ON, keywords augment evidence so the LLM sees them.

### Configuration

Add to `config.py` under a new `QuantitativeSettings` class:

```python
class QuantitativeSettings(BaseSettings):
    """Quantitative assessment configuration."""

    model_config = SettingsConfigDict(
        env_prefix="QUANTITATIVE_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )

    enable_keyword_backfill: bool = Field(
        default=True,
        description=(
            "Enable rule-based keyword backfill for evidence extraction. "
            "When True (default), sentences matching symptom keywords are added "
            "when the LLM misses evidence. Set False for paper-parity evaluation "
            "(pure LLM capability measurement). Paper Section 2.3.2."
        ),
    )

    track_na_reasons: bool = Field(
        default=True,
        description="Track why items return N/A (for debugging/analysis).",
    )

    keyword_backfill_cap: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum keyword-matched sentences per domain.",
    )
```

Environment variables:
- `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false` → paper parity mode
- `QUANTITATIVE_TRACK_NA_REASONS=true` → enable reason tracking

Add the group to the root `Settings` model:

```python
# config.py
class Settings(BaseSettings):
    ...
    quantitative: QuantitativeSettings = Field(default_factory=QuantitativeSettings)
```

### Wiring / Dependency Injection

`QuantitativeAssessmentAgent` currently has no dependency on a quantitative-specific settings group.
To make this spec implementable, add a `quantitative_settings` parameter and wire it through
the API layer and scripts:

1. **Agent constructor** (`src/ai_psychiatrist/agents/quantitative.py`)
   - Add `quantitative_settings: QuantitativeSettings | None = None`
   - Store `self._settings = quantitative_settings or QuantitativeSettings()`
   - Use `self._settings.enable_keyword_backfill` and `self._settings.keyword_backfill_cap`

2. **API server** (`server.py`)
   - Load `Settings` once at startup as already done
   - Pass `settings.quantitative` into `QuantitativeAssessmentAgent(...)`

3. **Reproduction scripts** (`scripts/reproduce_results.py`, etc.)
   - Prefer `get_settings().quantitative` and pass it into the agent constructor
   - This avoids silent divergence between “script default” and “server default”

### Data Structures

#### New Enum: `NAReason`

```python
# domain/enums.py
from enum import StrEnum

class NAReason(StrEnum):
    """Reason for N/A (unable to assess) score."""

    NO_MENTION = "no_mention"
    """No evidence from LLM and no keyword matches found."""

    LLM_ONLY_MISSED = "llm_only_missed"
    """LLM found nothing but keywords matched (backfill was OFF)."""

    SCORE_NA_WITH_EVIDENCE = "score_na_with_evidence"
    """Evidence exists, but the scorer returned N/A (abstained)."""
```

#### Updated `ItemAssessment`

```python
# domain/value_objects.py
@dataclass(frozen=True, slots=True)
class ItemAssessment:
    item: PHQ8Item
    evidence: str
    reason: str
    score: int | None

    # New fields
    na_reason: NAReason | None = None
    """Reason for N/A (only set when score is None)."""

    evidence_source: Literal["llm", "keyword", "mixed"] | None = None
    """Source of evidence (for observability)."""

    llm_evidence_count: int = 0
    """Number of evidence items from LLM extraction."""

    keyword_evidence_count: int = 0
    """Number of evidence items from keyword backfill."""
```

### Implementation Changes

#### `quantitative.py`

```python
@dataclass(frozen=True, slots=True)
class EvidenceResult:
    """Evidence extraction output with observability metadata."""

    evidence: dict[str, list[str]]
    llm_counts: dict[str, int]
    keyword_hit_counts: dict[str, int]
    keyword_added_counts: dict[str, int]

async def _extract_evidence(self, transcript_text: str) -> EvidenceResult:
    """Extract evidence and compute LLM/keyword counts."""

    # Step 1: LLM extraction
    # NOTE: In the current codebase, LLM extraction happens inside _extract_evidence via
    # make_evidence_prompt() + llm_client.simple_chat(). You may keep that inline, or
    # extract it into a helper such as _extract_evidence_llm() for readability/testing.
    llm_evidence = await self._extract_evidence_llm(transcript_text)

    # Track counts per domain
    llm_counts = {k: len(v) for k, v in llm_evidence.items()}

    # Step 2: Keyword hits (always computed for observability)
    # NOTE: _find_keyword_hits() is a NEW helper to implement. It should return up to `cap`
    # matched sentences per PHQ-8 key, using the same sentence splitting and substring matching
    # semantics as _keyword_backfill().
    keyword_hits = self._find_keyword_hits(transcript_text, cap=self._settings.keyword_backfill_cap)
    keyword_hit_counts = {k: len(v) for k, v in keyword_hits.items()}

    # Step 3: Apply keyword backfill (optional)
    if self._settings.enable_keyword_backfill:
        enriched = self._keyword_backfill(transcript_text, llm_evidence, cap=self._settings.keyword_backfill_cap)
        keyword_added_counts = {k: max(0, len(enriched.get(k, [])) - llm_counts.get(k, 0)) for k in enriched}
    else:
        enriched = llm_evidence
        keyword_added_counts = {k: 0 for k in llm_evidence}

    return EvidenceResult(
        evidence=enriched,
        llm_counts=llm_counts,
        keyword_hit_counts=keyword_hit_counts,
        keyword_added_counts=keyword_added_counts,
    )

def _determine_na_reason(
    self,
    llm_count: int,
    keyword_hit_count: int,
    final_evidence_count: int,
    backfill_enabled: bool,
) -> NAReason:
    """Determine N/A reason from pipeline state."""

    if llm_count == 0 and keyword_hit_count == 0:
        return NAReason.NO_MENTION

    if llm_count == 0 and keyword_hit_count > 0 and not backfill_enabled:
        return NAReason.LLM_ONLY_MISSED

    # Evidence existed (LLM and/or keyword-added) but scorer returned N/A.
    if final_evidence_count > 0:
        return NAReason.SCORE_NA_WITH_EVIDENCE

    return NAReason.NO_MENTION
```

### Output Format Changes

#### Per-Item Results (JSON)

```json
{
  "predicted_items": {
    "NoInterest": {
      "score": null,
      "na_reason": "no_mention",
      "evidence_source": null,
      "llm_evidence_count": 0,
      "keyword_evidence_count": 0
    },
    "Depressed": {
      "score": 2,
      "na_reason": null,
      "evidence_source": "mixed",
      "llm_evidence_count": 1,
      "keyword_evidence_count": 2
    }
  }
}
```

#### Aggregate Statistics

```json
{
  "na_reason_breakdown": {
    "NoInterest": {
      "no_mention": 12,
      "llm_only_missed": 3,
      "score_na_with_evidence": 0
    }
  },
  "backfill_impact": {
    "items_rescued_by_backfill": 45,
    "total_keyword_evidence_added": 123
  }
}
```

---

## Migration Path

### Phase 1: Config + Tracking (This PR)
1. Add `QuantitativeSettings` to `config.py`
2. Add `NAReason` enum
3. Update `ItemAssessment` value object
4. Modify `_extract_evidence()` to track sources
5. Modify `_validate_and_normalize()` to set `na_reason`
6. Update reproduction script to output new fields

### Phase 2: Documentation
1. Create `docs/concepts/backfill-explained.md`
2. Create `docs/guides/paper-parity-guide.md`
3. Update `docs/reference/configuration.md`
4. Update `docs/bugs/coverage-investigation.md`

### Phase 3: Validation
1. Run ablation: backfill ON vs OFF on paper test split
2. Compare coverage/MAE deltas
3. Document findings in reproduction notes

---

## Testing

### Unit Tests

```python
# tests/unit/agents/test_quantitative.py

def test_backfill_disabled_no_enrichment():
    """Verify backfill OFF doesn't add keyword evidence."""

def test_backfill_enabled_adds_evidence():
    """Verify backfill ON adds keyword matches."""

def test_na_reason_no_mention():
    """Verify NO_MENTION when LLM and keywords find nothing."""

def test_na_reason_llm_only_missed():
    """Verify LLM_ONLY_MISSED when backfill OFF but keywords match."""

def test_na_reason_score_na_with_evidence():
    """Verify SCORE_NA_WITH_EVIDENCE when evidence exists but score is N/A."""
```

### Integration Tests

```python
# tests/integration/test_quantitative_backfill.py

async def test_paper_parity_mode():
    """Verify backfill=false produces lower coverage."""

async def test_na_reason_tracking():
    """Verify N/A reasons are populated correctly."""
```

---

## Documentation Updates

### Documents (Already Present)

These documents already exist, but must remain aligned with the implementation:

1. **`docs/concepts/backfill-explained.md`** (mechanism + tradeoffs)
2. **`docs/guides/paper-parity-guide.md`** (paper-parity workflow; blocked until implemented)

### Updated Documents

1. **`docs/reference/configuration.md`**
   - Add `QuantitativeSettings` section
   - Document new environment variables

2. **`docs/bugs/coverage-investigation.md`**
   - Reference this spec as the resolution
   - Update status to RESOLVED

3. **`docs/results/reproduction-notes.md`**
   - Add ablation results (after running)

---

## Acceptance Criteria

- [ ] Backfill can be disabled via `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false`
- [ ] With backfill disabled, no keyword evidence is injected into the scorer prompt
- [ ] N/A reasons are deterministically assigned when `QUANTITATIVE_TRACK_NA_REASONS=true`
- [ ] Default behavior unchanged (backfill enabled, reason tracking enabled)
- [ ] All existing tests pass
- [ ] New tests cover backfill toggle and N/A reason tracking
- [ ] Documentation updated to reflect “paper does not describe backfill” (no unsupported claims)

---

## References

- [GitHub Issue #49](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/49)
- Paper Section 2.3.2: Quantitative Assessment
- Paper Section 3.2: Results (~50% abstention rate)
- `docs/bugs/coverage-investigation.md`: Coverage analysis
- `docs/concepts/extraction-mechanism.md`: How extraction works
