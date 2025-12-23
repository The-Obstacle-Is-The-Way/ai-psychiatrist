# SPEC-003: Make Keyword Backfill Optional + N/A Reason Tracking

**GitHub Issue**: [#49](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/49)
**Status**: Draft
**Author**: Claude Code
**Created**: 2025-12-23

---

## Problem Statement

### 1. Backfill Divergence from Paper

Our implementation includes a **keyword backfill mechanism** (`quantitative.py:228-268`) that always runs after LLM evidence extraction. This causes our results to diverge from the paper:

| Metric | Paper | Us (backfill ON) | Expected (backfill OFF) |
|--------|-------|------------------|-------------------------|
| Coverage | ~50% | 74.1% | ~50% |
| Item MAE | 0.619 | 0.753 | ~0.62 |

**Paper's approach (Section 2.3.2):**
> "If no relevant evidence was found for a given PHQ-8 item, the model produced no output."

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

Following the ChatGPT analysis, we define these **deterministic** N/A categories:

| Reason | Description | Computed From |
|--------|-------------|---------------|
| `NO_MENTION` | No evidence from LLM and no keyword matches | `llm_evidence=[], keyword_hits=[]` |
| `LLM_ONLY_MISSED` | LLM found nothing, keywords found matches (backfill OFF) | `llm_evidence=[], keyword_hits=[...], backfill=false` |
| `KEYWORDS_INSUFFICIENT` | Keywords matched but LLM still returned N/A | Evidence exists but `score=null` |
| `SCORING_REFUSED` | Evidence exists but LLM declined to score | Evidence exists but `score=null` |

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

### Data Structures

#### New Enum: `NAReason`

```python
# domain/enums.py
class NAReason(str, Enum):
    """Reason for N/A (unable to assess) score."""

    NO_MENTION = "no_mention"
    """No evidence from LLM and no keyword matches found."""

    LLM_ONLY_MISSED = "llm_only_missed"
    """LLM found nothing but keywords matched (backfill was OFF)."""

    KEYWORDS_INSUFFICIENT = "keywords_insufficient"
    """Keywords matched but combined evidence still insufficient."""

    SCORING_REFUSED = "scoring_refused"
    """Evidence exists but LLM declined to assign a score."""
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
async def _extract_evidence(self, transcript_text: str) -> EvidenceResult:
    """Extract evidence with tracking."""

    # Step 1: LLM extraction
    llm_evidence = await self._llm_extract(transcript_text)

    # Track counts per domain
    llm_counts = {k: len(v) for k, v in llm_evidence.items()}

    # Step 2: Keyword backfill (if enabled)
    if self._settings.enable_keyword_backfill:
        enriched = self._keyword_backfill(transcript_text, llm_evidence)
        keyword_counts = {k: len(enriched[k]) - llm_counts.get(k, 0) for k in enriched}
    else:
        enriched = llm_evidence
        keyword_counts = {k: 0 for k in llm_evidence}

    return EvidenceResult(
        evidence=enriched,
        llm_counts=llm_counts,
        keyword_counts=keyword_counts,
    )

def _determine_na_reason(
    self,
    llm_count: int,
    keyword_count: int,
    has_final_evidence: bool,
    backfill_enabled: bool,
) -> NAReason:
    """Determine N/A reason from pipeline state."""

    if llm_count == 0 and keyword_count == 0:
        return NAReason.NO_MENTION

    if llm_count == 0 and keyword_count > 0 and not backfill_enabled:
        return NAReason.LLM_ONLY_MISSED

    if has_final_evidence:
        return NAReason.SCORING_REFUSED

    return NAReason.KEYWORDS_INSUFFICIENT
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
      "llm_only_missed": 0,
      "keywords_insufficient": 3,
      "scoring_refused": 0
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

def test_na_reason_scoring_refused():
    """Verify SCORING_REFUSED when evidence exists but score is N/A."""
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

### New Documents

1. **`docs/concepts/backfill-explained.md`**
   - What backfill is and why it exists
   - How it affects coverage vs accuracy tradeoff
   - Decision tree: evidence → score/N/A

2. **`docs/guides/paper-parity-guide.md`**
   - How to run in pure LLM mode
   - Expected results vs paper
   - When to use each mode

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

- [ ] `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL=false` produces ~50% coverage
- [ ] N/A reasons are tracked and exported in results JSON
- [ ] Default behavior unchanged (backfill ON, ~74% coverage)
- [ ] All existing tests pass
- [ ] New tests cover backfill toggle and N/A reason tracking
- [ ] Documentation complete

---

## References

- [GitHub Issue #49](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/49)
- Paper Section 2.3.2: Quantitative Assessment
- Paper Section 3.2: Results (~50% abstention rate)
- `docs/bugs/coverage-investigation.md`: Coverage analysis
- `docs/concepts/extraction-mechanism.md`: How extraction works
