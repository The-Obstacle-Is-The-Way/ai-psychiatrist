# Spec 047: Remove Deprecated Keyword Backfill Feature

**Status**: DRAFT
**Issue**: #82 - Remove deprecated keyword backfill feature
**Author**: Claude
**Date**: 2026-01-02

---

## Overview

The keyword backfill feature is a flawed heuristic that matches keywords like "sleep" or "tired" without semantic understanding. It was retained for historical comparison but should now be completely removed to reduce code complexity and eliminate dead code paths.

**Why Remove Now?**
1. Default is OFF (`enable_keyword_backfill=false`) and documented as deprecated
2. Enabling it harms validity without improving clinical outcomes
3. It adds ~200 lines of code that are never executed in production
4. It creates confusion for new contributors

---

## Components to Remove

### Phase 1: Core Implementation (Breaking Changes)

#### 1.1 Configuration (`src/ai_psychiatrist/config.py`)

**Remove these fields from `QuantitativeSettings`:**

```python
# DELETE lines 383-396
enable_keyword_backfill: bool = Field(
    default=False,
    description="DEPRECATED: Do NOT enable. Flawed heuristic retained for ablation only.",
)
track_na_reasons: bool = Field(  # KEEP THIS ONE
    default=True,
    description="Track why items return N/A",
)
keyword_backfill_cap: int = Field(  # DELETE
    default=3,
    ge=1,
    le=10,
    description="DEPRECATED: Irrelevant since backfill should remain OFF.",
)
```

**Update docstring:**

```python
class QuantitativeSettings(BaseSettings):
    """Quantitative assessment configuration."""

    model_config = SettingsConfigDict(
        env_prefix="QUANTITATIVE_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )

    track_na_reasons: bool = Field(
        default=True,
        description="Track why items return N/A (for diagnostics)",
    )
```

#### 1.2 Quantitative Agent (`src/ai_psychiatrist/agents/quantitative.py`)

**Remove these methods entirely:**

| Method | Lines | Purpose |
|--------|-------|---------|
| `_find_keyword_hits()` | 388-418 | Find keyword matches in transcript |
| `_merge_evidence()` | 420-453 | Merge LLM + keyword evidence |

**Simplify these methods:**

| Method | Lines | Change |
|--------|-------|--------|
| `_determine_na_reason()` | 494-507 | Remove `LLM_ONLY_MISSED` and `KEYWORDS_INSUFFICIENT` cases |
| `_determine_evidence_source()` | 509-519 | Remove, always return `"llm"` or `None` |

**Remove from `assess()` method (lines 163-185):**

```python
# DELETE this entire block
keyword_hits: dict[str, list[str]] = {}
keyword_hit_counts: dict[str, int] = {}
if self._settings.enable_keyword_backfill or self._settings.track_na_reasons:
    keyword_hits = self._find_keyword_hits(
        transcript.text,
        cap=self._settings.keyword_backfill_cap,
    )
    keyword_hit_counts = {k: len(v) for k, v in keyword_hits.items()}

# Step 3: Conditional backfill
if self._settings.enable_keyword_backfill:
    # Merge LLM evidence with keyword hits
    final_evidence = self._merge_evidence(
        llm_evidence, keyword_hits, cap=self._settings.keyword_backfill_cap
    )
else:
    final_evidence = llm_evidence

# Calculate added evidence from backfill
keyword_added_counts = {
    k: len(final_evidence.get(k, [])) - llm_counts.get(k, 0) for k in final_evidence
}
```

**Simplify `_determine_na_reason()` to:**

```python
def _determine_na_reason(self, llm_count: int) -> NAReason:
    """Determine why an item has no score."""
    if llm_count == 0:
        return NAReason.NO_MENTION
    return NAReason.SCORE_NA_WITH_EVIDENCE
```

#### 1.3 NAReason Enum (`src/ai_psychiatrist/domain/enums.py`)

**Remove these enum values:**

```python
# DELETE
LLM_ONLY_MISSED = "llm_only_missed"
"""LLM missed evidence that keywords would have found (backfill disabled)."""

KEYWORDS_INSUFFICIENT = "keywords_insufficient"
"""Keywords matched but still insufficient for scoring."""
```

**Keep:**
- `NO_MENTION` - Neither LLM nor transcript mentions the symptom
- `SCORE_NA_WITH_EVIDENCE` - Evidence exists but scorer abstained

#### 1.4 Value Objects (`src/ai_psychiatrist/domain/value_objects.py`)

**Remove field:**

```python
# DELETE line 166-167
keyword_evidence_count: int = 0
"""Number of evidence items added from keyword hits (injected into scorer evidence)."""
```

**Update `evidence_source` type (line 160):**

```python
# BEFORE
evidence_source: Literal["llm", "keyword", "both"] | None = None

# AFTER
evidence_source: Literal["llm"] | None = None
```

**Update docstring:**

```python
evidence_source: Literal["llm"] | None = None
"""Source of evidence provided to the scorer (None means no evidence found)."""
```

#### 1.5 Prompts (`src/ai_psychiatrist/agents/prompts/quantitative.py`)

**Remove:**

```python
# DELETE lines 20, 35-72
_KEYWORDS_RESOURCE_PATH = "resources/phq8_keywords.yaml"

@lru_cache(maxsize=1)
def _load_domain_keywords() -> dict[str, list[str]]:
    """..."""
    # ... entire function

DOMAIN_KEYWORDS: dict[str, list[str]] = _load_domain_keywords()
```

**Keep the import removal in quantitative.py:**

```python
# DELETE from import statement
from ai_psychiatrist.agents.prompts.quantitative import (
    DOMAIN_KEYWORDS,  # DELETE THIS
    QUANTITATIVE_SYSTEM_PROMPT,
    make_evidence_prompt,
    make_scoring_prompt,
)
```

#### 1.6 Resources

**Delete file:**
- `src/ai_psychiatrist/resources/phq8_keywords.yaml`

---

### Phase 2: Supporting Code

#### 2.1 Tests

**Delete entirely:**
- `tests/unit/agents/test_quantitative_backfill.py`

**Update `tests/conftest.py` (remove from cleanup list):**

```python
# DELETE from _ENV_VARS_TO_CLEAR
"QUANTITATIVE_ENABLE_KEYWORD_BACKFILL",
"QUANTITATIVE_KEYWORD_BACKFILL_CAP",
```

**Update `tests/unit/test_config.py`:**

```python
# DELETE TestQuantitativeSettings.test_env_override_enable_backfill (lines 47-51)
# UPDATE TestQuantitativeSettings.test_defaults to only check track_na_reasons
```

#### 2.2 Experiment Tracking (`src/ai_psychiatrist/services/experiment_tracking.py`)

**Remove field from Run dataclass:**

```python
# DELETE line 141
enable_keyword_backfill: bool
```

**Update `from_settings()` method:**

```python
# DELETE line 169
enable_keyword_backfill=settings.quantitative.enable_keyword_backfill,
```

#### 2.3 Reproduce Results Script (`scripts/reproduce_results.py`)

**Remove backfill parameter:**

```python
# DELETE line 826
backfill=settings.quantitative.enable_keyword_backfill,
```

---

### Phase 3: Documentation

#### 3.1 Active Documentation Updates

| File | Change |
|------|--------|
| `.env.example` | Remove `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL` and `QUANTITATIVE_KEYWORD_BACKFILL_CAP` |
| `docs/configs/configuration.md` | Remove backfill settings from table |
| `docs/configs/configuration-philosophy.md` | Remove backfill from "OFF permanently" section |
| `docs/preflight-checklist/preflight-checklist-zero-shot.md` | Remove backfill checks |
| `docs/preflight-checklist/preflight-checklist-few-shot.md` | Remove backfill checks |

#### 3.2 Archive Documentation (Leave As-Is)

Files in `docs/_archive/` should remain untouched as historical record:
- `docs/_archive/specs/SPEC-003-backfill-toggle.md`
- `docs/_archive/concepts/backfill-explained.md`
- etc.

---

## Output Schema Changes

### Before (with backfill fields)

```json
{
  "items": {
    "NoInterest": {
      "score": 2,
      "evidence": "...",
      "reason": "...",
      "na_reason": null,
      "evidence_source": "llm",
      "llm_evidence_count": 2,
      "keyword_evidence_count": 0,  // REMOVED
      "retrieval_reference_count": 2,
      "retrieval_similarity_mean": 0.72,
      "retrieval_similarity_max": 0.85
    }
  }
}
```

### After

```json
{
  "items": {
    "NoInterest": {
      "score": 2,
      "evidence": "...",
      "reason": "...",
      "na_reason": null,
      "evidence_source": "llm",
      "llm_evidence_count": 2,
      "retrieval_reference_count": 2,
      "retrieval_similarity_mean": 0.72,
      "retrieval_similarity_max": 0.85
    }
  }
}
```

### Breaking Changes

| Field | Change |
|-------|--------|
| `keyword_evidence_count` | Removed |
| `evidence_source` | Type narrowed: `"keyword"` and `"both"` values no longer possible |
| `na_reason` | Values `"llm_only_missed"` and `"keywords_insufficient"` no longer possible |

---

## Migration

### For Users

1. **No action needed** if using default settings (`enable_keyword_backfill=false`)
2. After update, remove these from `.env` if present:
   - `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL`
   - `QUANTITATIVE_KEYWORD_BACKFILL_CAP`
3. If parsing output JSON, update schemas to remove `keyword_evidence_count`

### For Downstream Analysis Scripts

Scripts that read `keyword_evidence_count` or check for `evidence_source == "keyword"` will need updates.

---

## Implementation Order

Execute in this order to maintain test coverage at each step:

1. **Update NAReason enum** (remove unused values)
2. **Update value_objects.py** (remove `keyword_evidence_count`, narrow `evidence_source`)
3. **Update config.py** (remove settings)
4. **Update conftest.py** (remove env vars from cleanup)
5. **Delete test_quantitative_backfill.py**
6. **Update test_config.py** (remove backfill tests)
7. **Update quantitative.py** (remove methods, simplify logic)
8. **Update prompts/quantitative.py** (remove DOMAIN_KEYWORDS)
9. **Delete phq8_keywords.yaml**
10. **Update experiment_tracking.py** (remove field)
11. **Update reproduce_results.py** (remove parameter)
12. **Update documentation**

---

## Verification

After implementation:

```bash
# 1. Run tests
uv run pytest -v

# 2. Check for remaining references
rg "keyword.?backfill|BACKFILL" --type py --type yaml

# 3. Verify .env.example has no backfill settings
grep -i backfill .env.example

# 4. Run a quick assessment to verify nothing breaks
uv run python -c "
from ai_psychiatrist.agents.quantitative import QuantitativeAssessmentAgent
from ai_psychiatrist.domain.entities import Transcript
print('Import successful')
"
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking downstream scripts | Low | Medium | Document output schema changes |
| Historical run comparison | Low | Low | Archive files preserved |
| Test coverage drop | Low | Low | ~50 lines of test code removed, but it tested dead code |

---

## Estimated Effort

- **Code changes**: ~2 hours
- **Testing**: ~1 hour
- **Documentation**: ~30 minutes
- **Total**: ~3.5 hours

---

## Approval Checklist

- [ ] Spec reviewed by maintainer
- [ ] Output schema changes acceptable
- [ ] No active users depend on backfill feature
- [ ] Archive documentation preserved
