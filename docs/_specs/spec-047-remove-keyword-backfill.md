# Spec 047: Remove Deprecated Keyword Backfill Feature

**Status**: Implemented (2026-01-03)
**Issue**: #82 - Remove deprecated keyword backfill feature
**Author**: Claude
**Date**: 2026-01-02

---

## Overview

The keyword backfill feature is a flawed heuristic that matches keywords like "sleep" or "tired" without semantic understanding. It was retained for historical comparison but should now be completely removed to reduce code complexity and eliminate dead code paths.

**Why Remove Now?**
1. Feature was OFF by default and documented as deprecated
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

**IMPORTANT:** The keyword hit computation (line 166) is triggered by EITHER `enable_keyword_backfill=True` OR `track_na_reasons=True`. This is because `_determine_na_reason()` uses keyword counts to distinguish `NO_MENTION` from `LLM_ONLY_MISSED`.

After removing backfill, we no longer need keyword counts for N/A reason tracking because we're removing the `LLM_ONLY_MISSED` reason entirely.

```python
# DELETE this entire block (lines 163-185)
# Step 2: Find keyword hits (always computed for observability/N/A reasons)
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

**Replace with:**

```python
# Use LLM evidence directly (no backfill)
final_evidence = llm_evidence
```

**Update ItemAssessment construction (lines 233-268):**

```python
# BEFORE (lines 235-238)
evidence_source = self._determine_evidence_source(
    llm_count=llm_counts.get(legacy_key, 0),
    keyword_added_count=keyword_added_counts.get(legacy_key, 0),
)

# AFTER - inline the simplified logic
llm_count = llm_counts.get(legacy_key, 0)
evidence_source = "llm" if llm_count > 0 else None

# BEFORE (lines 240-245)
if score is None and self._settings.track_na_reasons:
    na_reason = self._determine_na_reason(
        llm_count=llm_counts.get(legacy_key, 0),
        keyword_count=keyword_hit_counts.get(legacy_key, 0),
        backfill_enabled=self._settings.enable_keyword_backfill,
    )

# AFTER
if score is None and self._settings.track_na_reasons:
    na_reason = self._determine_na_reason(llm_count)

# BEFORE (line 264)
keyword_evidence_count=keyword_added_counts.get(legacy_key, 0),

# AFTER - remove this line entirely from ItemAssessment constructor
```

**Simplify `_determine_na_reason()` to:**

```python
def _determine_na_reason(self, llm_count: int) -> NAReason:
    """Determine why an item has no score."""
    if llm_count == 0:
        return NAReason.NO_MENTION
    return NAReason.SCORE_NA_WITH_EVIDENCE
```

**Remove `_determine_evidence_source()` method entirely** (lines 509-519) - the simplified logic is inlined above.

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

**Update import in quantitative.py:**

```python
# BEFORE
from ai_psychiatrist.agents.prompts.quantitative import (
    DOMAIN_KEYWORDS,  # DELETE THIS
    QUANTITATIVE_SYSTEM_PROMPT,
    make_evidence_prompt,
    make_scoring_prompt,
)

# AFTER
from ai_psychiatrist.agents.prompts.quantitative import (
    PHQ8_DOMAIN_KEYS,  # USE THIS INSTEAD (already exists)
    QUANTITATIVE_SYSTEM_PROMPT,
    make_evidence_prompt,
    make_scoring_prompt,
)
```

**Update `_extract_evidence()` method (line 378):**

```python
# BEFORE
for key in DOMAIN_KEYWORDS:

# AFTER
for key in PHQ8_DOMAIN_KEYS:
```

**Note:** `DOMAIN_KEYWORDS` is used in two places:
1. `_extract_evidence()` line 378 - only needs keys, use `PHQ8_DOMAIN_KEYS`
2. `_find_keyword_hits()` line 408 - needs keyword lists, but this method is deleted

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
# DELETE assertion: assert settings.keyword_backfill_cap == 3 (line 45)
```

**Update `tests/unit/agents/test_quantitative.py`:**

```python
# UPDATE import (line 27):
# BEFORE
from ai_psychiatrist.agents.prompts.quantitative import (
    DOMAIN_KEYWORDS,
    ...
)
# AFTER
from ai_psychiatrist.agents.prompts.quantitative import (
    PHQ8_DOMAIN_KEYS,
    ...
)

# UPDATE line 374:
# BEFORE
empty_evidence = json.dumps({k: [] for k in DOMAIN_KEYWORDS})
# AFTER
empty_evidence = json.dumps({k: [] for k in PHQ8_DOMAIN_KEYS})

# DELETE entire TestKeywordBackfill class (tests keyword backfill behavior)
# DELETE entire TestDomainKeywords class (tests keyword list + normalization)
```

**Update `tests/unit/agents/test_quantitative_coverage.py`:**

```python
# UPDATE import (line 16):
# BEFORE
from ai_psychiatrist.agents.prompts.quantitative import DOMAIN_KEYWORDS
# AFTER
from ai_psychiatrist.agents.prompts.quantitative import PHQ8_DOMAIN_KEYS

# UPDATE line 25:
# BEFORE
SAMPLE_EVIDENCE_RESPONSE = json.dumps({k: ["evidence"] for k in DOMAIN_KEYWORDS})
# AFTER
SAMPLE_EVIDENCE_RESPONSE = json.dumps({k: ["evidence"] for k in PHQ8_DOMAIN_KEYS})
```

**Update `tests/unit/domain/test_value_objects.py`:**

- Remove `keyword_evidence_count` construction + assertions.
- Narrow `evidence_source` expectations to `"llm"` or `None` only.

**Update `tests/unit/scripts/test_evaluate_selective_prediction_confidence.py`:**

- Remove `keyword_evidence_count` from synthetic `item_signals` fixtures.

**Update `tests/integration/test_selective_prediction_from_output.py`:**

- Remove `keyword_evidence_count` from synthetic `item_signals` fixtures.

**Add regression tests (new):**

- `tests/unit/test_spec_047_remove_keyword_backfill.py`
  - Asserts keyword backfill config + schema are removed (field-level).
  - Asserts run output filename no longer includes `backfill-*`.

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

**Remove backfill parameter (line 826):**

```python
# DELETE
backfill=settings.quantitative.enable_keyword_backfill,
```

**Remove keyword_evidence_count from output serialization (line 305):**

```python
# DELETE
"keyword_evidence_count": item_assessment.keyword_evidence_count,
```

#### 2.4 Selective Prediction Evaluation Script (`scripts/evaluate_selective_prediction.py`)

**Update confidence calculation (line 237):**

```python
# BEFORE
sig.get("keyword_evidence_count", 0)

# AFTER - remove this term from the calculation entirely
# The total_evidence calculation becomes: llm_evidence_count only
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
| `docs/architecture/pipeline.md` | Remove backfill mention in quantitative stage |
| `docs/results/run-output-schema.md` | Remove `keyword_evidence_count` from schema |
| `docs/results/reproduction-results.md` | Update “Results saved to” filename format |
| `docs/statistics/metrics-and-evaluation.md` | Simplify confidence formula (remove `+ keyword_evidence_count`) |
| `docs/statistics/statistical-methodology-aurc-augrc.md` | Remove `keyword_evidence_count` reference |
| `docs/statistics/coverage.md` | Remove backfill ablation mention (feature removed) |
| `docs/research/augrc-improvement-techniques-2026.md` | Remove `keyword_evidence_count` from code sample |
| `docs/_specs/spec-046-selective-prediction-confidence-signals.md` | Simplify `total_evidence` formula |
| `docs/data/artifact-namespace-registry.md` | Update outputs filename pattern (no `backfill-*`) |

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

1. **No action needed** for baseline runs (keyword backfill was removed; no runtime toggle remains).
2. After update, remove these from `.env` if present:
   - `QUANTITATIVE_ENABLE_KEYWORD_BACKFILL`
   - `QUANTITATIVE_KEYWORD_BACKFILL_CAP`
3. If parsing output JSON, update schemas to remove `keyword_evidence_count`

### For Downstream Analysis Scripts

Scripts that read `keyword_evidence_count` or check for `evidence_source == "keyword"` will need updates.

**Note on historical artifacts**: existing run outputs may still include `backfill-off` in filenames.
This is a historical naming convention; new runs after this spec should use the updated naming scheme.

---

## Implementation Order

Execute in this order to maintain test coverage at each step:

### Step 1: Remove enum values and narrow types (domain layer)

```bash
# 1a. Update NAReason enum (domain/enums.py)
# Remove LLM_ONLY_MISSED, KEYWORDS_INSUFFICIENT

# 1b. Update ItemAssessment (domain/value_objects.py)
# Remove keyword_evidence_count field
# Change evidence_source type: Literal["llm", "keyword", "both"] → Literal["llm"]
```

### Step 2: Update configuration (config layer)

```bash
# 2a. Update QuantitativeSettings (config.py)
# Remove enable_keyword_backfill, keyword_backfill_cap fields

# 2b. Update conftest.py
# Remove QUANTITATIVE_ENABLE_KEYWORD_BACKFILL, QUANTITATIVE_KEYWORD_BACKFILL_CAP from cleanup
```

### Step 3: Delete keyword infrastructure (resources + prompts)

```bash
# 3a. Delete phq8_keywords.yaml

# 3b. Update prompts/quantitative.py
# Remove DOMAIN_KEYWORDS, _load_domain_keywords(), _KEYWORDS_RESOURCE_PATH
```

### Step 4: Update agent implementation (agents layer)

```bash
# 4a. Update quantitative.py imports
# Change DOMAIN_KEYWORDS → PHQ8_DOMAIN_KEYS

# 4b. Delete _find_keyword_hits() and _merge_evidence() methods

# 4c. Simplify assess() method
# Remove keyword hit computation, backfill logic, keyword counts
# Inline evidence_source logic, simplify _determine_na_reason() call

# 4d. Simplify _determine_na_reason() signature
# Remove keyword_count and backfill_enabled parameters

# 4e. Delete _determine_evidence_source() method
```

### Step 5: Delete and update tests

```bash
# 5a. Delete test_quantitative_backfill.py entirely

# 5b. Update test_config.py
# Remove backfill-related tests

# 5c. Update test_quantitative.py
# Change DOMAIN_KEYWORDS → PHQ8_DOMAIN_KEYS
# Delete TestDomainKeywords class

# 5d. Update test_quantitative_coverage.py
# Change DOMAIN_KEYWORDS → PHQ8_DOMAIN_KEYS
```

### Step 6: Update services and scripts

```bash
# 6a. Update experiment_tracking.py
# Remove enable_keyword_backfill field

# 6b. Update reproduce_results.py
# Remove backfill parameter
# Remove keyword_evidence_count from output serialization

# 6c. Update evaluate_selective_prediction.py
# Remove keyword_evidence_count from total_evidence calculation
```

### Step 7: Update documentation

```bash
# 7a. Update .env.example
# 7b. Update docs/configs/configuration.md
# 7c. Update docs/configs/configuration-philosophy.md
# 7d. Update preflight checklists
# 7e. Update docs/results/run-output-schema.md
# 7f. Update docs/statistics/metrics-and-evaluation.md
# 7g. Update docs/statistics/statistical-methodology-aurc-augrc.md
# 7h. Update docs/research/augrc-improvement-techniques-2026.md
# 7i. Update docs/_specs/spec-046-selective-prediction-confidence-signals.md
```

### Step 8: Verify

```bash
uv run pytest -v
rg "keyword.?backfill|BACKFILL" --type py --type yaml
```

---

## Verification

After implementation:

```bash
# 1. Run tests
uv run pytest -v

# 2. Check for remaining references
rg "keyword.?backfill|BACKFILL" --type py --type yaml

# 3. Verify .env.example has no backfill settings (expect no output)
rg -n "backfill" .env.example || true

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

## Code Metrics

**Files Deleted:**
- `src/ai_psychiatrist/resources/phq8_keywords.yaml` (~439 lines)
- `tests/unit/agents/test_quantitative_backfill.py` (~276 lines)

**Lines Removed (approximate):**
- Config fields: ~15 lines
- Quantitative agent methods: ~75 lines
- Prompt loading: ~35 lines
- Test updates: ~30 lines
- Documentation: ~100 lines across multiple files

**Total:** ~970 lines removed (net reduction in codebase)

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
