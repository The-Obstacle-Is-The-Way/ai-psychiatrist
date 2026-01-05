# PHQ-8: Patient Health Questionnaire

This document explains the PHQ-8 depression screening tool that AI Psychiatrist uses to assess depression severity.

---

## What is PHQ-8?

The PHQ-8 (Patient Health Questionnaire-8) is a validated self-report depression screening instrument. It consists of 8 questions that assess the frequency of depressive symptoms over the past two weeks.

PHQ-8 is derived from the PHQ-9, which includes a 9th question about suicide ideation. PHQ-8 is often preferred in research settings because:
- It avoids mandatory suicide protocol triggers
- Maintains strong psychometric properties
- Correlates highly with PHQ-9 (as reported in PHQ-8 validation literature)

## Task Validity in Transcript-Based Scoring (Important)

PHQ-8 item scores are defined by **2-week frequency**, but DAIC-WOZ transcripts are not structured as PHQ administration. Many interviews do not contain explicit “days out of 14” frequency statements for each item.

In this repo:
- The quantitative agent returns **`N/A`** when it cannot justify an item score from transcript evidence.
- Coverage (how many items are scored vs `N/A`) is expected to be well below 100%, and must be reported alongside accuracy.

See: [Task Validity](task-validity.md).

---

## The 8 Items

Each item corresponds to a DSM criterion for Major Depressive Episode:

| # | Item | Clinical Domain | Code |
|---|------|-----------------|------|
| 1 | Little interest or pleasure in doing things | Anhedonia | `NO_INTEREST` |
| 2 | Feeling down, depressed, or hopeless | Depressed Mood | `DEPRESSED` |
| 3 | Trouble falling/staying asleep, or sleeping too much | Sleep Disturbance | `SLEEP` |
| 4 | Feeling tired or having little energy | Fatigue | `TIRED` |
| 5 | Poor appetite or overeating | Appetite Changes | `APPETITE` |
| 6 | Feeling bad about yourself — or that you are a failure | Low Self-Esteem | `FAILURE` |
| 7 | Trouble concentrating on things | Concentration Problems | `CONCENTRATING` |
| 8 | Moving or speaking slowly, or being fidgety/restless | Psychomotor Changes | `MOVING` |

---

## Scoring

### Item Scores (0-3)

Each item is scored based on symptom frequency over the past 2 weeks:

| Score | Label | Frequency |
|-------|-------|-----------|
| 0 | Not at all | 0-1 days |
| 1 | Several days | 2-6 days |
| 2 | More than half the days | 7-11 days |
| 3 | Nearly every day | 12-14 days |

### N/A Scores

When the LLM cannot determine a score due to insufficient evidence in the transcript, it returns `N/A`. These items represent **unknown** values, not zeroes.

```python
class ItemAssessment:
    score: int | None  # None = N/A

    @property
    def score_value(self) -> int:
        """Lower-bound value: treats N/A as 0."""
        return self.score if self.score is not None else 0
```

### Total Score Bounds (0-24)

When some items are `N/A`, the true total score is **bounded**, not known exactly:

- **`min_total_score`**: Sum treating N/A as 0 (lower bound)
- **`max_total_score`**: Sum treating N/A as 3 (upper bound, since each item maxes at 3)

```python
@property
def min_total_score(self) -> int:
    """Lower bound total score treating N/A as 0."""
    return sum(item.score_value for item in self.items.values())

@property
def max_total_score(self) -> int:
    """Upper bound total score treating N/A as 3 (max per PHQ-8 item)."""
    return sum(item.score if item.score is not None else 3 for item in self.items.values())
```

The legacy `total_score` property returns `min_total_score` for backward compatibility, but **this is a lower bound when items are missing**.

---

## Severity Levels

Total score maps to depression severity:

| Score Range | Level | Enum Value | MDD? |
|-------------|-------|------------|------|
| 0-4 | Minimal/None | `MINIMAL` | No |
| 5-9 | Mild | `MILD` | No |
| 10-14 | Moderate | `MODERATE` | Yes |
| 15-19 | Moderately Severe | `MOD_SEVERE` | Yes |
| 20-24 | Severe | `SEVERE` | Yes |

**MDD Threshold:** Score ≥ 10 indicates likely Major Depressive Disorder.

### Code Implementation

```python
class SeverityLevel(IntEnum):
    MINIMAL = 0      # 0-4
    MILD = 1         # 5-9
    MODERATE = 2     # 10-14 (MDD threshold)
    MOD_SEVERE = 3   # 15-19
    SEVERE = 4       # 20-24

    @classmethod
    def from_total_score(cls, total: int) -> SeverityLevel:
        if total <= 4:
            return cls.MINIMAL
        if total <= 9:
            return cls.MILD
        if total <= 14:
            return cls.MODERATE
        if total <= 19:
            return cls.MOD_SEVERE
        return cls.SEVERE

    @property
    def is_mdd(self) -> bool:
        return self >= SeverityLevel.MODERATE
```

### Severity Bounds (Partial Assessments)

When items are N/A, severity is **bounded**, not uniquely identified:

```python
@property
def severity_lower_bound(self) -> SeverityLevel:
    """Lower bound severity derived from min_total_score."""
    return SeverityLevel.from_total_score(self.min_total_score)

@property
def severity_upper_bound(self) -> SeverityLevel:
    """Upper bound severity derived from max_total_score."""
    return SeverityLevel.from_total_score(self.max_total_score)

@property
def severity(self) -> SeverityLevel | None:
    """Determinate severity, or None if bounds differ."""
    lower, upper = self.severity_bounds
    if lower == upper:
        return lower
    return None
```

**Key insight:** A single severity label is only meaningful when the assessment is complete OR when missing items cannot change the severity band.

**Example:** If 4 items are scored (total=8) and 4 are N/A:
- `min_total_score = 8` → `severity_lower_bound = MILD`
- `max_total_score = 8 + 12 = 20` → `severity_upper_bound = SEVERE`
- `severity = None` (indeterminate)

---

## How AI Psychiatrist Assesses PHQ-8

### Step 1: Evidence Extraction

The Quantitative Agent first extracts relevant transcript quotes for each item:

```jsonc
{
  "PHQ8_NoInterest": [
    "i used to love hiking but now i can't even get motivated",
    "nothing really seems fun anymore"
  ],
  "PHQ8_Tired": [
    "i have zero energy",
    "some days i can't even get out of bed"
  ],
  "PHQ8_Appetite": []  // No relevant quotes found
}
```

### Step 2: Few-Shot Reference Retrieval

For items with evidence, similar examples from the training set are retrieved:

```
Evidence: "nothing really seems fun anymore"
         │
         ▼ Embedding similarity search
┌────────────────────────────────────────────────┐
│ Reference 1 (score: 2)                         │
│ "i've lost interest in things i used to enjoy" │
├────────────────────────────────────────────────┤
│ Reference 2 (score: 3)                         │
│ "nothing brings me any pleasure at all"        │
└────────────────────────────────────────────────┘
```

### Step 3: Scoring with Reasoning

The LLM predicts scores using evidence and references:

```json
{
  "PHQ8_NoInterest": {
    "evidence": "i used to love hiking but now i can't even get motivated",
    "reason": "Clear loss of interest in previously enjoyed activities, consistent with 'more than half the days' based on frequency cues",
    "score": 2
  },
  "PHQ8_Appetite": {
    "evidence": "No relevant evidence found",
    "reason": "Transcript does not discuss eating habits or appetite changes",
    "score": "N/A"
  }
}
```

---

## Clinical Context

### DSM-5 Criteria for Major Depressive Episode

The PHQ-8 items map to DSM-5 criteria:

| DSM-5 Criterion | PHQ-8 Item |
|-----------------|------------|
| Depressed mood | DEPRESSED |
| Diminished interest/pleasure | NO_INTEREST |
| Weight/appetite change | APPETITE |
| Sleep disturbance | SLEEP |
| Psychomotor agitation/retardation | MOVING |
| Fatigue/loss of energy | TIRED |
| Feelings of worthlessness | FAILURE |
| Concentration difficulties | CONCENTRATING |
| (Suicidal ideation) | Not in PHQ-8 |

### Limitations

**Important:** PHQ-8 is a screening tool, not a diagnostic instrument.

- Scores suggest likelihood of depression, not diagnosis
- Clinical interview required for formal diagnosis
- Self-report nature may underestimate or overestimate symptoms
- Cultural and linguistic factors affect interpretation

---

## Paper Performance Metrics

### Quantitative Agent Accuracy

| Mode | MAE | Improvement |
|------|-----|-------------|
| Zero-shot (Gemma 3) | 0.796 | Baseline |
| Few-shot (Gemma 3) | 0.619 | 22% better |
| Few-shot (MedGemma) | 0.505 | 18% better than Gemma few-shot |

**MAE** = Mean Absolute Error between predicted and actual item scores.

Note: the paper’s MAE is reported on the subset of items where the system produced a score (i.e., excludes N/A). When coverages differ, prefer coverage-aware selective prediction metrics (AURC/AUGRC); see [Statistical methodology (AURC/AUGRC)](../statistics/statistical-methodology-aurc-augrc.md).

### Item-Specific Observations

The paper notes that some items are harder to predict:

- **APPETITE**: Often not discussed in clinical interviews
- **MOVING**: Subtle behavioral observations needed
- **NO_INTEREST/DEPRESSED**: Most frequently discussed and easier to detect

---

## Examples

### Example 1: Moderate Depression

**Transcript excerpt:**
> "I've been feeling really down for the past few weeks. I can't seem to find pleasure in anything anymore. Even my favorite hobbies feel pointless. I'm exhausted all the time but I can't sleep well. Most nights I'm up until 3am."

**Predicted scores:**
- NO_INTEREST: 2 (more than half the days)
- DEPRESSED: 2 (more than half the days)
- SLEEP: 2 (more than half the days)
- TIRED: 2 (more than half the days)
- Others: N/A (not discussed)

**Score bounds:**
- `min_total_score = 8` (N/A → 0)
- `max_total_score = 8 + 12 = 20` (N/A → 3)

**Severity bounds:** MILD → SEVERE (indeterminate; `severity = None`)

**Note:** With 4 N/A items, we cannot determine a single severity label. The system reports bounds instead.

### Example 2: Minimal Symptoms

**Transcript excerpt:**
> "I've been doing pretty well lately. Work is busy but manageable. I still enjoy my weekend activities and I'm sleeping fine."

**Predicted scores:**
- NO_INTEREST: 0 (not at all)
- DEPRESSED: 0 (not at all)
- SLEEP: 0 (not at all)
- Others: N/A or 0

**Total:** 0-2 → **MINIMAL** severity

---

## Code Reference

### Enum Definition (`src/ai_psychiatrist/domain/enums.py`)

```python
class PHQ8Item(StrEnum):
    NO_INTEREST = "NoInterest"
    DEPRESSED = "Depressed"
    SLEEP = "Sleep"
    TIRED = "Tired"
    APPETITE = "Appetite"
    FAILURE = "Failure"
    CONCENTRATING = "Concentrating"
    MOVING = "Moving"
```

### Assessment Entity (`src/ai_psychiatrist/domain/entities.py`)

```python
@dataclass
class PHQ8Assessment:
    items: Mapping[PHQ8Item, ItemAssessment]
    mode: AssessmentMode
    participant_id: int

    @property
    def min_total_score(self) -> int:
        """Lower bound total (N/A → 0)."""
        return sum(item.score_value for item in self.items.values())

    @property
    def max_total_score(self) -> int:
        """Upper bound total (N/A → 3)."""
        return sum(item.score if item.score is not None else 3 for item in self.items.values())

    @property
    def severity_lower_bound(self) -> SeverityLevel:
        return SeverityLevel.from_total_score(self.min_total_score)

    @property
    def severity_upper_bound(self) -> SeverityLevel:
        return SeverityLevel.from_total_score(self.max_total_score)

    @property
    def severity(self) -> SeverityLevel | None:
        """Determinate severity, or None if bounds differ."""
        if self.severity_lower_bound == self.severity_upper_bound:
            return self.severity_lower_bound
        return None
```

---

## See Also

- [Pipeline](../architecture/pipeline.md) - How PHQ-8 scoring fits in the pipeline
- [Glossary](glossary.md) - Clinical terminology
- [DAIC-WOZ Schema](../data/daic-woz-schema.md) - Ground truth data format
