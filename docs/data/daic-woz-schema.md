# DAIC-WOZ Dataset Schema

**Purpose**: Enable development without direct data access
**Dataset**: Distress Analysis Interview Corpus - Wizard of Oz (DAIC-WOZ)
**Access**: Requires EULA agreement from USC ICT
**Reference**: [AVEC 2017 Challenge](https://dl.acm.org/doi/10.1145/3133944.3133953)

---

## Overview

DAIC-WOZ is a clinical interview dataset for depression detection research. It contains semi-structured interviews conducted by an animated virtual interviewer named Ellie with participants who may or may not have depression.

### Key Statistics

| Metric | Value |
|--------|-------|
| Total participants | 189 |
| Labeled participants | 142 (train + dev) |
| Unlabeled participants | 47 (test) |
| ID range | 300-492 (with gaps) |
| Interview duration | 5-25 minutes |
| Total size | ~86 GB (with all modalities) |

### Missing Participant IDs

Not all IDs in range 300-492 exist. Known gaps include:
```
342, 394, 398, 460, ...
```

Always validate participant existence before processing.

---

## Directory Structure

### Expected Layout (after `scripts/prepare_dataset.py`)

```
data/
├── transcripts/                         # Extracted transcripts
│   ├── 300_P/
│   │   └── 300_TRANSCRIPT.csv
│   ├── 301_P/
│   │   └── 301_TRANSCRIPT.csv
│   └── .../
├── embeddings/                          # Pre-computed (Spec 08)
│   ├── reference_embeddings.npz
│   └── reference_embeddings.json
├── train_split_Depression_AVEC2017.csv  # Ground truth (train)
├── dev_split_Depression_AVEC2017.csv    # Ground truth (dev)
├── test_split_Depression_AVEC2017.csv   # Identifiers only
└── full_test_split.csv                  # Test totals (if available)
```

### Configuration Paths

Defined in `src/ai_psychiatrist/config.py`:

```python
class DataSettings(BaseSettings):
    base_dir: Path = Path("data")
    transcripts_dir: Path = Path("data/transcripts")
    embeddings_path: Path = Path("data/embeddings/reference_embeddings.npz")
    train_csv: Path = Path("data/train_split_Depression_AVEC2017.csv")
    dev_csv: Path = Path("data/dev_split_Depression_AVEC2017.csv")
```

---

## Transcript Format

### File Location

```
data/transcripts/{id}_P/{id}_TRANSCRIPT.csv
```

Example: `data/transcripts/300_P/300_TRANSCRIPT.csv`

### Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `start_time` | float | Utterance start (seconds) | `36.588` |
| `stop_time` | float | Utterance end (seconds) | `39.668` |
| `speaker` | string | Speaker identifier | `"Ellie"` or `"Participant"` |
| `value` | string | Transcript text | `"hi i'm ellie thanks for coming in today"` |

### Format Details

- **Separator**: Tab (`\t`)
- **Encoding**: UTF-8
- **Text style**: Lowercase, minimal punctuation
- **Headers**: First row is header
- **Typical size**: ~100-300 rows per transcript

### Synthetic Example

```tsv
start_time	stop_time	speaker	value
0.000	2.500	Ellie	hi i'm ellie thanks for coming in today
3.100	4.200	Participant	hello
5.000	8.500	Ellie	how are you doing today
9.200	12.800	Participant	i'm doing okay i guess
13.500	18.000	Ellie	tell me about the last time you felt really happy
19.200	28.500	Participant	um i don't know it's been a while i guess maybe when i saw my family last month
30.000	35.500	Ellie	that sounds nice can you tell me more about that visit
```

### How It's Loaded

`TranscriptService._parse_daic_woz_transcript()` in `services/transcript.py`:

```python
df = pd.read_csv(path, sep="\t")
df = df.dropna(subset=["speaker", "value"])
df["dialogue"] = df["speaker"] + ": " + df["value"]
return "\n".join(df["dialogue"].tolist())
```

**Output format** (what agents see):
```
Ellie: hi i'm ellie thanks for coming in today
Participant: hello
Ellie: how are you doing today
Participant: i'm doing okay i guess
...
```

---

## Ground Truth Format

### Train/Dev Split CSVs

**Files**:
- `train_split_Depression_AVEC2017.csv` (107 participants)
- `dev_split_Depression_AVEC2017.csv` (35 participants)

### Schema

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `Participant_ID` | int | 300-492 | Unique identifier |
| `PHQ8_Binary` | int | 0-1 | MDD indicator (1 if score >= 10) |
| `PHQ8_Score` | int | 0-24 | Total PHQ-8 score |
| `Gender` | int | 0-1 | 0 = male, 1 = female |
| `PHQ8_NoInterest` | int | 0-3 | Item 1: Little interest or pleasure |
| `PHQ8_Depressed` | int | 0-3 | Item 2: Feeling down, depressed |
| `PHQ8_Sleep` | int | 0-3 | Item 3: Sleep problems |
| `PHQ8_Tired` | int | 0-3 | Item 4: Feeling tired |
| `PHQ8_Appetite` | int | 0-3 | Item 5: Appetite changes |
| `PHQ8_Failure` | int | 0-3 | Item 6: Feeling bad about self |
| `PHQ8_Concentrating` | int | 0-3 | Item 7: Trouble concentrating |
| `PHQ8_Moving` | int | 0-3 | Item 8: Moving/speaking slowly or fidgety |

### PHQ-8 Item Score Meaning

Each item scored 0-3 based on frequency over past 2 weeks:

| Score | Meaning |
|-------|---------|
| 0 | Not at all |
| 1 | Several days |
| 2 | More than half the days |
| 3 | Nearly every day |

### Severity Levels

Derived from total PHQ-8 score (0-24):

| Score Range | Severity Level | MDD Classification |
|-------------|----------------|-------------------|
| 0-4 | None/Minimal | No MDD |
| 5-9 | Mild | No MDD |
| 10-14 | Moderate | MDD |
| 15-19 | Moderately Severe | MDD |
| 20-24 | Severe | MDD |

### Synthetic Example (train CSV)

```csv
Participant_ID,PHQ8_Binary,PHQ8_Score,Gender,PHQ8_NoInterest,PHQ8_Depressed,PHQ8_Sleep,PHQ8_Tired,PHQ8_Appetite,PHQ8_Failure,PHQ8_Concentrating,PHQ8_Moving
300,0,3,1,0,0,1,1,0,0,1,0
301,0,7,0,1,1,1,1,1,0,1,1
302,1,15,1,2,2,2,2,1,2,2,2
303,0,0,0,0,0,0,0,0,0,0,0
304,1,20,0,2,3,3,3,3,3,3,0
```

### Column Mapping in Code

`GroundTruthService.COLUMN_MAPPING` in `services/ground_truth.py`:

```python
COLUMN_MAPPING = {
    "PHQ8_NoInterest": PHQ8Item.NO_INTEREST,
    "PHQ8_Depressed": PHQ8Item.DEPRESSED,
    "PHQ8_Sleep": PHQ8Item.SLEEP,
    "PHQ8_Tired": PHQ8Item.TIRED,
    "PHQ8_Appetite": PHQ8Item.APPETITE,
    "PHQ8_Failure": PHQ8Item.FAILURE,
    "PHQ8_Concentrating": PHQ8Item.CONCENTRATING,
    "PHQ8_Moving": PHQ8Item.MOVING,
}
```

---

## Test Split Format

### AVEC2017 Test Split

**File**: `test_split_Depression_AVEC2017.csv`

**Note**: Does NOT include PHQ-8 scores (evaluation set).

| Column | Type | Description |
|--------|------|-------------|
| `participant_ID` | int | Note: lowercase 'p' (column name differs) |
| `Gender` | int | 0 = male, 1 = female |

### Full Test Split (if available)

**File**: `full_test_split.csv`

Some distributions include total scores but NOT item-wise scores:

| Column | Type | Description |
|--------|------|-------------|
| `Participant_ID` | int | Note: uppercase 'P' |
| `PHQ_Binary` | int | Note: no '8' in column name |
| `PHQ_Score` | int | Note: no '8' in column name |
| `Gender` | int | |

---

## Data Splits

### AVEC2017 Official Splits

| Split | Count | PHQ-8 Items | Purpose |
|-------|-------|-------------|---------|
| Train | 107 | Available | Model training, few-shot retrieval |
| Dev | 35 | Available | Hyperparameter tuning |
| Test | 47 | **Not available** | Final evaluation |

### Paper Re-Split (Section 2.4.1)

The paper creates a custom 58/43/41 split from the 142 labeled participants:

| Split | Count | Percentage | Purpose |
|-------|-------|------------|---------|
| Train | 58 | 41% | Few-shot reference store |
| Dev | 43 | 30% | Hyperparameter tuning |
| Test | 41 | 29% | Final evaluation |

**Implementation**: See `scripts/generate_embeddings.py` for split logic.

---

## Embeddings Format

### File Structure

```
data/embeddings/
├── reference_embeddings.npz   # NumPy compressed archive
└── reference_embeddings.json  # Text sidecar (participant IDs, chunks)
```

### NPZ Format

Contains numpy arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `embeddings` | (N, 4096) | Embedding vectors |
| `participant_ids` | (N,) | Source participant IDs |
| `item_indices` | (N,) | PHQ-8 item index (0-7) |
| `scores` | (N,) | Ground truth scores (0-3) |

### JSON Sidecar

```json
{
  "model": "qwen3-embedding:8b",
  "dimension": 4096,
  "chunk_size": 8,
  "chunk_step": 2,
  "created_at": "2025-12-21T10:00:00Z",
  "participant_count": 58,
  "total_embeddings": 1856,
  "chunks": [
    {
      "participant_id": 300,
      "item": "NO_INTEREST",
      "score": 0,
      "text": "Ellie: how are you doing...",
      "embedding_index": 0
    },
    ...
  ]
}
```

### Configuration

From `EmbeddingSettings` in `config.py`:

| Setting | Default | Paper Reference |
|---------|---------|-----------------|
| `EMBEDDING_DIMENSION` | 4096 | Appendix D |
| `EMBEDDING_CHUNK_SIZE` | 8 | Appendix D |
| `EMBEDDING_CHUNK_STEP` | 2 | Section 2.4.2 |
| `EMBEDDING_TOP_K_REFERENCES` | 2 | Appendix D |

---

## Raw Download Structure

### Before Preparation

```
downloads/
├── participants/
│   ├── 300_P.zip           # ~475MB each
│   ├── 301_P.zip
│   └── .../
├── train_split_Depression_AVEC2017.csv
├── dev_split_Depression_AVEC2017.csv
├── test_split_Depression_AVEC2017.csv
├── full_test_split.csv
└── DAICWOZDepression_Documentation_AVEC2017.pdf
```

### Zip Contents (per participant)

| File | Size | Used by System |
|------|------|----------------|
| `{id}_TRANSCRIPT.csv` | ~10KB | **YES** - Primary input |
| `{id}_AUDIO.wav` | ~20MB | Future (multimodal) |
| `{id}_COVAREP.csv` | ~37MB | Future |
| `{id}_FORMANT.csv` | ~2MB | Future |
| `{id}_CLNF_AUs.txt` | ~2MB | Future |
| `{id}_CLNF_features.txt` | ~24MB | Future |
| `{id}_CLNF_features3D.txt` | ~36MB | Future |
| `{id}_CLNF_gaze.txt` | ~3MB | Future |
| `{id}_CLNF_hog.txt` | ~350MB | Future |
| `{id}_CLNF_pose.txt` | ~2MB | Future |

---

## Domain Model Mapping

### Transcript → Entity

`domain/entities.py`:

```python
@dataclass
class Transcript:
    participant_id: int      # From directory name ({id}_P)
    text: str                # Formatted dialogue
    created_at: datetime     # Load timestamp
    id: UUID                 # Instance UUID
```

### Ground Truth → Entity

```python
@dataclass
class PHQ8Assessment:
    items: Mapping[PHQ8Item, ItemAssessment]  # All 8 items
    mode: AssessmentMode                       # ZERO_SHOT or FEW_SHOT
    participant_id: int
```

### PHQ8Item Enum

`domain/enums.py`:

```python
class PHQ8Item(StrEnum):
    NO_INTEREST = "NO_INTEREST"      # Item 1
    DEPRESSED = "DEPRESSED"          # Item 2
    SLEEP = "SLEEP"                  # Item 3
    TIRED = "TIRED"                  # Item 4
    APPETITE = "APPETITE"            # Item 5
    FAILURE = "FAILURE"              # Item 6
    CONCENTRATING = "CONCENTRATING"  # Item 7
    MOVING = "MOVING"                # Item 8
```

---

## Validation Checklist

When working with data, verify:

- [ ] Participant ID exists (not all 300-492 are present)
- [ ] Transcript file is tab-separated, not comma-separated
- [ ] Speaker column contains "Ellie" or "Participant"
- [ ] Ground truth CSV uses `Participant_ID` (uppercase P)
- [ ] Test split uses `participant_ID` (lowercase p)
- [ ] PHQ-8 scores are in range 0-3 (items) or 0-24 (total)
- [ ] Embeddings dimension matches model (4096 for qwen3-embedding:8b)

---

## See Also

- [Spec 04A: Data Organization](../specs/04A_DATA_ORGANIZATION.md) - Implementation details
- [Spec 05: Transcript Service](../specs/05_TRANSCRIPT_SERVICE.md) - Loading logic
- [Spec 08: Embedding Service](../specs/08_EMBEDDING_SERVICE.md) - Few-shot retrieval
- [Model Registry](../models/MODEL_REGISTRY.md) - Embedding model options
