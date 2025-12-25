# DAIC-WOZ Dataset Schema

**Purpose**: Enable development without direct data access
**Dataset**: Distress Analysis Interview Corpus - Wizard of Oz (DAIC-WOZ)
**Access**: Requires EULA from USC ICT
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
│   ├── paper_reference_embeddings.npz   # Default paper-style knowledge base
│   ├── paper_reference_embeddings.json
│   ├── paper_reference_embeddings.meta.json  # Optional: provenance metadata
│   ├── reference_embeddings.npz         # Optional: AVEC train knowledge base
│   └── reference_embeddings.json
├── paper_splits/                        # Optional: paper-style 58/43/41 split
│   ├── paper_split_train.csv
│   ├── paper_split_val.csv
│   ├── paper_split_test.csv
│   └── paper_split_metadata.json
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
    embeddings_path: Path = Path("data/embeddings/paper_reference_embeddings.npz")
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

`TranscriptService._parse_daic_woz_transcript()` in `src/ai_psychiatrist/services/transcript.py`:

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

`GroundTruthService.COLUMN_MAPPING` in `src/ai_psychiatrist/services/ground_truth.py`:

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

**Implementation**:
- `scripts/create_paper_split.py` generates `data/paper_splits/paper_split_{train,val,test}.csv`
  deterministically (seeded) from the AVEC2017 train+dev labeled set.
- `scripts/generate_embeddings.py --split paper-train` generates
  `data/embeddings/{backend}_{model_slug}_paper_train.{npz,json,meta.json}` by default, or use
  `--output data/embeddings/paper_reference_embeddings.npz` for the legacy filename.
- `scripts/reproduce_results.py --split paper` evaluates on the 41-participant paper test set and
  computes **item-level MAE** excluding N/A, matching the paper’s metric definition.

---

## Embeddings Format

### File Structure

```
data/embeddings/
├── paper_reference_embeddings.npz   # NumPy compressed archive
├── paper_reference_embeddings.json  # Text sidecar (participant IDs, chunks)
├── paper_reference_embeddings.meta.json  # Optional: provenance metadata
├── reference_embeddings.npz         # Optional: AVEC train knowledge base
└── reference_embeddings.json
```

### NPZ Format

The NPZ stores one array per participant:

- Key: `emb_{participant_id}` (example: `emb_300`)
- Value: `float32` array of shape `(num_chunks, EMBEDDING_DIMENSION)`

This matches `ReferenceStore._load_embeddings()` in `src/ai_psychiatrist/services/reference_store.py`.

### JSON Sidecar

```json
{
  "300": [
    "Ellie: ...\nParticipant: ...",
    "Ellie: ...\nParticipant: ...",
    "... (chunk text strings in the same order as the NPZ rows)"
  ],
  "301": ["..."],
  "...": ["..."]
}
```

The JSON maps participant ID (string) → list of chunk texts. The list order must match
the corresponding NPZ array row order for that participant.

### Configuration

From `EmbeddingSettings` in `src/ai_psychiatrist/config.py`:

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

`src/ai_psychiatrist/domain/entities.py`:

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

`src/ai_psychiatrist/domain/enums.py`:

```python
class PHQ8Item(StrEnum):
    NO_INTEREST = "NoInterest"      # Item 1
    DEPRESSED = "Depressed"         # Item 2
    SLEEP = "Sleep"                 # Item 3
    TIRED = "Tired"                 # Item 4
    APPETITE = "Appetite"           # Item 5
    FAILURE = "Failure"             # Item 6
    CONCENTRATING = "Concentrating" # Item 7
    MOVING = "Moving"               # Item 8
```

---

## Known Data Issues

| Participant | Issue | Status | Reference |
|-------------|-------|--------|-----------|
| **487** | Corrupted transcript (AppleDouble file, not CSV) | Resolved ✓ | [BUG-022](../archive/bugs/bug-022-corrupted-transcript-487.md) |

> **Note**: Issue was caused by macOS AppleDouble extraction, not source data. Re-download and careful extraction fixed it.

---

## Validation Checklist

When working with data, verify:

- [ ] Participant ID exists (not all 300-492 are present)
- [ ] Transcript file is tab-separated, not comma-separated
- [ ] Transcript is valid UTF-8 (not AppleDouble metadata)
- [ ] Speaker column contains "Ellie" or "Participant"
- [ ] Ground truth CSV uses `Participant_ID` (uppercase P)
- [ ] Test split uses `participant_ID` (lowercase p)
- [ ] PHQ-8 scores are in range 0-3 (items) or 0-24 (total)
- [ ] Embeddings dimension matches model (4096 for qwen3-embedding:8b)

---

## See Also

- [Legacy Spec 04A: Data Organization](../archive/specs/04A_DATA_ORGANIZATION.md) - Data preparation details
- [Legacy Spec 05: Transcript Service](../archive/specs/05_TRANSCRIPT_SERVICE.md) - Loading logic
- [Legacy Spec 08: Embedding Service](../archive/specs/08_EMBEDDING_SERVICE.md) - Few-shot retrieval
- [Model Registry](../models/model-registry.md) - Embedding model options
