# Spec 15: Experiment Tracking with Full Provenance

> **STATUS: IMPLEMENTED**
>
> BUG-023 introduced baseline provenance. This spec upgrades it to **full provenance**
> with git info, checksums, semantic naming, **per-experiment tracking**, and a registry.
>
> **Tracked by**: [GitHub Issue #53](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/53)
>
> **Last Updated**: 2025-12-26

---

## Objective

Implement professional experiment tracking with full provenance metadata so that
every reproduction run is verifiable, comparable, and auditable.

## Paper Reference

- **Section 4 (Discussion)**: "The stochastic nature of LLMs renders a key limitation...
  making it challenging to obtain consistent performance metrics."
- **Paper Results**: MAE 0.619 (few-shot) vs 0.796 (zero-shot) must be reproducible

---

## Current State & Gaps

### What Exists (BUG-023 Fix)

`scripts/reproduce_results.py` records baseline provenance with settings.

### What's Missing

| Feature | Current State | Impact |
|---------|--------------|--------|
| Git commit / dirty flag | ❌ Not recorded | Can't reproduce exact code |
| Embeddings checksum | ❌ Not recorded | Can't verify artifact identity |
| `.meta.json` checksum | ❌ Not recorded | Can't verify metadata |
| Semantic filenames | ❌ Timestamp only | Can't identify run purpose |
| **Per-experiment provenance** | ❌ One for all modes | Misleading for multi-mode runs |
| Experiment registry | ❌ None | No centralized comparison |

---

## Critical Gap: Multi-Mode Runs

### The Problem

`reproduce_results.py` runs **both** zero-shot AND few-shot by default:

```python
# Lines 636-660: Runs BOTH modes unless restricted
if not args.zero_shot_only:
    experiments.append(await run_experiment(mode=ZERO_SHOT, ...))
if not args.few_shot_only:
    experiments.append(await run_experiment(mode=FEW_SHOT, ...))
```

Current provenance has a single `mode` field, which is ambiguous.

### The Solution

**Two-level provenance**:
1. **Run Metadata**: Shared across all experiments (git, timestamp, platform)
2. **Experiment Provenance**: Per-mode (embeddings, participants evaluated)

```json
{
    "run_metadata": {
        "run_id": "a1b2c3d4",
        "git_commit": "abc1234",
        "git_dirty": false,
        "timestamp": "2025-12-25T14:30:00Z"
    },
    "experiments": [
        {
            "mode": "zero_shot",
            "provenance": { ... },
            "results": { ... }
        },
        {
            "mode": "few_shot",
            "provenance": { ... },
            "results": { ... }
        }
    ]
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    reproduce_results.py                         │
│                                                                 │
│  1. Capture RunMetadata (git, timestamp, platform) - ONCE       │
│  2. For each mode (zero_shot, few_shot):                        │
│     - Capture ExperimentProvenance (embeddings, participants)   │
│     - Run experiment                                            │
│     - Attach provenance to results                              │
│  3. Save with semantic filename                                 │
│  4. Update experiment registry                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  data/outputs/few_shot_paper-test_backfill-off_20251225.json   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  data/experiments/registry.yaml (central index)                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Deliverables

### 1. Experiment Tracking Service

**File**: `src/ai_psychiatrist/services/experiment_tracking.py`

```python
"""Experiment tracking with full provenance.

References:
    - https://neptune.ai/blog/ml-experiment-tracking
    - https://madewithml.com/courses/mlops/experiment-tracking/
    - https://developer.ibm.com/blogs/10-diy-tips-for-machine-learning-experiment-tracking-and-reproducibility/
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ai_psychiatrist.config import Settings


def get_git_info() -> tuple[str, bool]:
    """Get current git commit and dirty status.

    Returns:
        Tuple of (short_sha, is_dirty).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit = result.stdout.strip()

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = bool(result.stdout.strip())

        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True


def compute_file_checksum(path: Path) -> str | None:
    """Compute SHA256 checksum (first 16 chars).

    Returns:
        Checksum string or None if file doesn't exist.
    """
    if not path.exists():
        return None

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def generate_output_filename(
    *,
    mode: str,
    split: str,
    backfill: bool,
    timestamp: datetime,
) -> str:
    """Generate semantic output filename.

    Format: {mode}_{split}_{backfill}_{timestamp}.json

    Examples:
        few_shot_paper-test_backfill-off_20251225_143022.json
        zero_shot_dev_backfill-on_20251225_150000.json
    """
    backfill_str = "backfill-on" if backfill else "backfill-off"
    ts = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"{mode}_{split}_{backfill_str}_{ts}.json"


@dataclass(frozen=True)
class RunMetadata:
    """Shared metadata for a reproduction run (captures ONCE)."""

    run_id: str
    timestamp: str
    git_commit: str
    git_dirty: bool
    python_version: str
    platform: str
    ollama_base_url: str

    @classmethod
    def capture(cls, *, ollama_base_url: str) -> RunMetadata:
        """Capture current run metadata."""
        git_commit, git_dirty = get_git_info()
        return cls(
            run_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            git_commit=git_commit,
            git_dirty=git_dirty,
            python_version=platform.python_version(),
            platform=f"{platform.system().lower()}-{platform.machine()}",
            ollama_base_url=ollama_base_url,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentProvenance:
    """Provenance for a single experiment (one mode)."""

    mode: Literal["zero_shot", "few_shot"]
    split: str
    quantitative_model: str
    embedding_model: str
    llm_backend: str
    embedding_backend: str
    enable_keyword_backfill: bool
    embeddings_path: str | None
    embeddings_checksum: str | None
    embeddings_meta_checksum: str | None
    participants_requested: int  # Total in ground truth
    participants_evaluated: int  # Actually ran (respects --limit)

    @classmethod
    def capture(
        cls,
        *,
        mode: Literal["zero_shot", "few_shot"],
        split: str,
        settings: "Settings",
        embeddings_path: Path | None,
        participants_requested: int,
        participants_evaluated: int,
    ) -> ExperimentProvenance:
        """Capture experiment provenance."""
        meta_path = embeddings_path.with_suffix(".meta.json") if embeddings_path else None

        return cls(
            mode=mode,
            split=split,
            quantitative_model=settings.model.quantitative_model,
            embedding_model=settings.model.embedding_model,
            llm_backend=settings.backend.backend.value,
            embedding_backend=settings.embedding_config.backend.value,
            enable_keyword_backfill=settings.quantitative.enable_keyword_backfill,
            embeddings_path=str(embeddings_path) if embeddings_path else None,
            embeddings_checksum=compute_file_checksum(embeddings_path) if embeddings_path else None,
            embeddings_meta_checksum=compute_file_checksum(meta_path) if meta_path else None,
            participants_requested=participants_requested,
            participants_evaluated=participants_evaluated,
        )

    def to_dict(self) -> dict:
        return asdict(self)
```

### 2. Updated Output Schema

```json
{
    "run_metadata": {
        "run_id": "a1b2c3d4",
        "timestamp": "2025-12-25T14:30:00Z",
        "git_commit": "abc1234",
        "git_dirty": false,
        "python_version": "3.12.0",
        "platform": "darwin-arm64",
        "ollama_base_url": "http://localhost:11434"
    },
    "experiments": [
        {
            "provenance": {
                "mode": "zero_shot",
                "split": "paper-test",
                "quantitative_model": "gemma3:27b",
                "embedding_model": "qwen3-embedding:8b",
                "llm_backend": "ollama",
                "embedding_backend": "huggingface",
                "enable_keyword_backfill": false,
                "embeddings_path": null,
                "embeddings_checksum": null,
                "embeddings_meta_checksum": null,
                "participants_requested": 41,
                "participants_evaluated": 41
            },
            "results": {
                "mode": "zero_shot",
                "total_subjects": 41,
                "item_mae_weighted": 0.796
            }
        },
        {
            "provenance": {
                "mode": "few_shot",
                "split": "paper-test",
                "embeddings_path": "data/embeddings/huggingface_qwen3_8b_paper_train.npz",
                "embeddings_checksum": "a1b2c3d4e5f6g7h8",
                "embeddings_meta_checksum": "i9j0k1l2m3n4o5p6",
                "participants_requested": 41,
                "participants_evaluated": 41
            },
            "results": {
                "mode": "few_shot",
                "total_subjects": 41,
                "item_mae_weighted": 0.619
            }
        }
    ]
}
```

### 3. Experiment Registry

**File**: `data/experiments/registry.yaml`

```yaml
# Experiment Registry
# Auto-generated by scripts/reproduce_results.py
# DO NOT EDIT MANUALLY

runs:
  a1b2c3d4:
    timestamp: "2025-12-25T14:30:00Z"
    git_commit: abc1234
    git_dirty: false
    output_file: few_shot_paper-test_backfill-off_20251225_143022.json

    experiments:
      - mode: zero_shot
        split: paper-test
        mae_weighted: 0.796
        coverage_pct: 48.2  # percent (0-100), not fraction
        status: completed

      - mode: few_shot
        split: paper-test
        mae_weighted: 0.619
        coverage_pct: 50.2  # percent (0-100), not fraction
        status: completed
```

### 4. Registry Update Function

Authoritative implementation: `src/ai_psychiatrist/services/experiment_tracking.py`.

Behavior (implemented):
- Uses `yaml.safe_load` / `yaml.safe_dump` (no `yaml.load`)
- Normalizes the registry shape (`{"runs": {...}}`) even if an older file is malformed
- Writes updates atomically (temp file + `replace`) to avoid partial/empty registries on crash
- Converts `prediction_coverage` (0–1 fraction) into `coverage_pct` (0–100 percent)

---

## Script Updates

**File**: `scripts/reproduce_results.py`

Key integration points:

```python
from ai_psychiatrist.services.experiment_tracking import (
    RunMetadata,
    ExperimentProvenance,
    generate_output_filename,
    update_experiment_registry,
)

async def main_async(args: argparse.Namespace) -> int:
    # 1. Capture run metadata ONCE at start
    run_metadata = RunMetadata.capture(ollama_base_url=settings.ollama.base_url)

    if run_metadata.git_dirty:
        logger.warning(
            "Running with uncommitted changes",
            git_commit=run_metadata.git_commit,
        )

    experiments_with_provenance = []

    # 2. For each mode, capture provenance and run
    for mode in modes_to_run:
        # Capture BEFORE running (participants_evaluated updated after)
        participants_requested = len(ground_truth)
        participant_ids = list(ground_truth.keys())[:args.limit] if args.limit else list(ground_truth.keys())
        participants_evaluated = len(participant_ids)

        provenance = ExperimentProvenance.capture(
            mode=mode.value,
            split=args.split,
            settings=settings,
            embeddings_path=embeddings_path if mode == AssessmentMode.FEW_SHOT else None,
            participants_requested=participants_requested,
            participants_evaluated=participants_evaluated,
        )

        result = await run_experiment(mode=mode, ...)

        experiments_with_provenance.append({
            "provenance": provenance.to_dict(),
            "results": result.to_dict(),
        })

    # 3. Save with semantic filename
    primary_mode = modes_to_run[-1].value  # Use last mode (few_shot if both)
    filename = generate_output_filename(
        mode=primary_mode,
        split=args.split,
        backfill=settings.quantitative.enable_keyword_backfill,
        timestamp=datetime.now(),
    )

    output_data = {
        "run_metadata": run_metadata.to_dict(),
        "experiments": experiments_with_provenance,
    }

    output_path = output_dir / filename
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # 4. Update registry
    update_experiment_registry(run_metadata, experiments_with_provenance, output_path)
```

---

## Acceptance Criteria

- [ ] `run_metadata` captures git commit, dirty flag, timestamp, platform
- [ ] Each experiment has its own `provenance` object
- [ ] `participants_evaluated` respects `--limit` flag correctly
- [ ] Embeddings `.npz` checksum recorded for few-shot
- [ ] Embeddings `.meta.json` checksum recorded when present
- [ ] Output filenames are semantic: `{mode}_{split}_{backfill}_{timestamp}.json`
- [ ] Experiment registry YAML auto-updated after each run
- [ ] Warning logged when `git_dirty=true`

---

## Testing

```python
def test_git_info_extraction():
    """Git info is extracted correctly."""
    commit, dirty = get_git_info()
    assert len(commit) >= 7 or commit == "unknown"
    assert isinstance(dirty, bool)


def test_file_checksum():
    """File checksum is computed correctly."""
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"test content")
        path = Path(f.name)

    checksum = compute_file_checksum(path)
    assert checksum is not None
    assert len(checksum) == 16
    path.unlink()


def test_semantic_filename():
    """Semantic filename is generated correctly."""
    ts = datetime(2025, 12, 25, 14, 30, 22)
    name = generate_output_filename("few_shot", "paper-test", False, ts)
    assert name == "few_shot_paper-test_backfill-off_20251225_143022.json"


def test_per_experiment_provenance():
    """Each experiment mode gets separate provenance."""
    # Verify structure has experiments list with individual provenance
```

---

## References

- [Neptune.ai: ML Experiment Tracking](https://neptune.ai/blog/ml-experiment-tracking)
- [Made With ML: Experiment Tracking](https://madewithml.com/courses/mlops/experiment-tracking/)
- [IBM: 10 Tips for ML Experiment Tracking](https://developer.ibm.com/blogs/10-diy-tips-for-machine-learning-experiment-tracking-and-reproducibility/)
- GitHub Issue #53: Experiment tracking request
- BUG-023: Discovery of provenance gap
