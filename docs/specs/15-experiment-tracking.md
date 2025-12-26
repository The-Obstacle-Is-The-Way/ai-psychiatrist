# Spec 15: Experiment Tracking with Full Provenance

> **STATUS: REQUIRED FOR PAPER REPRODUCTION**
>
> This spec addresses a critical gap discovered during BUG-023 investigation:
> historical runs could not be verified because provenance metadata was missing.
> Without proper experiment tracking, we cannot claim reproducibility.
>
> **Tracked by**: [GitHub Issue #53](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/53)
>
> **Last Updated**: 2025-12-25

---

## Objective

Implement professional experiment tracking with full provenance metadata so that
every reproduction run is verifiable, comparable, and auditable.

## Paper Reference

- **Section 4 (Discussion)**: "The stochastic nature of LLMs renders a key limitation...
  making it challenging to obtain consistent performance metrics."
- **Paper Results**: MAE 0.619 (few-shot) vs 0.796 (zero-shot) must be reproducible

## Problem Statement

### What Happened

On Dec 23, we had to delete `reproduction_results_20251223_014119.json` because:
1. Coverage was 74.1% but we couldn't prove if backfill was ON or OFF
2. No record of which embeddings were used
3. No record of model settings

This violates basic ML reproducibility standards.

### What's Missing

| Feature | Current State | Impact |
|---------|--------------|--------|
| Config snapshot | Partial (PR #52 added some) | Can't verify historical configs |
| Experiment namespace | Just timestamp | Can't identify run purpose |
| Feature flags in output | Not recorded | Can't verify experimental conditions |
| Experiment registry | None | No run comparison capability |
| Git commit hash | Not recorded | Can't reproduce exact code state |

---

## Goals

1. **Full Provenance**: Every output file contains complete configuration snapshot
2. **Semantic Naming**: Output filenames indicate mode/split/conditions
3. **Experiment Registry**: Central YAML tracking all runs with metrics
4. **Reproducibility Guarantee**: Any historical run can be exactly replicated

## Non-Goals

- MLflow/W&B integration (deferred to future spec)
- Web UI for experiment comparison
- Automatic hyperparameter tuning

---

## Deliverables

### 1. Enhanced Output Schema

Update `scripts/reproduce_results.py` output to include:

```python
# File: src/ai_psychiatrist/domain/entities.py (add or update)

@dataclass(frozen=True)
class ExperimentProvenance:
    """Complete experiment provenance for reproducibility."""

    # Identification
    experiment_id: str  # UUID or semantic name
    timestamp: datetime
    git_commit: str  # Short SHA
    git_dirty: bool  # Uncommitted changes?

    # Configuration
    split: Literal["paper", "dev", "train"]
    mode: Literal["few_shot", "zero_shot"]

    # Model settings
    quantitative_model: str  # e.g., "gemma3:27b"
    embedding_model: str  # e.g., "qwen3-embedding:8b"
    embedding_backend: Literal["ollama", "huggingface"]

    # Feature flags
    enable_keyword_backfill: bool
    enable_few_shot: bool

    # Embedding artifact
    embeddings_path: str
    embeddings_checksum: str  # SHA256 of NPZ file

    # Environment
    ollama_host: str
    python_version: str
    platform: str  # e.g., "darwin-arm64"

    # Results summary (filled after run)
    participants_evaluated: list[int]
    participants_failed: list[int]
    total_duration_seconds: float
```

### 2. Semantic Output Naming

**Format**: `{mode}_{split}_{backfill}_{timestamp}.json`

```python
# Examples:
"fewshot_paper_backfill-off_20251225_143022.json"
"zeroshot_dev_backfill-on_20251225_150000.json"
```

**Implementation** in `scripts/reproduce_results.py`:

```python
def generate_output_filename(
    mode: str,
    split: str,
    backfill: bool,
    timestamp: datetime,
) -> str:
    """Generate semantic output filename."""
    backfill_str = "backfill-on" if backfill else "backfill-off"
    ts = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"{mode}_{split}_{backfill_str}_{ts}.json"
```

### 3. Experiment Registry

**File**: `data/experiments/registry.yaml`

```yaml
# Experiment Registry
# Generated and maintained by scripts/reproduce_results.py
# DO NOT EDIT MANUALLY

experiments:
  fewshot_paper_backfill-off_20251225_143022:
    # Identification
    output_file: "fewshot_paper_backfill-off_20251225_143022.json"
    git_commit: "abc1234"
    git_dirty: false

    # Configuration
    split: paper
    mode: few_shot
    quantitative_model: gemma3:27b
    embedding_model: qwen3-embedding:8b
    embedding_backend: huggingface
    embeddings_file: huggingface_qwen3_8b_paper_train.npz
    enable_keyword_backfill: false

    # Results
    participants_evaluated: 41
    participants_failed: 0
    total_duration_seconds: 287.5

    # Metrics (summary)
    mae_weighted: 0.619
    mae_by_item: 0.605
    coverage_pct: 50.2
    binary_accuracy: 0.78

    # Status
    status: paper_parity  # or: baseline, experimental, failed
    notes: "First successful paper reproduction with HF embeddings"
```

### 4. Updated Reproduction Script

**File**: `scripts/reproduce_results.py` additions

```python
import subprocess
import hashlib
from pathlib import Path
import yaml

def get_git_info() -> tuple[str, bool]:
    """Get current git commit and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent.parent,
            text=True,
        ).strip()

        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).parent.parent,
            text=True,
        ).strip())

        return commit, dirty
    except subprocess.CalledProcessError:
        return "unknown", True


def compute_file_checksum(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]  # Short hash for readability


def build_provenance(
    settings: Settings,
    split: str,
    mode: str,
    start_time: datetime,
) -> dict:
    """Build complete provenance metadata."""
    git_commit, git_dirty = get_git_info()
    embeddings_path = resolve_reference_embeddings_path(
        settings.data, settings.embedding
    )

    return {
        "experiment_id": f"{mode}_{split}_{start_time.strftime('%Y%m%d_%H%M%S')}",
        "timestamp": start_time.isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "split": split,
        "mode": mode,
        "quantitative_model": settings.model.quantitative_model,
        "embedding_model": settings.model.embedding_model,
        "embedding_backend": settings.embedding_config.backend.value,
        "enable_keyword_backfill": settings.quantitative.enable_keyword_backfill,
        "enable_few_shot": settings.enable_few_shot,
        "embeddings_path": str(embeddings_path),
        "embeddings_checksum": compute_file_checksum(embeddings_path) if embeddings_path.exists() else "N/A",
        "ollama_host": f"{settings.ollama.host}:{settings.ollama.port}",
        "python_version": platform.python_version(),
        "platform": f"{platform.system().lower()}-{platform.machine()}",
    }


def update_experiment_registry(
    provenance: dict,
    metrics: dict,
    output_file: Path,
    status: str = "completed",
    notes: str = "",
) -> None:
    """Update the experiment registry YAML."""
    registry_path = Path("data/experiments/registry.yaml")
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing registry or create new
    if registry_path.exists():
        with open(registry_path) as f:
            registry = yaml.safe_load(f) or {"experiments": {}}
    else:
        registry = {"experiments": {}}

    # Add new experiment
    exp_id = provenance["experiment_id"]
    registry["experiments"][exp_id] = {
        "output_file": output_file.name,
        "git_commit": provenance["git_commit"],
        "git_dirty": provenance["git_dirty"],
        "split": provenance["split"],
        "mode": provenance["mode"],
        "quantitative_model": provenance["quantitative_model"],
        "embedding_model": provenance["embedding_model"],
        "embedding_backend": provenance["embedding_backend"],
        "embeddings_file": Path(provenance["embeddings_path"]).name,
        "enable_keyword_backfill": provenance["enable_keyword_backfill"],
        "participants_evaluated": metrics.get("participants_evaluated", 0),
        "participants_failed": metrics.get("participants_failed", 0),
        "total_duration_seconds": metrics.get("total_duration_seconds", 0),
        "mae_weighted": metrics.get("mae_weighted"),
        "mae_by_item": metrics.get("mae_by_item"),
        "coverage_pct": metrics.get("coverage_pct"),
        "binary_accuracy": metrics.get("binary_accuracy"),
        "status": status,
        "notes": notes,
    }

    # Write updated registry
    with open(registry_path, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)
```

---

## Implementation Plan

### Phase 1: Provenance in Output (Immediate)

1. Add `provenance` section to reproduction output JSON
2. Include git commit, config snapshot, checksums
3. **Acceptance**: Every new run has full provenance

### Phase 2: Semantic Naming (Immediate)

1. Update `reproduce_results.py` filename generation
2. Document naming convention
3. **Acceptance**: Filenames indicate mode/split/backfill

### Phase 3: Experiment Registry (Follow-up)

1. Create `data/experiments/registry.yaml`
2. Auto-update registry after each run
3. **Acceptance**: `cat data/experiments/registry.yaml` shows all runs

---

## Acceptance Criteria

- [ ] Output JSON includes `provenance` section with all fields
- [ ] Output filenames follow `{mode}_{split}_{backfill}_{timestamp}.json` format
- [ ] Experiment registry YAML exists and is auto-updated
- [ ] Git commit and dirty flag recorded
- [ ] Embeddings checksum recorded
- [ ] All feature flags recorded
- [ ] Preflight checklists updated to reference experiment tracking
- [ ] README documents the naming convention

---

## Testing

### Unit Tests

```python
def test_get_git_info_returns_commit_and_dirty():
    """Test git info extraction."""
    commit, dirty = get_git_info()
    assert len(commit) >= 7  # Short SHA
    assert isinstance(dirty, bool)


def test_compute_file_checksum():
    """Test file checksum computation."""
    # Create temp file with known content
    checksum = compute_file_checksum(Path("test_file.bin"))
    assert len(checksum) == 16  # Short hash


def test_generate_output_filename():
    """Test semantic filename generation."""
    ts = datetime(2025, 12, 25, 14, 30, 22)
    name = generate_output_filename("fewshot", "paper", False, ts)
    assert name == "fewshot_paper_backfill-off_20251225_143022.json"


def test_update_experiment_registry_creates_file():
    """Test registry creation."""
    # ... test implementation
```

### Integration Tests

```python
def test_full_run_creates_provenance():
    """Integration test: run creates full provenance."""
    # Run minimal reproduction
    # Check output JSON has provenance section
    # Check registry updated
```

---

## Design Principles Applied

### Single Responsibility (SRP)
- `ExperimentProvenance`: only holds provenance data
- `update_experiment_registry()`: only updates registry

### Open/Closed (OCP)
- Registry format is extensible (add new fields without breaking)
- Provenance dataclass can be extended via inheritance

### Don't Repeat Yourself (DRY)
- Git info extracted once, reused in provenance and registry
- Checksum function reused for any file

### Explicit is Better than Implicit
- All configuration recorded, nothing assumed
- Checksums verify exact artifact versions

---

## References

- GitHub Issue #53: Experiment tracking request
- BUG-023: Discovery of provenance gap
- SPEC-003: Backfill toggle that triggered discovery
- Paper Section 4: Reproducibility discussion
