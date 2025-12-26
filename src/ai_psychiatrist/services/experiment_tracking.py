"""Experiment tracking with full provenance for paper reproduction.

This module is used by `scripts/reproduce_results.py` to capture:
- Git commit + dirty state
- Environment details (python, platform)
- Embedding artifact identity via checksums
- Per-experiment provenance for multi-mode runs
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

import yaml

from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from ai_psychiatrist.config import Settings

logger = get_logger(__name__)


def get_git_info() -> tuple[str, bool]:
    """Get current git commit and dirty status.

    Returns:
        Tuple of (short_sha, is_dirty).
    """
    try:
        # Let git find the repository root automatically from current working directory
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

        logger.debug("Git info captured", commit=commit, dirty=dirty)
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Failed to capture git info, using fallback")
        return "unknown", True


def compute_file_checksum(path: Path) -> str | None:
    """Compute SHA256 checksum (first 16 chars).

    Returns:
        Checksum string or None if file doesn't exist or is not a regular file.
    """
    if not path.exists() or not path.is_file():
        if path.exists() and not path.is_file():
            logger.warning("Cannot compute checksum for non-file path", path=str(path))
        return None

    sha256 = hashlib.sha256()
    with path.open("rb") as f:
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
    """Shared metadata for a reproduction run (captured once)."""

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

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dict."""
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
    participants_requested: int
    participants_evaluated: int

    @classmethod
    def capture(
        cls,
        *,
        mode: Literal["zero_shot", "few_shot"],
        split: str,
        settings: Settings,
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

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dict."""
        return asdict(self)


def update_experiment_registry(
    *,
    run_metadata: RunMetadata,
    experiments: list[dict[str, object]],
    output_file: Path,
    registry_path: Path = Path("data/experiments/registry.yaml"),
) -> None:
    """Update the experiment registry YAML."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    if registry_path.exists():
        with registry_path.open("r") as f:
            registry: dict[str, object] = yaml.safe_load(f) or {"runs": {}}
    else:
        registry = {"runs": {}}

    runs = registry.get("runs")
    if not isinstance(runs, dict):
        runs = {}
        registry["runs"] = runs

    run_id = run_metadata.run_id

    experiments_list: list[dict[str, object]] = []
    for exp in experiments:
        provenance = exp.get("provenance")
        results = exp.get("results")

        if not isinstance(provenance, dict) or not isinstance(results, dict):
            logger.warning(
                "Skipping malformed experiment entry",
                has_provenance=isinstance(provenance, dict),
                has_results=isinstance(results, dict),
            )
            continue

        coverage_frac = results.get("prediction_coverage")
        coverage_pct: float | None
        if isinstance(coverage_frac, (int, float)):
            coverage_pct = round(float(coverage_frac) * 100, 2)
        else:
            coverage_pct = None

        experiments_list.append(
            {
                "mode": provenance.get("mode"),
                "split": provenance.get("split"),
                "mae_weighted": results.get("item_mae_weighted"),
                "coverage_pct": coverage_pct,
                "status": "completed",
            }
        )

    runs[run_id] = {
        "timestamp": run_metadata.timestamp,
        "git_commit": run_metadata.git_commit,
        "git_dirty": run_metadata.git_dirty,
        "output_file": output_file.name,
        "experiments": experiments_list,
    }

    tmp_path = registry_path.with_suffix(registry_path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        yaml.safe_dump(registry, f, default_flow_style=False, sort_keys=False)
    tmp_path.replace(registry_path)

    logger.info(
        "Updated experiment registry",
        run_id=run_id,
        num_experiments=len(experiments_list),
        registry_path=str(registry_path),
    )
