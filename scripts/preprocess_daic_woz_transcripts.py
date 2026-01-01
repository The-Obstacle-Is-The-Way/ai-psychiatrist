#!/usr/bin/env python3
"""Deterministic DAIC-WOZ transcript preprocessing (bias-aware variants).

This script creates cleaned transcript variants in a separate output directory
without modifying the raw extracted transcripts.

See: docs/data/daic-woz-preprocessing.md
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

Variant = Literal["both_speakers_clean", "participant_only", "participant_qa"]

# Known interruption windows (seconds), per Bailey/Plumbley preprocessing tool.
INTERRUPTION_WINDOWS: dict[int, tuple[float, float]] = {
    373: (395.0, 428.0),
    444: (286.0, 387.0),
}

# Known sessions missing Ellie transcriptions in DAIC-WOZ.
MISSING_ELLIE_SESSIONS: set[int] = {451, 458, 480}

# Common sync markers found in transcripts.
SYNC_MARKERS: set[str] = {
    "<sync>",
    "<synch>",
    "[sync]",
    "[synch]",
    "[syncing]",
    "[synching]",
}

ALLOWED_SPEAKERS: set[str] = {"Ellie", "Participant"}


def _is_sync_marker(value: str) -> bool:
    """Return True if value is a sync marker (tolerate minor punctuation)."""
    normalized = value.strip().lower().rstrip(".")
    return (
        normalized in SYNC_MARKERS
        or normalized.startswith("<sync")
        or normalized.startswith("[sync")
    )


def _parse_participant_id(transcript_path: Path) -> int:
    """Parse participant ID from a DAIC-WOZ transcript path."""
    # Expected: .../{pid}_P/{pid}_TRANSCRIPT.csv
    try:
        return int(transcript_path.name.split("_", 1)[0])
    except (IndexError, ValueError) as e:  # pragma: no cover (defensive)
        raise ValueError(f"Could not parse participant id from {transcript_path}") from e


def _normalize_speaker(speaker: str) -> str:
    """Normalize speaker values to canonical labels."""
    s = speaker.strip()
    lower = s.lower()
    if lower == "ellie":
        return "Ellie"
    if lower == "participant":
        return "Participant"
    return s


def normalize_speaker(speaker: str) -> str:
    """Normalize and validate a speaker label.

    Raises:
        ValueError: If the speaker is not Ellie or Participant (case-insensitive).
    """
    normalized = _normalize_speaker(speaker)
    if normalized not in ALLOWED_SPEAKERS:
        raise ValueError(f"Unknown speaker: {speaker!r}")
    return normalized


def is_sync_marker(value: str) -> bool:
    """Return True if value is a transcript sync marker."""
    return _is_sync_marker(value)


def is_in_interruption_window(participant_id: int, start_time: float, stop_time: float) -> bool:
    """Return True if a row overlaps a known interruption window."""
    window = INTERRUPTION_WINDOWS.get(participant_id)
    if window is None:
        return False
    window_start, window_end = window
    return start_time < window_end and stop_time > window_start


def remove_preamble(df: pd.DataFrame, participant_id: int | None = None) -> pd.DataFrame:
    """Remove pre-interview preamble rows.

    If Ellie is present, drops rows before the first Ellie utterance. If Ellie is
    absent, drops leading sync markers / empty rows until the first real utterance.
    """
    del participant_id  # Warnings are tracked in stats/manifest, not here.
    if df.empty:
        return df.copy()

    has_ellie = bool((df["speaker"] == "Ellie").any())
    cleaned, _ = _drop_pre_intro(df, has_ellie=has_ellie)
    return cleaned


def apply_variant_filter(df: pd.DataFrame, variant: Variant) -> pd.DataFrame:
    """Apply the variant-specific speaker selection rule."""
    cleaned, _ = _apply_variant(df, variant)
    return cleaned


def clean_transcript(df: pd.DataFrame, *, participant_id: int, variant: Variant) -> pd.DataFrame:
    """Clean a transcript DataFrame according to deterministic rules."""
    required_cols = {"start_time", "stop_time", "speaker", "value"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns {sorted(missing_cols)}")

    cleaned = df.copy()

    # Drop rows with missing speaker/value
    cleaned = cleaned.dropna(subset=["speaker", "value"]).copy()

    # Validate and normalize speakers
    cleaned["speaker"] = cleaned["speaker"].astype(str).map(_normalize_speaker)
    unknown = sorted(set(cleaned["speaker"]) - ALLOWED_SPEAKERS)
    if unknown:
        raise ValueError(f"Unknown speakers: {unknown}")

    # Validate timestamps (fail-fast)
    try:
        cleaned["start_time"] = pd.to_numeric(cleaned["start_time"], errors="raise")
        cleaned["stop_time"] = pd.to_numeric(cleaned["stop_time"], errors="raise")
    except Exception as e:  # pragma: no cover (defensive, data-dependent)
        raise ValueError("Invalid start_time/stop_time values") from e

    if bool((cleaned["stop_time"] < cleaned["start_time"]).any()):
        raise ValueError("Found stop_time earlier than start_time")

    has_ellie = bool((cleaned["speaker"] == "Ellie").any())

    cleaned, _ = _drop_pre_intro(cleaned, has_ellie=has_ellie)
    cleaned, _ = _drop_sync_markers(cleaned)
    cleaned, _ = _drop_interruption_windows(cleaned, participant_id)
    cleaned, _ = _apply_variant(cleaned, variant)

    if not bool((cleaned["speaker"] == "Participant").any()):
        raise ValueError("Transcript has no participant utterances after preprocessing")

    return cleaned


@dataclass
class TranscriptPreprocessStats:
    participant_id: int
    input_path: str
    output_path: str
    has_ellie: bool
    rows_raw: int
    rows_out: int
    removed_missing_fields: int = 0
    removed_pre_intro: int = 0
    removed_sync_markers: int = 0
    removed_interruptions: int = 0
    removed_by_speaker_filter: int = 0
    warnings: list[str] = field(default_factory=list)


def _drop_pre_intro(df: pd.DataFrame, *, has_ellie: bool) -> tuple[pd.DataFrame, int]:
    """Drop rows before the interview starts."""
    if df.empty:
        return df, 0

    if has_ellie:
        ellie_mask = df["speaker"] == "Ellie"
        if not bool(ellie_mask.any()):
            return df, 0
        first_ellie_pos = int(ellie_mask.to_numpy().argmax())
        return df.iloc[first_ellie_pos:].copy(), first_ellie_pos

    # No Ellie present: drop leading sync markers (and any leading empties).
    values = df["value"].fillna("").astype(str).tolist()
    start_pos = 0
    for pos, v in enumerate(values):
        if v.strip() and not _is_sync_marker(v):
            start_pos = pos
            break
    return df.iloc[start_pos:].copy(), start_pos


def _drop_sync_markers(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0

    mask = df["value"].fillna("").astype(str).map(_is_sync_marker)
    removed = int(mask.sum())
    return df.loc[~mask].copy(), removed


def _drop_interruption_windows(df: pd.DataFrame, participant_id: int) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0

    window = INTERRUPTION_WINDOWS.get(participant_id)
    if window is None:
        return df, 0

    start, end = window
    mask = (df["start_time"] < end) & (df["stop_time"] > start)
    removed = int(mask.sum())
    return df.loc[~mask].copy(), removed


def _apply_variant(df: pd.DataFrame, variant: Variant) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0

    if variant == "both_speakers_clean":
        return df, 0

    if variant == "participant_only":
        mask = df["speaker"] == "Participant"
        removed = int((~mask).sum())
        return df.loc[mask].copy(), removed

    # participant_qa
    keep = [False] * len(df)
    last_ellie_pos: int | None = None

    # Iterate by position to build the keep mask.
    for pos, row in enumerate(df.itertuples(index=False)):
        speaker = str(row.speaker)
        if speaker == "Ellie":
            last_ellie_pos = pos
            continue
        if speaker == "Participant":
            if last_ellie_pos is not None and not keep[last_ellie_pos]:
                keep[last_ellie_pos] = True
            keep[pos] = True

    kept_count = sum(keep)
    removed = len(df) - kept_count
    return df.iloc[[i for i, k in enumerate(keep) if k]].copy(), removed


def preprocess_transcript(
    transcript_path: Path,
    *,
    output_path: Path,
    variant: Variant,
) -> tuple[pd.DataFrame, TranscriptPreprocessStats]:
    """Preprocess a single transcript TSV and return (processed_df, stats)."""
    participant_id = _parse_participant_id(transcript_path)

    df = pd.read_csv(
        transcript_path,
        sep="\t",
        dtype={"speaker": "string", "value": "string"},
    )
    rows_raw = len(df)

    required_cols = {"start_time", "stop_time", "speaker", "value"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns {sorted(missing_cols)} in {transcript_path}")

    # Drop rows with missing speaker/value
    before = len(df)
    df = df.dropna(subset=["speaker", "value"]).copy()
    removed_missing_fields = int(before - len(df))

    # Validate timestamps (fail-fast)
    df["start_time"] = pd.to_numeric(df["start_time"], errors="raise")
    df["stop_time"] = pd.to_numeric(df["stop_time"], errors="raise")
    if bool((df["stop_time"] < df["start_time"]).any()):
        raise ValueError(f"Found stop_time earlier than start_time in {transcript_path}")

    # Normalize speakers and validate.
    df["speaker"] = df["speaker"].astype(str).map(_normalize_speaker)
    unknown = sorted(set(df["speaker"]) - ALLOWED_SPEAKERS)
    if unknown:
        raise ValueError(f"Unknown speakers in {transcript_path}: {unknown}")

    has_ellie = bool((df["speaker"] == "Ellie").any())

    stats = TranscriptPreprocessStats(
        participant_id=participant_id,
        input_path=str(transcript_path),
        output_path=str(output_path),
        has_ellie=has_ellie,
        rows_raw=rows_raw,
        rows_out=0,
        removed_missing_fields=removed_missing_fields,
    )

    if not has_ellie and participant_id not in MISSING_ELLIE_SESSIONS:
        stats.warnings.append("no_ellie_speaker_unexpected")

    # 1) Drop pre-intro
    df, removed_pre_intro = _drop_pre_intro(df, has_ellie=has_ellie)
    stats.removed_pre_intro = removed_pre_intro

    # 2) Drop sync markers
    df, removed_sync = _drop_sync_markers(df)
    stats.removed_sync_markers = removed_sync

    # 3) Drop interruption windows
    df, removed_interrupt = _drop_interruption_windows(df, participant_id)
    stats.removed_interruptions = removed_interrupt

    # 4) Apply variant speaker selection
    df, removed_by_filter = _apply_variant(df, variant)
    stats.removed_by_speaker_filter = removed_by_filter

    # Final sanity: must have participant content.
    if not bool((df["speaker"] == "Participant").any()):
        raise ValueError(
            f"Transcript has no participant utterances after preprocessing: {transcript_path}"
        )

    stats.rows_out = len(df)
    return df, stats


@dataclass
class Manifest:
    created_at: str
    variant: Variant
    input_dir: str
    output_dir: str
    transcript_count: int
    stats: list[TranscriptPreprocessStats]


def _write_manifest(manifest_path: Path, manifest: Manifest) -> None:
    payload: dict[str, Any] = asdict(manifest)
    # Serialize dataclass items
    payload["stats"] = [asdict(s) for s in manifest.stats]
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess DAIC-WOZ transcripts into clean variants"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/transcripts"),
        help="Input transcripts directory (default: data/transcripts)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for preprocessed transcripts (must be different from input)",
    )
    parser.add_argument(
        "--variant",
        choices=["both_speakers_clean", "participant_only", "participant_qa"],
        default="participant_only",
        help="Preprocessing variant to write (default: participant_only)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats and validate, but do not write outputs",
    )
    return parser.parse_args(argv)


def _collect_transcripts(input_dir: Path) -> list[Path]:
    transcript_paths = sorted(input_dir.glob("*_P/*_TRANSCRIPT.csv"))
    if not transcript_paths:
        raise ValueError(f"No transcript files found under {input_dir}")
    return transcript_paths


def _prepare_staging_dir(*, output_dir: Path, overwrite: bool, dry_run: bool) -> Path:
    staging_dir = output_dir.with_name(output_dir.name + ".tmp")

    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    if output_dir.exists() and not dry_run:
        if not overwrite:
            raise ValueError(f"Output dir exists (use --overwrite): {output_dir}")
        shutil.rmtree(output_dir)

    if not dry_run:
        staging_dir.mkdir(parents=True, exist_ok=True)

    return staging_dir


def _process_all(
    transcript_paths: list[Path],
    *,
    staging_dir: Path,
    variant: Variant,
    dry_run: bool,
) -> list[TranscriptPreprocessStats]:
    stats: list[TranscriptPreprocessStats] = []
    for transcript_path in transcript_paths:
        pid = _parse_participant_id(transcript_path)
        out_path = staging_dir / f"{pid}_P" / transcript_path.name

        processed_df, s = preprocess_transcript(
            transcript_path,
            output_path=out_path,
            variant=variant,
        )
        stats.append(s)

        if not dry_run:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            processed_df.to_csv(out_path, sep="\t", index=False)

    return stats


def _finalize_output(
    stats: list[TranscriptPreprocessStats],
    *,
    staging_dir: Path,
    input_dir: Path,
    output_dir: Path,
    variant: Variant,
) -> None:
    manifest = Manifest(
        created_at=datetime.now(tz=UTC).isoformat(),
        variant=variant,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        transcript_count=len(stats),
        stats=stats,
    )
    _write_manifest(staging_dir / "preprocess_manifest.json", manifest)
    staging_dir.rename(output_dir)


def _print_summary(
    stats: list[TranscriptPreprocessStats], *, variant: Variant, output_dir: Path, dry_run: bool
) -> None:
    warnings = sum(len(s.warnings) for s in stats)
    print("PREPROCESS COMPLETE")
    print(f"  Variant: {variant}")
    print(f"  Transcripts: {len(stats)}")
    print(f"  Total warnings: {warnings}")
    if dry_run:
        print("  Mode: DRY-RUN (no files written)")
    else:
        print(f"  Output: {output_dir}")
        print(f"  Manifest: {output_dir / 'preprocess_manifest.json'}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    variant: Variant = args.variant

    if input_dir.resolve() == output_dir.resolve():
        print("ERROR: --output-dir must be different from --input-dir", file=sys.stderr)
        return 1

    try:
        transcript_paths = _collect_transcripts(input_dir)
        staging_dir = _prepare_staging_dir(
            output_dir=output_dir,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    try:
        stats = _process_all(
            transcript_paths,
            staging_dir=staging_dir,
            variant=variant,
            dry_run=bool(args.dry_run),
        )
        if not args.dry_run:
            _finalize_output(
                stats,
                staging_dir=staging_dir,
                input_dir=input_dir,
                output_dir=output_dir,
                variant=variant,
            )
    except Exception as e:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    _print_summary(stats, variant=variant, output_dir=output_dir, dry_run=bool(args.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
