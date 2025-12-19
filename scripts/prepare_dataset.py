#!/usr/bin/env python3
"""Prepare DAIC-WOZ dataset from raw downloads.

This script extracts transcripts from participant zip files and organizes
them into the canonical directory structure expected by the codebase.

Usage:
    python scripts/prepare_dataset.py --downloads-dir downloads --output-dir data

    # Validate only
    python scripts/prepare_dataset.py --validate-only

    # Extract audio too (for future multimodal work)
    python scripts/prepare_dataset.py --include-audio

Requirements:
    - Raw DAIC-WOZ downloads in downloads/participants/
    - Split CSVs in downloads/

Spec Reference: docs/specs/04A_DATA_ORGANIZATION.md
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

# Try to import pandas for validation, but make it optional for extraction
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def log_info(msg: str, **kwargs: Any) -> None:
    """Log info message."""
    extras = " ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"[INFO] {msg} {extras}".strip())


def log_warning(msg: str, **kwargs: Any) -> None:
    """Log warning message."""
    extras = " ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"[WARN] {msg} {extras}".strip())


def log_error(msg: str, **kwargs: Any) -> None:
    """Log error message."""
    extras = " ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"[ERROR] {msg} {extras}".strip())


def _read_first_matching(zf: zipfile.ZipFile, suffix: str) -> bytes | None:
    """Read the first zip member matching a suffix."""
    for name in zf.namelist():
        if name.endswith(suffix):
            return zf.read(name)
    return None


def _extract_member(
    zf: zipfile.ZipFile,
    suffix: str,
    target_path: Path,
    label: str,
    zip_path: Path,
) -> bool:
    """Extract a matching zip member to target_path."""
    data = _read_first_matching(zf, suffix)
    if data is None:
        log_warning(f"No {label} in zip", zip_path=str(zip_path))
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(data)
    return True


def _needs_extraction(transcript_path: Path, audio_path: Path | None) -> tuple[bool, bool]:
    """Return whether transcript/audio extraction is needed."""
    need_transcript = not transcript_path.exists()
    need_audio = audio_path is not None and not audio_path.exists()
    return need_transcript, need_audio


def _log_progress(idx: int, total: int) -> None:
    """Log extraction progress every 20 participants."""
    if idx % 20 == 0 or idx == total:
        print(f"  Progress: {idx}/{total} participants...")


def _extract_requested(
    zf: zipfile.ZipFile,
    zip_path: Path,
    transcript_path: Path,
    audio_path: Path | None,
    need_transcript: bool,
    need_audio: bool,
) -> tuple[bool, int]:
    """Extract requested members and return (extracted_any, errors)."""
    extracted_any = False
    errors = 0

    if need_transcript:
        if _extract_member(
            zf,
            "_TRANSCRIPT.csv",
            transcript_path,
            "transcript",
            zip_path,
        ):
            extracted_any = True
        else:
            errors += 1

    if need_audio:
        if audio_path is not None and _extract_member(
            zf,
            "_AUDIO.wav",
            audio_path,
            "audio",
            zip_path,
        ):
            extracted_any = True
        else:
            errors += 1

    return extracted_any, errors


def extract_transcripts(
    downloads_dir: Path,
    output_dir: Path,
    include_audio: bool = False,
) -> dict[str, int]:
    """Extract transcript files from participant zips.

    Args:
        downloads_dir: Path to downloads directory containing participants/.
        output_dir: Path to output data directory.
        include_audio: If True, also extract audio files.

    Returns:
        Dictionary with extraction statistics.
    """
    participants_dir = downloads_dir / "participants"
    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = output_dir / "audio" if include_audio else None
    if audio_dir is not None:
        audio_dir.mkdir(parents=True, exist_ok=True)

    stats = {"extracted": 0, "skipped": 0, "errors": 0}

    zip_files = sorted(participants_dir.glob("*_P.zip"))
    total = len(zip_files)

    for idx, zip_path in enumerate(zip_files, 1):
        participant_id = zip_path.stem  # e.g., "300_P"
        pid_num = participant_id.replace("_P", "")
        output_subdir = transcripts_dir / participant_id
        transcript_name = f"{pid_num}_TRANSCRIPT.csv"
        transcript_path = output_subdir / transcript_name
        audio_path = audio_dir / f"{pid_num}_AUDIO.wav" if audio_dir is not None else None

        need_transcript, need_audio = _needs_extraction(transcript_path, audio_path)

        _log_progress(idx, total)

        # Skip if everything requested is already present
        if not (need_transcript or need_audio):
            stats["skipped"] += 1
            continue

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                extracted_any, errors = _extract_requested(
                    zf,
                    zip_path,
                    transcript_path,
                    audio_path,
                    need_transcript,
                    need_audio,
                )
                stats["errors"] += errors
                if extracted_any:
                    stats["extracted"] += 1

        except zipfile.BadZipFile as e:
            log_error("Bad zip file", zip_path=str(zip_path), error=str(e))
            stats["errors"] += 1
        except Exception as e:
            log_error("Extraction failed", zip_path=str(zip_path), error=str(e))
            stats["errors"] += 1

    return stats


def copy_split_csvs(downloads_dir: Path, output_dir: Path) -> int:
    """Copy ground truth CSV files to data directory.

    Args:
        downloads_dir: Path to downloads directory.
        output_dir: Path to output data directory.

    Returns:
        Number of files copied.
    """
    csv_files = [
        "train_split_Depression_AVEC2017.csv",
        "dev_split_Depression_AVEC2017.csv",
        "test_split_Depression_AVEC2017.csv",
        "full_test_split.csv",
    ]

    copied = 0
    for csv_name in csv_files:
        src = downloads_dir / csv_name
        dst = output_dir / csv_name

        if src.exists():
            shutil.copy2(src, dst)
            log_info("Copied CSV", file=csv_name)
            copied += 1
        else:
            log_warning("CSV not found", file=csv_name, path=str(src))

    return copied


def _sample_transcript_lines(transcripts_dir: Path) -> int:
    """Return line count for a sample transcript, if available."""
    transcript_dirs = list(transcripts_dir.glob("*_P"))
    if not transcript_dirs:
        return 0
    sample_files = list(transcript_dirs[0].glob("*_TRANSCRIPT.csv"))
    if not sample_files:
        return 0
    try:
        with sample_files[0].open() as handle:
            return len(handle.readlines())
    except Exception:
        return 0


def _find_pid_column(columns: list[str]) -> str | None:
    """Find participant ID column name in a CSV header."""
    for col_name in ["Participant_ID", "participant_ID", "participant_id"]:
        if col_name in columns:
            return col_name
    return None


def _missing_transcripts(pids: list[int], transcripts_dir: Path) -> list[int]:
    """Return participant IDs missing transcript directories."""
    missing = []
    for pid in pids:
        transcript_dir = transcripts_dir / f"{pid}_P"
        if not transcript_dir.exists():
            missing.append(int(pid))
    return missing


def _update_split_results(
    results: dict[str, Any],
    split_name: str,
    csv_path: Path,
    transcripts_dir: Path,
) -> None:
    """Update results for a single split CSV."""
    if not csv_path.exists():
        log_warning("Split CSV not found", split=split_name)
        return

    df = pd.read_csv(csv_path)
    results[f"{split_name}_count"] = len(df)

    pid_col = _find_pid_column(list(df.columns))
    if pid_col is None:
        log_warning(
            "No participant ID column found",
            split=split_name,
            columns=list(df.columns),
        )
        return

    missing = _missing_transcripts(df[pid_col].tolist(), transcripts_dir)
    if missing:
        results["missing_transcripts"].extend(missing)
        results["valid"] = False


def validate_dataset(output_dir: Path) -> dict[str, Any]:
    """Validate the prepared dataset.

    Args:
        output_dir: Path to data directory.

    Returns:
        Validation results.
    """
    results: dict[str, Any] = {
        "valid": True,
        "transcript_count": 0,
        "train_count": 0,
        "dev_count": 0,
        "test_count": 0,
        "missing_transcripts": [],
        "sample_transcript_lines": 0,
    }

    transcripts_dir = output_dir / "transcripts"
    if transcripts_dir.exists():
        transcript_dirs = list(transcripts_dir.glob("*_P"))
        results["transcript_count"] = len(transcript_dirs)
        results["sample_transcript_lines"] = _sample_transcript_lines(transcripts_dir)

    if not HAS_PANDAS:
        log_warning("pandas not installed, skipping split validation")
        return results

    for split_name, csv_name in [
        ("train", "train_split_Depression_AVEC2017.csv"),
        ("dev", "dev_split_Depression_AVEC2017.csv"),
        ("test", "test_split_Depression_AVEC2017.csv"),
    ]:
        csv_path = output_dir / csv_name
        _update_split_results(results, split_name, csv_path, transcripts_dir)

    return results


def print_summary(results: dict[str, Any]) -> None:
    """Print validation summary."""
    print("\n" + "=" * 60)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Transcripts extracted: {results['transcript_count']}")
    print(f"  Train split:           {results['train_count']} participants")
    print(f"  Dev split:             {results['dev_count']} participants")
    print(f"  Test split:            {results['test_count']} participants")

    if results["sample_transcript_lines"]:
        print(f"  Sample transcript:     {results['sample_transcript_lines']} lines")

    if results["missing_transcripts"]:
        print(f"\n  MISSING TRANSCRIPTS:   {results['missing_transcripts'][:10]}")
        if len(results["missing_transcripts"]) > 10:
            remaining = len(results["missing_transcripts"]) - 10
            print(f"                         ... and {remaining} more")

    print("=" * 60)

    if results["valid"]:
        print("✓ Dataset is valid and ready for use!")
    else:
        print("✗ Dataset has issues - see missing transcripts above")

    print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare DAIC-WOZ dataset from raw downloads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard extraction (transcripts only)
    python scripts/prepare_dataset.py

    # Custom paths
    python scripts/prepare_dataset.py --downloads-dir /path/to/downloads --output-dir /path/to/data

    # Include audio files (for future multimodal work)
    python scripts/prepare_dataset.py --include-audio

    # Validate existing dataset
    python scripts/prepare_dataset.py --validate-only
        """,
    )
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=Path("downloads"),
        help="Path to raw downloads directory (default: downloads)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Path to output data directory (default: data)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing dataset, don't extract",
    )
    parser.add_argument(
        "--include-audio",
        action="store_true",
        help="Also extract audio files (adds ~4GB)",
    )
    args = parser.parse_args()

    # Validate-only mode
    if args.validate_only:
        if not args.output_dir.exists():
            log_error("Output directory does not exist", path=str(args.output_dir))
            return 1
        results = validate_dataset(args.output_dir)
        print_summary(results)
        return 0 if results["valid"] else 1

    # Check downloads exist
    participants_dir = args.downloads_dir / "participants"
    if not participants_dir.exists():
        log_error(
            "Downloads directory not found",
            expected=str(participants_dir),
            hint="Run download script first or specify --downloads-dir",
        )
        return 1

    zip_count = len(list(participants_dir.glob("*_P.zip")))
    if zip_count == 0:
        log_error("No participant zip files found", path=str(participants_dir))
        return 1

    print(f"\nFound {zip_count} participant zip files")
    print(f"Extracting to: {args.output_dir}")
    if args.include_audio:
        print("Including audio files (this will use more disk space)")
    print()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Extract transcripts
    print("Extracting transcripts...")
    stats = extract_transcripts(args.downloads_dir, args.output_dir, args.include_audio)
    summary = (
        f"  Extracted: {stats['extracted']}, "
        f"Skipped: {stats['skipped']}, "
        f"Errors: {stats['errors']}"
    )
    print(summary)

    # Copy split CSVs
    print("\nCopying split CSVs...")
    copied = copy_split_csvs(args.downloads_dir, args.output_dir)
    print(f"  Copied {copied} CSV files")

    # Create embeddings directory (placeholder)
    embeddings_dir = args.output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCreated embeddings directory: {embeddings_dir}")

    # Validate
    print("\nValidating dataset...")
    results = validate_dataset(args.output_dir)
    print_summary(results)

    return 0 if results["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
