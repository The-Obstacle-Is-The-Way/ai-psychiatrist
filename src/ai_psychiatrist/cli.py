"""CLI entry point for AI Psychiatrist."""

from __future__ import annotations

import sys

import ai_psychiatrist


def main() -> int:
    """Main entry point for the AI Psychiatrist CLI."""
    print(f"AI Psychiatrist v{ai_psychiatrist.__version__}")
    print("Run with --help for usage information.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
