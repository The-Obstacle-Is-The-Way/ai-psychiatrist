"""Tests for CLI entry point."""

import ai_psychiatrist
from ai_psychiatrist.cli import main


class TestCLI:
    """Test CLI entry point."""

    def test_main_returns_zero(self, capsys: object) -> None:
        """main() should return 0 on success."""
        result = main()
        assert result == 0

        captured = capsys.readouterr()
        assert f"AI Psychiatrist v{ai_psychiatrist.__version__}" in captured.out
