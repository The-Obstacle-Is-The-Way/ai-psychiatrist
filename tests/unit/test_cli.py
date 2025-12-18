"""Tests for CLI entry point."""

from ai_psychiatrist.cli import main


class TestCLI:
    """Test CLI entry point."""

    def test_main_returns_zero(self) -> None:
        """main() should return 0 on success."""
        result = main()
        assert result == 0
