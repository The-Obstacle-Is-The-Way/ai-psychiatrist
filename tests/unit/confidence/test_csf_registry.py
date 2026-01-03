"""Unit tests for confidence scoring function (CSF) registry (Spec 051)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from ai_psychiatrist.confidence.csf_registry import CSFRegistry

pytestmark = pytest.mark.unit

if TYPE_CHECKING:
    from collections.abc import Mapping


class TestCSFRegistry:
    def test_register_and_get(self) -> None:
        registry = CSFRegistry()

        @registry.register("example")
        def example_csf(item_signals: Mapping[str, Any]) -> float:
            return float(item_signals["x"])

        assert registry.get("example") is example_csf

    def test_get_unknown_raises(self) -> None:
        registry = CSFRegistry()
        with pytest.raises(ValueError, match="Unknown CSF"):
            registry.get("does_not_exist")

    def test_secondary_average_combines_two_csfs(self) -> None:
        registry = CSFRegistry()

        @registry.register("a")
        def csf_a(item_signals: Mapping[str, Any]) -> float:
            return float(item_signals["a"])

        @registry.register("b")
        def csf_b(item_signals: Mapping[str, Any]) -> float:
            return float(item_signals["b"])

        fn = registry.create_secondary("a", "b", combine="average")
        assert fn({"a": 0.2, "b": 0.6}) == pytest.approx(0.4)

    def test_parse_variant_secondary(self) -> None:
        registry = CSFRegistry()

        @registry.register("x")
        def csf_x(item_signals: Mapping[str, Any]) -> float:
            return float(item_signals["x"])

        @registry.register("y")
        def csf_y(item_signals: Mapping[str, Any]) -> float:
            return float(item_signals["y"])

        fn = registry.parse_variant("secondary:x+y:product")
        assert fn({"x": 0.5, "y": 0.2}) == pytest.approx(0.1)
