from __future__ import annotations

from ai_psychiatrist.infrastructure.hashing import (
    HASH_PREFIX_LENGTH,
    stable_bytes_hash,
    stable_text_hash,
)


class TestStableHashing:
    def test_stable_text_hash_is_deterministic_and_short(self) -> None:
        value = stable_text_hash("hello world")
        assert value == stable_text_hash("hello world")
        assert len(value) == HASH_PREFIX_LENGTH

    def test_stable_bytes_hash_matches_text_hash_for_utf8_payload(self) -> None:
        assert stable_bytes_hash(b"hello") == stable_text_hash("hello")
