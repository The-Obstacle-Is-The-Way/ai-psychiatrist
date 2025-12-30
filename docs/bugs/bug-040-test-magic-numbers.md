# BUG-040: Test Suite Magic Numbers Will Break

| Field | Value |
|-------|-------|
| **Status** | OPEN |
| **Severity** | MEDIUM |
| **Affects** | Test suite maintainability |
| **Introduced** | Original design |
| **Discovered** | 2025-12-30 |

## Summary

The test suite contains 94+ hard-coded numeric literals that mirror production defaults. When Spec 37/38 add new configuration options, many tests will break because assertions check hard-coded values instead of config references.

---

## High-Risk Hard-Coded Values

### 1. Timeout Assertions (Will Break When Timeout Becomes Configurable)

**File**: `tests/unit/infrastructure/llm/test_protocols.py`

```python
# Line 80
assert request.timeout_seconds == 300  # Hard-coded

# Line 192
assert request.timeout_seconds == 120  # Hard-coded embedding timeout
```

When Spec 37 adds `query_embed_timeout_seconds` config, these tests will need updating.

---

### 2. Dimension Assertions

**File**: `tests/unit/infrastructure/llm/test_protocols.py`

```python
# Lines 199, 203
EmbeddingRequest(..., dimension=4096)
assert req.dimension == 4096  # Hard-coded
```

**File**: `tests/unit/services/test_reference_store.py`

```python
# Line 43
EmbeddingSettings(dimension=4096, ...)

# Line 72
assert metadata["dimension"] == 4096  # Hard-coded
```

---

### 3. Test Expects Exception That Spec 38 Would Remove

**File**: `tests/unit/services/test_embedding.py:1074-1100`

```python
def test_mismatched_tags_length_raises(self, tmp_path: Path) -> None:
    # ...
    with pytest.raises(EmbeddingArtifactMismatchError, match="Tag count mismatch"):
        store._load_embeddings()
```

This test EXPECTS the exception to be raised. If we implement "graceful degradation" (which is wrong anyway), this test breaks.

**Correct Approach**: Keep the test, because the behavior SHOULD crash when `enable_item_tag_filter=True` and tags are invalid.

---

### 4. Feedback Loop Thresholds

**File**: `tests/integration/test_qualitative_pipeline.py`

```python
# Lines 121, 184, 241
FeedbackLoopSettings(score_threshold=3, ...)  # Hard-coded in 6 places
```

---

## Pattern Analysis

| Category | Count | Example |
|----------|-------|---------|
| Timeout values | 14 | `120`, `300`, `600` |
| Dimension values | 25+ | `4096`, `768`, `256`, `2` |
| Top-k values | 10 | `2`, `3`, `10` |
| Score thresholds | 11 | `3`, `4` |
| Chunk size/step | 15+ | `8`, `2`, `4` |

---

## Fix

### 1. Extract Test Constants

Create `tests/conftest.py` constants:

```python
# Paper-optimal defaults for testing
TEST_PAPER_DIMENSION = 4096
TEST_PAPER_CHUNK_SIZE = 8
TEST_PAPER_CHUNK_STEP = 2
TEST_PAPER_TOP_K = 2
TEST_PAPER_SCORE_THRESHOLD = 3

# Mock dimensions for unit tests (don't need real values)
TEST_MOCK_DIMENSION_SMALL = 2
TEST_MOCK_DIMENSION_MEDIUM = 256
```

### 2. Reference Config in Assertions

```python
# Before
assert request.timeout_seconds == 120

# After
from ai_psychiatrist.infrastructure.llm.protocols import EmbeddingRequest
assert request.timeout_seconds == EmbeddingRequest.__dataclass_fields__["timeout_seconds"].default
```

Or better:
```python
# After (using test constant)
assert request.timeout_seconds == TEST_DEFAULT_EMBED_TIMEOUT
```

### 3. Use Fixtures with Config

```python
@pytest.fixture
def embedding_settings() -> EmbeddingSettings:
    """Return settings matching paper-optimal defaults."""
    return EmbeddingSettings()  # Uses defaults from config
```

---

## Priority

**Before Spec 37/38**: Must update tests that will break:
1. `test_mismatched_tags_length_raises` - keep as-is (correct behavior)
2. Add `MockLLMClient.embed_batch()` method
3. Add new tests for batch embedding

**After implementation**: Refactor magic numbers to test constants (maintenance, not blocking).

---

## Verification

- [ ] Test suite passes with current code
- [ ] Test suite passes after Spec 37 implementation
- [ ] Test suite passes after Spec 38 revision
- [ ] No hard-coded values that duplicate config defaults
