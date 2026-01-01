# Spec 043: Deterministic JSON “Missing Comma” Repair (BUG-043)

**Status**: Ready to implement
**Bug**: `docs/_bugs/bug-043-json-missing-comma-repair.md`
**Primary affected code**: `src/ai_psychiatrist/agents/extractors.py`, `src/ai_psychiatrist/agents/quantitative.py`
**Related utilities**: `src/ai_psychiatrist/infrastructure/llm/responses.py`

## 0. Problem Statement

Some LLM responses (observed with Gemma 3 27B at `temperature=0`) return **almost-correct** JSON that is invalid due to **missing commas between object members**, typically formatted like:

```text
"value"
"next_key":
```

Because temperature is zero, Pydantic AI retries can be **deterministically identical**, so the same malformed JSON repeats until retries are exhausted.

## 1. Goals / Non-Goals

### 1.1 Goals

- Make structured-output parsing resilient to the “missing comma between object entries” failure mode.
- Keep repairs **deterministic**, low-latency, and dependency-free.
- Avoid altering valid JSON (repairs must be no-ops on already-valid JSON).
- Reduce duplicated “JSON fixup” logic across agents by centralizing it.
- Add unit tests covering the repaired patterns to prevent regressions.

### 1.2 Non-Goals

- Implement a general “JSON5” parser or accept non-JSON syntax broadly.
- Make additional LLM repair calls by default (can be a follow-on feature due to runtime cost).
- Fix semantic/model errors (only syntactic JSON repair is in scope).

## 2. Requirements (Normative)

### 2.1 Repair must be conservative

The repair step **must only** insert commas in a narrowly defined pattern that indicates an object-member boundary:

- There is a JSON *value terminator* immediately followed by whitespace + a newline + whitespace + a JSON object key (`"..."` followed by `:`).
- The value terminator is one of:
  - a closing quote `"` (end of a JSON string)
  - a digit `0-9` (end of a JSON number)
  - the literal tokens `true`, `false`, or `null`
  - optionally, `}` or `]` (end of an object/array value)

The repair must **not** attempt to add commas inside strings or for non-object boundaries (e.g., arrays), unless explicitly added later under a separate spec.

### 2.2 Repairs must preserve validity + be idempotent

- If the JSON is already valid, applying fixups must not change the string.
- Applying fixups multiple times must yield the same output (idempotent).

### 2.3 Repairs must be shared

There must be one canonical tolerant-fixup function used by:

- `src/ai_psychiatrist/agents/extractors.py` (structured outputs: quantitative/judge/meta-review JSON)
- `src/ai_psychiatrist/agents/quantitative.py` (evidence extraction JSON parsing)
- `src/ai_psychiatrist/infrastructure/llm/responses.py` (general JSON extraction utilities)

This avoids drift where one path repairs a class of errors and another does not.

## 3. Proposed Design

### 3.1 Create a single public fixup function

Add a public utility in `src/ai_psychiatrist/infrastructure/llm/responses.py`:

```python
def tolerant_json_fixups(text: str) -> str:
    ...
```

Behavior (in order):

1) Replace smart quotes (`\u201c\u201d\u2018\u2019`) with ASCII quotes
2) Remove zero-width spaces (`\u200b`)
3) Insert missing commas between object entries (Section 3.2)
4) Remove trailing commas before `}` or `]`

Then update:

- `extract_json_from_response()` to call `tolerant_json_fixups()` (instead of `_normalize_json_text()`)
- `src/ai_psychiatrist/agents/extractors.py` to call `tolerant_json_fixups()` (remove or deprecate its local `_tolerant_fixups`)
- `src/ai_psychiatrist/agents/quantitative.py` to call `tolerant_json_fixups()` (remove or deprecate its method copy)

### 3.2 Missing comma insertion rule (exact patterns)

Apply **one or two** narrow regex substitutions.

#### Pattern A: end-of-primitive/string followed by a new key

Insert a comma between:

```text
... "value"\n  "key":
... 123\n  "key":
... true\n  "key":
... null\n  "key":
```

Regex (Python `re`, with multiline input):

```python
text = re.sub(
    r'("|\d|true|false|null)\s*\n\s*"([^"]+)"\s*:',
    r'\1,\n"\2":',
    text,
)
```

**Note on `\d` vs `\d+`**: The pattern uses `\d` (single digit) intentionally. For a number like `123`, only the final `3` needs to match to anchor the repair. The preceding digits are not consumed or altered. Using `\d+` would work but is unnecessary.

#### Pattern B (recommended): end-of-object/array followed by a new key

Insert a comma between:

```text
... }\n  "key":
... ]\n  "key":
```

Regex:

```python
text = re.sub(
    r'([}\]])\s*\n\s*"([^"]+)"\s*:',
    r'\1,\n"\2":',
    text,
)
```

Rationale:

- Pattern A fixes the observed failure mode in BUG-043.
- Pattern B covers the common nested-object case where the missing comma follows a `}` or `]`.

**Edge case: patterns inside string values**

These regexes could theoretically match content inside a JSON string value (e.g., `"description": "ends with 3\n\"next\": value"`). However:

1. LLM-generated JSON rarely contains embedded JSON-like patterns in string values.
2. The newline anchor (`\n`) makes false positives unlikely in practice.
3. If this becomes an issue, a more sophisticated parser-based approach would be needed (out of scope for this spec).

The conservative approach is acceptable given the observed failure mode.

### 3.3 Observability

When a repair modifies text, log a structured debug message (without including the full JSON):

- component: `json_fixups`
- applied_fixes: e.g. `["missing_commas", "trailing_commas"]`
- preview hashes or lengths only (avoid logging PHI-like content)

This should be low-noise (debug-level), but available for investigations.

## 4. Test Plan

Add deterministic unit tests (no LLM required).

### 4.1 Unit tests for `tolerant_json_fixups()`

Add tests that validate:

1) **Fixes BUG-043 pattern**:
   - Input with missing comma between two object members separated by newline parses successfully after fixups.
2) **Fixes nested-object missing comma** (Pattern B):
   - Input where a nested object closes with `}` then newline then next key parses after fixups.
3) **No-op on valid JSON**:
   - Valid JSON input remains identical after fixups.
4) **Idempotence**:
   - `fixups(fixups(x)) == fixups(x)` for representative broken inputs.

### 4.2 Integration verification (optional but recommended)

Run the existing reproduction that previously failed (participant 339, zero-shot mode) and confirm it succeeds without changing temperature or retry counts.

## 5. Acceptance Criteria

- BUG-043 reproduction succeeds (participant 339 no longer fails due to JSON decode errors).
- Unit tests pass and cover the missing-comma repair logic.
- No new failures appear in structured-output parsing paths (quantitative/judge/meta-review).
- `uv run mkdocs build --strict` passes.

## 6. Rollout Notes / Risk

- This fix is intentionally conservative and should not alter valid JSON.
- The regex is anchored to a newline boundary and a `"key":` pattern, which minimizes false positives.
- If future bugs show missing commas without newlines, handle them in a follow-on spec with a separately justified pattern (higher false-positive risk).
