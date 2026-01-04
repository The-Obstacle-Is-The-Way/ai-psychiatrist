# BUG-030: JSON Control Character Sanitization Incomplete - Persistent Parse Failures

**Status**: Open (Critical)
**Severity**: P0 (Inference Failures - Retries Exhausted)
**Filed**: 2026-01-04
**Component**: `src/ai_psychiatrist/infrastructure/llm/responses.py`
**Observed In**: Run 11 (run11_confidence_suite_20260103_215102.log)

---

## Summary

Despite implementing control character sanitization in Specs 058-059, JSON parsing is still failing with:
- "Invalid control character" errors
- "Invalid \escape" errors
- "unterminated string literal" errors

These failures exhaust retry budgets and cause participants to fail.

---

## Impact

| Metric | Run 10 | Run 11 | Delta |
|--------|--------|--------|-------|
| JSON Parse Failures | 0 | 11 | +11 |
| Retry Exhaustion | 0 | 4 | +4 |
| Consistency Sample Failures | 0 | 2 | +2 |

**Each failure triggers 3-5 retries** → ~40+ wasted LLM calls.

---

## Error Patterns

### Pattern 1: Invalid Control Character (Most Common)

```
json_error='Invalid control character at: line 15 column 90 (char 982)'
python_error='unterminated string literal (detected at line 15)'
text_hash=21333de71c46
```

The **same text_hash appears 4 times** = LLM is deterministically generating malformed JSON that our sanitization doesn't fix.

### Pattern 2: Invalid Escape Sequence

```
json_error='Invalid \\escape: line 9 column 80 (char 467)'
text_hash=ebb432c2d5b6
```

LLM is generating invalid escape sequences like `\x`, `\q`, or malformed `\uXXXX`.

### Pattern 3: Unterminated String

```
python_error='unterminated string literal (detected at line 9)'
```

Unbalanced quotes that `_escape_unescaped_quotes_in_strings()` didn't catch.

---

## Root Cause Analysis

### 1. Control Character Sanitization May Have Ordering Issue

Current order in `tolerant_json_fixups()`:
```python
# 4) Escape unescaped quotes inside JSON strings
escaped_quotes_fixed = _escape_unescaped_quotes_in_strings(fixed)

# 5) Escape control characters in string values
control_chars_fixed = _escape_control_chars_in_strings(fixed)
```

**Problem**: If step 4 incorrectly identifies string boundaries (due to control chars), step 5 may not process the right regions.

### 2. Invalid Escape Sequences Not Handled

The `_escape_control_chars_in_strings()` function handles:
- `\t`, `\n`, `\r` → Converts to `\\t`, `\\n`, `\\r`
- Control chars 0x00-0x1F → Converts to `\\uXXXX`

But it does NOT handle:
- Invalid escapes like `\q`, `\x`, `\1`
- Malformed Unicode escapes like `\uXXX` (3 chars instead of 4)
- Bare backslashes before non-escape chars

### 3. json-repair Not Recovering

```python
# parse_llm_json Step 4
result = json_repair.loads(fixed)
```

The `json-repair` library is failing to recover these cases, suggesting the corruption is severe or in unexpected patterns.

---

## Evidence from Logs

### Deterministic Failure (Same Hash, Multiple Attempts)

```
04:40:13 text_hash=01586c359dad text_length=4079  # Attempt 1
04:42:14 text_hash=01586c359dad text_length=4079  # Attempt 2
04:46:23 text_hash=01586c359dad text_length=4079  # Attempt 3
```

The retry mechanism is calling the LLM again, but getting THE SAME malformed response (because temperature=0.3 is low).

### Retry Exhaustion Leading to Failure

```
05:25:17 [error] Pydantic AI call failed during scoring
  error='Exceeded maximum retries (3) for output validation'
  temperature=0.3
  participant_id=427
```

---

## What Was Supposed to Be Fixed

Commits between Run 10 and Run 11:
```
ea758e8 feat(json): implement Specs 058-059 for robust JSON parsing
b8b7772 fix(json): add control character sanitization to tolerant_json_fixups
```

These were supposed to fix JSON parsing, but failures persist.

---

## Proposed Fixes (Based on 2025-2026 Best Practices)

### Fix 1: Multi-Pass Repair Pipeline (Industry Standard)

**Research Finding**: Best systems use a systematic multi-pass approach.

> "An n8n workflow demonstrates a systematic approach: it starts by trimming whitespace and removing Markdown code fences, then escapes unescaped control characters within strings, fixes invalid backslash escape sequences, removes trailing commas, and attempts to fix unescaped double quotes inside string values." — [n8n Workflow](https://n8n.io/workflows/5146-process-ai-output-to-structured-json-with-robust-json-parser/)

**Implementation** - Reorder `tolerant_json_fixups()`:
```python
def tolerant_json_fixups(text: str) -> str:
    # 1) Strip markdown code fences (NEW)
    fixed = _strip_markdown_fences(text)

    # 2) Smart quotes → ASCII quotes
    fixed = _replace_smart_quotes(fixed)

    # 3) Remove zero-width spaces
    fixed = _remove_zero_width(fixed)

    # 4) Control chars FIRST (before quote escaping)
    fixed = _escape_control_chars_in_strings(fixed)

    # 5) Fix invalid escape sequences (NEW)
    fixed = _fix_invalid_escapes(fixed)

    # 6) THEN escape unescaped quotes
    fixed = _escape_unescaped_quotes_in_strings(fixed)

    # 7) Missing commas
    fixed = _insert_missing_commas(fixed)

    # 8) Join stray fragments
    fixed = _join_stray_string_fragments(fixed)

    # 9) Trailing commas
    fixed = _remove_trailing_commas(fixed)

    return fixed
```

### Fix 2: Add Invalid Escape Sequence Handler

**Research Finding**: Invalid backslash escapes are a common LLM issue.

> "The parser applies contextual logic by examining characters following the quote to infer whether it's part of the string's content or a delimiter. If the quote is not followed by a delimiter, the parser prepends an escape character." — [TD Commons 2025](https://www.tdcommons.org/cgi/viewcontent.cgi?article=9955&context=dpubs_series)

```python
def _fix_invalid_escapes(text: str) -> str:
    """Fix invalid escape sequences in JSON strings.

    Valid JSON escapes: " \\ / b f n r t uXXXX
    Invalid escapes like \\q, \\x, \\1 → double the backslash
    """
    # Match backslash followed by invalid escape char
    # (not: " \ / b f n r t u)
    invalid_escape_re = re.compile(r'\\([^"\\/bfnrtu])')

    # Replace \x with \\x (escape the backslash)
    return invalid_escape_re.sub(r'\\\\\\1', text)
```

### Fix 3: Use `json_repair` with Raw String Mode

**Research Finding**: json_repair handles most cases with proper escaping.

> "For escape character issues, you can pass strings as raw strings like: `r\"string with escaping\\\"\"` to handle escaping properly." — [json_repair docs](https://github.com/mangiucugna/json_repair)

```python
# In parse_llm_json, before calling json_repair:
try:
    # Try with strict mode first to catch structural issues
    result = json_repair.loads(fixed, strict=True)
except ValueError as strict_error:
    # Fall back to lenient mode
    result = json_repair.loads(fixed, strict=False)
```

### Fix 4: Add Deterministic Retry Detection

**Problem**: Same broken output triggers multiple retries (waste).

```python
# Track failed text hashes
_failed_hashes: set[str] = set()

def parse_llm_json(text: str) -> dict[str, Any]:
    text_hash = _stable_text_hash(text)

    if text_hash in _failed_hashes:
        logger.warning(
            "Deterministic failure - skipping retry",
            text_hash=text_hash,
            recommendation="Increase temperature or modify prompt",
        )
        raise json.JSONDecodeError("Previously failed hash", text, 0)

    # ... existing logic ...

    # On failure, record hash
    _failed_hashes.add(text_hash)
    raise ...
```

### Fix 5: Add Raw Text Logging for Debugging

```python
# In parse_llm_json, on failure:
logger.error(
    "JSON parse failure - context for debugging",
    text_hash=_stable_text_hash(text),
    error_position=json_error.pos,
    char_at_error=repr(text[json_error.pos]) if json_error.pos < len(text) else "EOF",
    context_before=text[max(0, json_error.pos-50):json_error.pos],
    context_after=text[json_error.pos:json_error.pos+50],
    fixups_applied=applied_fixes,
)
```

---

## Observability Gaps

### Current Gaps

1. **No raw text logged**: Can't see WHAT the malformed JSON looks like
2. **No character position context**: Just "line X column Y" but no surrounding text
3. **No fix attempt logging**: Don't know WHICH fixup was applied before failure
4. **Deterministic retry waste**: Same prompt → same broken output → wasted retries

### Proposed Telemetry

```python
record_telemetry(
    TelemetryCategory.JSON_PARSE_FAILURE,
    text_hash=hash,
    error_type="control_char|escape|unterminated",
    error_position=982,
    text_sample=text[970:1000],  # Around error
    fixups_applied=["smart_quotes", "control_chars"],
    retry_attempt=2,
)
```

---

## Immediate Mitigation

If fixes take time, consider:

1. **Disable consistency sampling** temporarily:
   ```bash
   CONSISTENCY_ENABLED=false
   ```
   (Most failures are during consistency sampling at temp 0.3)

2. **Increase retry budget**:
   ```bash
   PYDANTIC_AI_RETRIES=10
   ```

3. **Log raw failures for analysis**:
   Add debug logging to capture actual malformed JSON

---

## Decision Points for Senior Review

- [ ] **Reorder sanitization steps** (control chars before quotes) — RECOMMENDED
- [ ] **Add invalid escape handler** — RECOMMENDED
- [ ] **Add deterministic retry detection** — Avoid wasted LLM calls
- [ ] **Add raw text logging** for debugging (with privacy controls)
- [ ] **Add telemetry** for specific failure modes
- [ ] **Consider temperature 0.0** for retries (avoid deterministic failures)
- [ ] **Use json_repair strict mode first** (catch structural issues early)

---

## References

### Internal
- Specs 058-059: JSON Parsing Robustness
- Run 11 Log: `data/outputs/run11_confidence_suite_20260103_215102.log`
- Code: `src/ai_psychiatrist/infrastructure/llm/responses.py`
- Commits: ea758e8, b8b7772

### 2025-2026 Research
1. [json_repair GitHub](https://github.com/mangiucugna/json_repair) - Python library for LLM JSON repair (v0.39.1+)
2. [Jaison npm](https://www.npmjs.com/package/jaison) - 100% success rate on 250K malformed JSON tests
3. [n8n JSON Repair Workflow](https://n8n.io/workflows/5146-process-ai-output-to-structured-json-with-robust-json-parser/) - Multi-step repair pipeline
4. [TD Commons: Relaxed JSON Parsing](https://www.tdcommons.org/cgi/viewcontent.cgi?article=9955&context=dpubs_series) - Delimiter-based repair method
5. [Goose Issue #2892](https://github.com/block/goose/issues/2892) - LLM control character issue documentation
6. [Medium: Handling Malformed JSON](https://medium.com/@sd24chakraborty/handling-and-fixing-malformed-json-in-llm-generated-responses-f6907d1d1aa7) - Common patterns
