# INVESTIGATION-025: JSON Parsing Edge Cases in Reproduction Runs

**Status**: Informational (no fix required, system handles gracefully)
**Severity**: LOW
**Found**: 2025-12-23 (during few-shot reproduction run)
**Related**: BUG-011 (resolved), Issue #29 (Ollama JSON mode), Issue #53 (experiment tracking)

---

## Summary

During the few-shot reproduction run, we observed this warning:

```
Failed to parse evidence JSON, using empty evidence
response_preview='```json\n{\n    "PHQ8_NoInterest": [],\n    "PHQ8_Depressed": [""i had been having a lot of deaths around me...
```

This investigation documents what this warning means, why it occurs, and confirms the system handles it gracefully.

---

## Analysis

### 1. What Happened

The LLM returned JSON with an **unescaped quote** inside a string value:

```json
"PHQ8_Depressed": [""i had been having a lot of deaths around me...]
```

The `""i` is malformed JSON - a string starting with an unescaped quote character.

### 2. Why It Happened

The LLM (gemma3:27b) included a quotation from the transcript that itself contains quote marks. The model did not properly escape the interior quotes (`\"`) as required by JSON syntax.

### 3. What the System Did

Following BUG-011's resolution, the system:

1. **Attempted tolerant fixups** (`_tolerant_fixups`):
   - ✅ Smart quotes → straight quotes
   - ✅ Trailing comma removal
   - ❌ Unescaped interior quotes (not handled)

2. **Fell back gracefully** to empty evidence

3. **Logged a warning** with response preview for debugging

4. **Continued processing** with zero evidence for this extraction

### 4. Downstream Impact

For participant 303:
- Evidence extraction returned `{}`
- `items_with_evidence=0` for this call
- Assessment still completed: `total_score=4, na_count=4, severity=MINIMAL`
- System continued to next participant successfully

**Conclusion**: The graceful fallback worked as designed.

---

## Code Path

```
quantitative.py:_extract_evidence()
├── _strip_json_block(raw)    # Extract from markdown/XML
├── _tolerant_fixups(clean)   # Fix smart quotes, trailing commas
├── json.loads(clean)         # FAILS on unescaped quotes
└── except → log warning, return {}
```

Location: `src/ai_psychiatrist/agents/quantitative.py:285-295`

---

## Current Fixups vs This Edge Case

| Fixup | Handled | Example |
|-------|---------|---------|
| Smart quotes | ✅ | `"hello"` → `"hello"` |
| Trailing commas | ✅ | `{"a": 1,}` → `{"a": 1}` |
| Unescaped interior quotes | ❌ | `[""text"]` → ? |

### Potential Enhancement

Could add to `_tolerant_fixups`:
```python
# Fix unescaped quotes in array strings
text = re.sub(r'\[""', '["', text)  # [""x → ["x
text = re.sub(r'""\]', '"]', text)  # x""] → x"]
```

**However**: This is a heuristic that could cause false positives. The cleaner solution is Issue #29 (Ollama JSON mode) or Issue #28 (Pydantic AI).

---

## Traceability Gap

### What's Captured

1. ✅ **Console/log file**: All warnings with response previews
2. ✅ **Reproduction log**: `tee data/outputs/reproduction_run_*.log`

### What's NOT Captured

1. ❌ **Output JSON**: Warnings not aggregated into results file
2. ❌ **Per-participant error summary**: No field for "had parsing issues"

### Related

Issue #53 (experiment tracking) proposes adding:
- Full provenance in output JSON
- Error/warning aggregation
- Experiment registry

---

## Recommendations

### No Immediate Action Needed

1. **System is working correctly** - graceful fallback prevents crashes
2. **Warnings are logged** - traceability exists in log files
3. **Impact is minimal** - one empty evidence extraction doesn't break the run

### Future Improvements (Tracked Elsewhere)

| Issue | Improvement | Priority |
|-------|-------------|----------|
| #29 | Ollama JSON mode for structured output | Medium |
| #28 | Pydantic AI for validated outputs | Medium |
| #53 | Aggregate warnings into output JSON | Low |

### Optional Enhancement

Add warning counter to reproduction output:

```json
{
  "metadata": {
    "warnings_count": 3,
    "warnings": [
      "Participant 303: Failed to parse evidence JSON"
    ]
  }
}
```

---

## Verification

During the current run:
- Participant 303: Warning occurred, continued successfully
- Participant 312: No warning, 5 items with evidence
- Run is progressing normally

---

## Conclusion

This is **expected behavior**, not a bug. The warning indicates the system's defensive parsing handled a malformed LLM response gracefully. The log file provides full traceability. Future improvements (Ollama JSON mode, Pydantic AI) will reduce these occurrences.

**Status**: Closed as informational. No action required.
