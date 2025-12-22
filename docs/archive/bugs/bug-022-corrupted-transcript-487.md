# BUG-022: Corrupted Transcript File for Participant 487

**Status**: Resolved ✓
**Severity**: Low
**Component**: Data / DAIC-WOZ Dataset
**GitHub Issue**: [#33](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/33)
**Discovered**: 2025-12-21 (Embedding Generation)
**Resolved**: 2025-12-21 (Re-download and re-extraction)

---

## Summary

Participant 487's transcript file is corrupted and cannot be parsed. The file is an AppleDouble encoded Macintosh metadata file rather than a valid CSV transcript.

---

## Error Message

```
[error] Failed to parse transcript
error="'utf-8' codec can't decode byte 0xb0 in position 37: invalid start byte"
participant_id=487
```

---

## File Analysis

```bash
$ file data/transcripts/487_P/487_TRANSCRIPT.csv
AppleDouble encoded Macintosh file

$ ls -la data/transcripts/487_P/
-rw-r--r--  4096 bytes  487_TRANSCRIPT.csv
```

**Key indicators of corruption:**
- File size is only 4KB (valid transcripts are 10-30KB)
- File type is AppleDouble metadata, not CSV
- Contains macOS extended attributes (com.apple.quarantine, Microsoft Excel metadata)
- No actual transcript content present

---

## Impact Assessment

| Metric | Value | Notes |
|--------|-------|-------|
| **Embedding Coverage** | 106/107 (99%) | Negligible impact |
| **Train Split Impact** | 1 participant missing | 487 is in train split |
| **Few-Shot Quality** | Minimal | 106 references still available |
| **Paper Reproducibility** | Unknown | Paper doesn't mention 487 specifically |

---

## Root Cause Analysis

AppleDouble files are created by macOS to store extended attributes and resource forks. This corruption likely occurred when:

1. **ZIP extraction issue**: The DAIC-WOZ dataset was extracted on macOS with improper handling of resource forks
2. **File replacement**: The actual CSV was replaced by its `._` AppleDouble sibling during copy/extraction
3. **Source corruption**: The original download may have been corrupted

---

## Investigation Checklist

- [ ] Check if `._487_TRANSCRIPT.csv` hidden file exists with real content
- [ ] Re-download participant 487 from DAIC-WOZ source
- [ ] Verify if issue exists in original DAIC-WOZ distribution
- [ ] Check if other users report this issue (DAIC-WOZ forums/papers)

---

## Potential Fixes

### Option A: Re-download from Source
```bash
# Re-request specific participant from USC ICT
# Requires EULA agreement
```

### Option B: Extract from AppleDouble (if data exists)
```bash
# AppleDouble format stores data in specific structure
# The actual content MAY be recoverable if present
python3 -c "
import struct
with open('data/transcripts/487_P/487_TRANSCRIPT.csv', 'rb') as f:
    data = f.read()
    # AppleDouble magic: 0x00051607
    # Data fork offset at bytes 24-27
    # Check if actual CSV data exists after header
    print(f'File size: {len(data)} bytes')
    print(f'Magic: {data[:4].hex()}')
"
```

### Option C: Mark as Unavailable
- Document in `docs/data/daic-woz-schema.md` that participant 487 has known data issues
- Update validation checklist to skip 487

---

## References

- [DAIC-WOZ Dataset](https://dcapswoz.ict.usc.edu/)
- [AppleDouble Format](https://en.wikipedia.org/wiki/AppleSingle_and_AppleDouble_formats)
- Paper Section 2.1: DAIC-WOZ dataset description (no mention of 487)

---

## Resolution

**Root Cause Confirmed**: AppleDouble metadata file replaced actual CSV during initial ZIP extraction.

**Fix Applied** (2025-12-21):
1. Re-downloaded `487_P.zip` from DAIC-WOZ source
2. Verified new ZIP contains valid 20KB transcript (not 4KB AppleDouble)
3. Extracted only `487_TRANSCRIPT.csv` (skipping `._487_TRANSCRIPT.csv`)
4. Replaced corrupted file in `data/transcripts/487_P/`
5. Generated embeddings for participant 487 (147 chunks × 4096 dimensions)
6. Verified all 107 train participants now have embeddings (100% coverage)

**Prevention**: When extracting DAIC-WOZ ZIPs on macOS, use `unzip -x '._*'` to exclude AppleDouble files.

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-12-21 | Document issue | Discovered during embedding generation |
| 2025-12-21 | Created GitHub #33 | Track for future resolution |
| 2025-12-21 | Re-download and replace | Confirmed extraction corruption, not source issue |
| 2025-12-21 | Generate 487 embeddings | Incremental update to existing NPZ+JSON |
