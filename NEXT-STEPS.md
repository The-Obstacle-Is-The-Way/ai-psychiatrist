# Next Steps

**Status**: NEEDS CLARIFICATION
**Last Updated**: 2026-01-02

---

## Open Question: Post-Ablation Default Reconciliation

**The confusion**: NEXT-STEPS.md previously said we should "flip" these defaults in `config.py`:

| Setting | Current Code Default | Suggested Target |
|---------|---------------------|------------------|
| `EMBEDDING_REFERENCE_SCORE_SOURCE` | `participant` | `chunk` |
| `EMBEDDING_ENABLE_ITEM_TAG_FILTER` | `false` | `true` |
| `EMBEDDING_MIN_REFERENCE_SIMILARITY` | `0.0` | `0.3` |
| `EMBEDDING_MAX_REFERENCE_CHARS_PER_ITEM` | `0` | `500` |
| `EMBEDDING_ENABLE_REFERENCE_VALIDATION` | `false` | `true` |

**But**: `.env.example` already recommends chunk scoring and participant-only transcripts.

**Questions to resolve**:
1. Are the code defaults intentionally conservative (paper-parity) while `.env.example` provides the "recommended production" config?
2. Or should we actually flip the code defaults now that ablations are done?
3. Where is this decision documented? (The referenced `POST-ABLATION-DEFAULTS.md` never existed)

**Action**: Clarify the intended relationship between code defaults and `.env.example` recommendations, then either:
- Document that code defaults = paper-parity, `.env.example` = recommended (current state), OR
- Flip the code defaults if that was the intent

---

*This file can be deleted once the above is resolved.*
