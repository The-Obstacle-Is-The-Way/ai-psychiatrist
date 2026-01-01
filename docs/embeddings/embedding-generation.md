# Embedding Generation (Fail-Fast, Tags, and Partial Mode)

**Audience**: Researchers generating few-shot reference artifacts
**Last Updated**: 2026-01-01

This guide describes how to generate embedding artifacts safely and reproducibly.

SSOT implementation:
- `scripts/generate_embeddings.py`

---

## What This Produces

Given an output basename `{name}`, embedding generation produces:

- `{name}.npz` (vectors; per-participant keys like `emb_303`)
- `{name}.json` (chunk texts; participant id → list[str])
- `{name}.meta.json` (provenance metadata for fail-fast mismatch detection)

Optional:
- `{name}.tags.json` (PHQ-8 item tags; only if `--write-item-tags`)

---

## Strict Mode (Default) — Recommended

Strict mode is fail-fast:
- transcript load failures → crash
- empty transcript → crash
- embedding failures → crash

Run:

```bash
uv run python scripts/generate_embeddings.py --split paper-train
```

---

## Partial Mode (Debug Only)

Partial mode is opt-in:

```bash
uv run python scripts/generate_embeddings.py --split paper-train --allow-partial
```

Behavior:
- skips failed participants/chunks
- exits with code **2**
- writes a skip manifest `{output}.partial.json` if any skips occur

Manifest schema:

```json
{
  "output_npz": "data/embeddings/....npz",
  "skipped_participants": [487],
  "skipped_participant_count": 1,
  "skipped_chunks": 12
}
```

**Rule**: any artifact produced in partial mode with skips is **not valid** for final evaluation.

---

## Generating Item Tags (Spec 34)

Generate tags alongside embeddings:

```bash
uv run python scripts/generate_embeddings.py --split paper-train --write-item-tags --tagger keyword
```

This writes `{name}.tags.json`.

See: `docs/embeddings/item-tagging-setup.md`.

---

## Verification Checklist (Quick)

After generation:

```bash
ls -la data/embeddings/{name}.npz data/embeddings/{name}.json data/embeddings/{name}.meta.json
```

If using item tags:

```bash
ls -la data/embeddings/{name}.tags.json
```

---

## Related Docs

- Few-shot preflight: `docs/preflight-checklist/preflight-checklist-few-shot.md`
- Feature index: `docs/pipeline-internals/features.md`
