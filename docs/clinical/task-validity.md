# Task Validity: What Can (and Cannot) Be Inferred from DAIC-WOZ Transcripts

**Audience**: Researchers, reviewers, and anyone interpreting results
**Last Updated**: 2026-01-05

This repository evaluates PHQ-8 **item-level scores (0–3)** from **clinical interview transcripts** (DAIC-WOZ). That is an unusually hard target: PHQ-8 is a *self-report frequency* instrument (“over the past 2 weeks, how often…”), while DAIC-WOZ interviews are not structured as PHQ administration.

This page defines the **valid scientific claims** the codebase supports and the **limitations** reviewers should assume unless proven otherwise by ablations.

---

## What PHQ-8 Actually Measures

PHQ-8 items are scored 0–3 based on **frequency over the past two weeks** (0–1, 2–6, 7–11, 12–14 days). This is a self-report questionnaire, not a clinician-rated interview rubric.

References:
- PHQ-8 validation / description (general population): https://pubmed.ncbi.nlm.nih.gov/18752852/

---

## What DAIC-WOZ Transcripts Actually Contain

DAIC-WOZ interviews are semi-structured conversations intended to elicit **verbal and nonverbal indicators** correlated with depression, not necessarily explicit “days per 2-week window” frequency statements.

References:
- DAIC-WOZ project + documentation: https://dcapswoz.ict.usc.edu/
- DAIC-WOZ documentation PDF: https://dcapswoz.ict.usc.edu/wp-content/uploads/2022/02/DAICWOZDepression_Documentation.pdf

---

## The Core Validity Threat (Construct Mismatch)

**Transcript-only item-level PHQ-8 frequency scoring is often underdetermined.**

In many interviews:
- Participants mention symptoms without quantifying 2-week frequency.
- Short answers become ambiguous without question context (especially in participant-only transcripts).
- The model must choose between:
  1. **Abstention** (`N/A`) when evidence is insufficient (methodologically conservative), or
  2. **Inference** of frequency from vague temporal language (“lately”, “sometimes”) (higher coverage, higher subjectivity).

This is not primarily a “model capability” issue; it is an **information availability** issue.

---

## What This Repo Currently Claims (and Why)

### 1) Selective, evidence-grounded item scoring

The quantitative agent returns per-item scores **or `N/A`** when it cannot justify a score from transcript evidence.

Consequences:
- Coverage is a first-class metric: **Cmax** and risk-coverage metrics (AURC/AUGRC) must be reported alongside MAE.
- “Low coverage” is not a pipeline failure; it is expected behavior when transcripts lack item-level evidence.

### 2) Few-shot vs zero-shot is an empirical question

Retrieval can only help if:
- enough grounded evidence exists to embed queries, and
- retrieved references are both symptom-relevant *and* severity-informative.

When evidence is sparse, few-shot can easily become “no-op” (few references retrieved) or add noise (misleading anchors).

---

## What Reviewers May Ask For (Recommended Ablations)

To defend item-level claims, expect to provide:
- **Transcript variant ablations**: `participant_only` vs `participant_qa` (question context) vs “both speakers”.
- **Prompt policy ablations**: strict “frequency required” vs “allow explicit inference with traceability”.
- **Retrieval controls**: identical prompt wrapper for both modes when no references exist (to remove prompt confounds).
- **Ordinal metrics**: e.g., weighted κ alongside MAE (ordinal scale).

---

## Alternative Task Definitions That May Be More Valid on DAIC-WOZ

If the research goal is not strictly “PHQ-8 item frequency”, these are often more defensible:
- **Binary depression classification** (e.g., PHQ-8 total ≥ 10).
- **Total PHQ-8 score regression** (0–24).
- **Severity bucket prediction** (minimal/mild/moderate/mod-severe/severe).
- **Symptom presence** per item (binary), instead of frequency (0–3).

These are still non-trivial, but they reduce the frequency-specific mismatch.

---

## Prior Art (Sanity Check)

Transcript-only prediction of PHQ-8 items/totals on DAIC-WOZ exists, but reported performance is imperfect and task framing varies. Use these as context, not as guarantees.

- LLMs for DAIC-WOZ + PHQ-8 items: https://pubmed.ncbi.nlm.nih.gov/40720397/
- Text-only PHQ-8 total score regression on DAIC-WOZ: https://pubmed.ncbi.nlm.nih.gov/37398577/

---

## Ground Truth Reliability (Upper Bound Context)

PHQ-8 is not noise-free; self-report has known variability. Some psychometric studies report test-retest reliability for PHQ-8 total score around ICC ≈ 0.83 (population-dependent).

- Example (Swedish PHQ-8; ICC): https://pubmed.ncbi.nlm.nih.gov/32661929/
