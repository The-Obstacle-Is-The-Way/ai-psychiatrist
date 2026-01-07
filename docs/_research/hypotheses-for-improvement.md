# Hypotheses for Improvement: First-Principles Analysis

**Status**: Research findings document
**Created**: 2026-01-05
**Analysis Scope**: Full pipeline from DAIC-WOZ → Evidence Extraction → Embedding → Scoring

---

## Executive Summary

A first-principles audit of the PHQ-8 scoring pipeline reveals several fundamental mismatches between the dataset, the task, and our implementation. These are not bugs in the traditional sense—the code executes correctly—but rather **methodological constraints** that limit what is achievable with this approach on this dataset.

**Key Finding**: DAIC-WOZ was designed to capture behavioral indicators of depression, not to elicit explicit PHQ-8 *frequency* information. Our quantitative prompts are (correctly) conservative about scoring without frequency evidence, but the dataset often does not provide it.

> **Task Validity SSOT**: `docs/clinical/task-validity.md` — comprehensive analysis of construct mismatch and valid scientific claims.

**Run 13 SSOT snapshot** (clean post-BUG-035 comparative baseline; 41 participants processed in both modes):
- **Zero-shot**: item MAE = **0.6079**, coverage = **50.0%** (40/41 evaluated; 1 excluded: no evidence)
- **Few-shot**: item MAE = **0.6571**, coverage = **48.5%** (41/41 evaluated)
- **Key result**: zero-shot beats few-shot after the BUG-035 fix, so the gap is not a prompt confound artifact.

**Run 12 pipeline stats snapshot** (pre-BUG-035; useful for evidence/grounding/retrieval distributions):
- Evidence grounding rejects ~**49.5%** of extracted quotes (deduped across modes).
- Only **32.0%** of item assessments had any grounded LLM evidence (105/328).
- Few-shot references are sparse: **15.2%** of item assessments had any references (50/328), receiving **52 total** references.

Run 13 is documented in `docs/results/run-history.md`. The Run 12 pipeline stats above are derived from Run 12 artifacts in `data/outputs/` and summarized in `docs/results/few-shot-analysis.md`.

---

## Peer-Review “Reject” Threats (Adversarial List)

These are the issues most likely to trigger rejection on *construct validity* / *method validity* grounds unless explicitly addressed via ablations or wording.

### A) Construct validity: PHQ-8 is self-report frequency; transcripts often lack frequency (Major)

- PHQ-8 is explicitly a “past two weeks / frequency” instrument; DAIC-WOZ is not a PHQ interview. Most interview statements are qualitative (no explicit day counts).
- Our prompts correctly push the model to abstain when frequency is unclear (`src/ai_psychiatrist/agents/prompts/quantitative.py:37-45` and `src/ai_psychiatrist/agents/prompts/quantitative.py:111-117`), but that means the system is fundamentally measuring “inferable PHQ evidence from transcript” rather than PHQ itself.

**Implication for claims**: You must frame the task as *selective, evidence-grounded inference* rather than “PHQ-8 from transcripts” in an absolute sense.

### B) Few-shot prompt confound (Fixed; historical runs only) (Major)

Historical runs had a prompt confound: few-shot prompting could still differ from zero-shot even
when retrieval returned **zero** usable references (an empty reference wrapper containing the
string “No valid evidence found”).

This is now fixed (BUG-035): empty reference bundles format to `""` and the `<Reference Examples>`
block is omitted, so **few-shot-with-no-refs is byte-identical to zero-shot**.

**Implication**: pre-fix “few-shot vs zero-shot” comparative claims are confounded and require
post-fix reruns to measure the true retrieval effect.

### C) Participant-only transcripts remove disambiguating question context (Major)

Participant-only transcripts are effective at reducing protocol leakage into embeddings, but they also remove the questions that disambiguate short answers (semantic void problem). This can reduce evidence yield and coverage.

**Mitigation**: Ablate against `transcripts_participant_qa` (minimal question context) and quantify the impact on evidence grounding rate, coverage, and MAE/AUGRC.

### D) Privacy/ethics risk: log artifacts leaking restricted text (Major)

Any workflow that logs raw transcript text, retrieved reference text, or LLM outputs can leak
restricted corpus content into run artifacts.

Current status:
- Retrieval audit logs in `EmbeddingService` are privacy-safe (Spec 064): they emit `chunk_hash` and
  `chunk_chars` (no raw chunk previews).
- Ensure auxiliary scripts follow the same policy (e.g., chunk scoring should avoid logging
  `chunk_preview` / `response_preview`).

---

## 1. Dataset-Task Mismatch (Critical)

### What DAIC-WOZ Was Designed For

Per the [DAIC-WOZ documentation](https://dcapswoz.ict.usc.edu/):

> "These interviews were collected as part of a larger effort to create a computer agent that interviews people and **identifies verbal and nonverbal indicators of mental illness**."

The virtual interviewer "Ellie" conducts semi-structured interviews designed to:
- Create interactional situations favorable to assessing distress indicators
- Capture behavioral markers correlated with depression
- Collect multimodal data (audio, video, text)

### What PHQ-8 Scoring Requires

PHQ-8 is a **frequency-based** instrument asking "Over the last 2 weeks, how often have you been bothered by [symptom]?":
- 0 = Not at all (0-1 days)
- 1 = Several days (2-6 days)
- 2 = More than half the days (7-11 days)
- 3 = Nearly every day (12-14 days)

### The Mismatch

**The interview doesn't ask about frequency. Participants don't state frequency.** They say things like:
- "I've been feeling tired" (no frequency)
- "I have trouble sleeping sometimes" (vague)
- "I've been stressed lately" (qualitative)

This explains why (see Run 12 pipeline stats snapshot above):
- Only **32.0%** of item assessments have any grounded evidence (105/328)
- ~**49.5%** of extracted quotes fail evidence grounding
- Coverage stabilizes around **46–49%** in both modes

---

## 2. Evidence Extraction Paradox

### Current Prompt Logic

Our prompts (see `src/ai_psychiatrist/agents/prompts/quantitative.py:111-117`) say:
```
5. If no relevant evidence exists, mark as "N/A" rather than assuming absence
6. Only assign numeric scores (0-3) when evidence clearly indicates frequency
```

### The Paradox

This is **methodologically correct but practically limiting**:
- Most transcripts don't contain explicit frequency statements
- Correct behavior: output N/A for most items
- Result: ~50% abstention rate

### Hypothesis 2A: Frequency Can Be Inferred

A skilled psychiatrist doesn't require patients to say "I felt tired 8 out of 14 days." They infer frequency from:
- Temporal markers ("lately", "recently", "since [event]")
- Intensity qualifiers ("always", "sometimes", "occasionally")
- Impact statements ("I can't function", "it's been hard")
- Context patterns (multiple mentions across the interview)

**Current Status**: Our prompts discourage inference. They demand explicit frequency.

**Improvement Hypothesis**: Update prompts to allow clinical inference while maintaining transparency:
```
When explicit frequency is not stated, you may infer approximate frequency from:
- Temporal language ("lately" → several days, "always" → nearly every day)
- Intensity markers ("sometimes" → several days)
- Functional impact ("can't work" → more than half the days)
Document your inference in the reason field.
```

**Trade-off**: Higher coverage, potentially lower precision. Needs ablation.

---

## 3. Chunk Scoring Validity Issues

### Observation from Scored Chunks

Examining chunks scored for PHQ8_Sleep (see `data/embeddings/*.chunk_scores.json`):

**Chunk 303:37 scored Sleep=3**:
```
"i need my rest because i'm out there driving that bus..."
"what am i like irritated tired um lazy"
"feel like i wanna lay down probably go to sleep"
```

**Problem**: This participant is expressing:
- Value for rest ("I need my rest")
- Desire to sleep ("feel like i wanna lay down")
- General tiredness

**NOT**: Trouble falling/staying asleep or sleeping too much (the actual PHQ-8 Sleep item)

### Hypothesis 3A: Semantic Confusion in Chunk Scoring

The LLM scorer is confusing:
| What participant said | What LLM inferred | Actual PHQ-8 construct |
|-----------------------|-------------------|------------------------|
| "I need rest" | Sleep problems | Not a symptom |
| "I feel tired" | Sleep issues | Different item (Tired) |
| "I want to nap" | Sleeping too much | Maybe, context-dependent |

**Improvement Hypothesis**: Add explicit symptom definitions to chunk scoring prompt:
```
PHQ8_Sleep asks about: "Trouble falling or staying asleep, OR sleeping too much"
- Wanting rest is NOT a sleep problem
- Feeling tired belongs to PHQ8_Tired, not PHQ8_Sleep
- "Sleeping too much" means actually sleeping excessive hours, not wanting to
```

---

## 4. Embedding Space Limitations

### Current Approach

1. Extract evidence text from test transcript
2. Embed evidence text
3. Find similar chunks from reference corpus
4. Use reference chunk scores as anchors

### Hypothesis 4A: Semantic Similarity ≠ Severity Similarity

Embedding captures **topic similarity**, not **severity similarity**:
- "I can't sleep at night" (severe) ≈ "I value good rest" (not a symptom)
- Both are "about sleep" in embedding space
- One is PHQ8_Sleep=3, one is PHQ8_Sleep=0

**Evidence**: Item-tag filtering helps (Spec 34), but doesn't solve the severity confusion within a topic.

### Hypothesis 4B: Score Reranking

**Improvement Hypothesis**: After semantic retrieval, rerank by:
1. Presence of severity markers in reference chunk
2. Score distribution (prefer balanced exemplars)
3. Exclude chunks that are topic-adjacent but not symptom-indicative

### Hypothesis 4C: Domain mismatch — general embeddings may be suboptimal (Major)

We currently use a general-purpose embedding model (`MODEL_EMBEDDING_MODEL=qwen3-embedding:8b`). Clinical NLP has multiple domain-adapted models (e.g., ClinicalBERT / PubMedBERT) that may better represent symptom language and reduce topical-but-not-clinical matches.

**Improvement Hypothesis**: Add an embeddings ablation suite:
- baseline: current `qwen3-embedding:8b`
- clinical-domain embedding baseline(s): ClinicalBERT / PubMedBERT style encoders (or a modern clinical embedding model)
- evaluate: retrieval sparsity, reference score usefulness, downstream MAE/AUGRC

This must be done as an ablation; do not assume improvements without measurement.

---

## 5. N/A Criteria Analysis

### Current Behavior

Two paths to N/A:
1. `NO_MENTION`: LLM evidence count = 0 (no relevant quotes found)
2. `SCORE_NA_WITH_EVIDENCE`: LLM found evidence but explicitly said N/A

### Hypothesis 5A: Over-Abstention on Implicit Evidence

Run 12 data: 51.5% abstention (zero-shot), 54% abstention (few-shot).

Many participants may have depression symptoms visible in their language patterns (word choice, response length, topic avoidance) without explicit symptom mentions.

**Question**: Should we abstain on items where behavioral indicators suggest pathology but explicit frequency is missing?

**Trade-off**:
- Abstaining is methodologically conservative (no hallucinated scores)
- But may miss clinically meaningful signals
- Psychiatrists use holistic assessment, not just verbal frequency statements

---

## 6. Frequency Inference Hierarchy

### Proposed Inference Rules (Hypothesis)

| Language Pattern | Inferred Frequency | PHQ-8 Score |
|------------------|-------------------|-------------|
| "every day", "constantly", "all the time" | 12-14 days | 3 |
| "most days", "usually" | 7-11 days | 2 |
| "sometimes", "a few times", "lately" | 2-6 days | 1 |
| "once", "rarely", "not really" | 0-1 days | 0 |
| No temporal marker, only symptom mention | Ambiguous | N/A or 1? |

**Current behavior**: Ambiguous → N/A
**Alternative**: Ambiguous → 1 (conservative non-zero) with low confidence

---

## 7. Pipeline Architecture Questions

### Question 7A: Evidence Extraction as Bottleneck

The current pipeline:
```
Transcript → Evidence Extraction → Embedding → Reference Retrieval → Scoring
```

Evidence extraction is a **filter**:
- Grounded quotes only (substring match)
- Rejects ~50% of extracted quotes as "hallucinated"

**Hypothesis**: Evidence grounding is too strict. "Hallucinated" quotes may be:
- Paraphrases (valid signal, wrong words)
- Composite statements (synthesized from multiple utterances)
- Reasonable inferences (not literal but implied)

**Improvement Hypothesis**: Fuzzy grounding with semantic similarity instead of substring match.

### Hypothesis 7C: Evidence extractor prompt may be inducing quote “hallucinations” (Major)

The evidence extraction prompt currently asks the model to both (a) extract quotes and (b) “determine the appropriate PHQ-8 score”, but the response schema is quote arrays only (`src/ai_psychiatrist/agents/prompts/quantitative.py:47-89`). This mixed objective can incentivize the model to synthesize/normalize quotes rather than copy verbatim.

**Improvement Hypothesis**: Rewrite evidence extraction as a pure “verbatim quote finder”:
- Remove any instruction about scoring in the evidence step.
- Add stronger constraints: “copy exact substrings; do not paraphrase; do not merge lines.”
- Evaluate impact on grounding rejection rate and few-shot reference coverage.

### Question 7B: Direct Scoring vs. Evidence-Mediated

Alternative architecture:
```
Transcript → Direct Scoring (no evidence extraction)
```

Let the LLM see the full transcript and score directly. Trade-offs:
- Pro: No evidence extraction bottleneck
- Con: Less interpretable, harder to ground
- Con: May increase hallucination

---

## 8. Ground Truth Reliability

### The Meta-Question

How reliable is the PHQ-8 ground truth?
- Patients self-report their symptoms
- Self-report has known biases (social desirability, recall error)
- The same patient might score differently on different days

**Implication**: Even perfect prediction can't exceed ground truth reliability. MAE floor may be ~0.5 due to label noise, not model error.

**Evidence** (examples of PHQ-8 reliability in the literature):
- Swedish PHQ-8 psychometrics report **test-retest ICC ≈ 0.83** for total score and Cronbach’s α ≈ 0.85 (Rheumatol Int, 2020; PubMed: 32661929).
- Another PHQ-8 psychometric study reports Cronbach’s α ≈ 0.922 (Hum Reprod Open, 2022; PubMed: 35591921).

These are not DAIC-WOZ-specific, but they provide an empirical anchor: the label is not noise-free, and extremely low MAE targets may be unrealistic without additional modalities or repeated measures.

---

## 9. Summary of Hypotheses

| ID | Hypothesis | Type | Effort | Status |
|----|-----------|------|--------|--------|
| 2A | Allow frequency inference from temporal/intensity markers | Prompt change | Low | **→ Spec 063** |
| 3A | Add explicit symptom definitions to chunk scorer | Prompt change | Low | Proposed |
| 4A | Embedding captures topic, not severity | Architecture | High | Research |
| 4B | Rerank by severity markers, not just similarity | Code change | Medium | Proposed |
| 5A | Consider behavioral indicators beyond verbal frequency | Research | High | Research |
| 7A | Fuzzy evidence grounding (semantic similarity) | Config + code | Medium | Proposed |
| 7B | Direct scoring without evidence extraction | Architecture | High | Research |

**Related Specs** (address task validity):
- **Spec 061**: Total PHQ-8 Score Prediction (0-24) — `docs/_specs/spec-061-total-phq8-score-prediction.md`
- **Spec 062**: Binary Depression Classification — `docs/_specs/spec-062-binary-depression-classification.md`
- **Spec 063**: Severity Inference Prompt Policy (implements Hypothesis 2A) — `docs/_specs/spec-063-severity-inference-prompt-policy.md`

---

## 10. Recommended Next Steps

### Immediate (Low-effort, testable)

1. **Hypothesis 3A**: Update chunk scoring prompt with explicit symptom definitions
2. **Hypothesis 2A**: Create a "frequency inference" prompt variant and ablate

### Research (Higher effort)

3. **Hypothesis 7A**: Implement fuzzy grounding and compare to substring match
4. **Hypothesis 4B**: Implement severity-aware reranking

### Fundamental Re-evaluation

5. Consider whether PHQ-8 frequency scoring is the right task for this dataset
6. Explore alternative tasks: binary depression detection, severity classification (none/mild/moderate/severe)

---

## 11. Related Documentation

- [Task Validity](../clinical/task-validity.md) — **SSOT**: construct mismatch and valid claims
- [Few-Shot Analysis](../results/few-shot-analysis.md) — Why few-shot may not beat zero-shot
- [RAG Design Rationale](../rag/design-rationale.md) — Original design decisions
- [Metrics and Evaluation](../statistics/metrics-and-evaluation.md) — AURC/AUGRC definitions
- [Specs Index](../_specs/index.md) — Implementation specs (061-063 address task validity)

---

## Sources

- [DAIC-WOZ Database](https://dcapswoz.ict.usc.edu/)
- [DAIC-WOZ Documentation](https://dcapswoz.ict.usc.edu/wp-content/uploads/2022/02/DAICWOZDepression_Documentation.pdf)
- [DAIC-WOZ: On the Validity of Using the Therapist's prompts](https://arxiv.org/abs/2404.14463)
- [The Distress Analysis Interview Corpus](https://www.researchgate.net/publication/311643727_The_Distress_Analysis_Interview_Corpus_of_human_and_computer_interviews)
- PHQ-8 validation: [The PHQ-8 as a measure of current depression in the general population](https://pubmed.ncbi.nlm.nih.gov/18752852/)
- PHQ-8 reliability example (test-retest ICC): https://pubmed.ncbi.nlm.nih.gov/32661929/
- PHQ-8 internal consistency example: https://pubmed.ncbi.nlm.nih.gov/35591921/
- DAIC-WOZ + PHQ-8 prediction prior art (LLMs): https://pubmed.ncbi.nlm.nih.gov/40720397/
- DAIC-WOZ + PHQ-8 prediction prior art (text regression): https://pubmed.ncbi.nlm.nih.gov/37398577/
- Selective classification evaluation pitfalls (AUGRC): http://arxiv.org/abs/2407.01032
- Clinical-domain language models (embedding ablations): http://arxiv.org/abs/1904.05342 (ClinicalBERT), http://arxiv.org/abs/2007.15779 (PubMedBERT)
