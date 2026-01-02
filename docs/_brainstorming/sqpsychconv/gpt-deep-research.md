Executive decisions (what I would do)
	1.	Score PHQ‑8 (item-level 0–3 + total 0–24) as the primary label.
Add a separate “self-harm / death content mentioned” tag (boolean) instead of producing a PHQ‑9 item‑9 numeric score.
Rationale: matches DAIC‑WOZ validation target, avoids high‑stakes unreliability around item‑9, and PHQ‑8 is already treated as a valid severity measure with the same cutpoints as PHQ‑9 in population research contexts.
	2.	Use a “panel-of-models → probabilistic aggregation → selective arbitration” design.
Your hypothesis is directionally right, but I’d tighten it:
	•	Prefer independent scoring (avoid “debate herding”)
	•	Aggregate as a posterior distribution over {0,1,2,3} per item
	•	Escalate to a judge only when needed (range ≥ 2 or low evidence)
This aligns with newer “jury” evidence for improving reliability of LLM judging.
	3.	Framework: Python-only custom orchestrator + Pydantic schemas + SQLite checkpointing (MVP), optionally swapping in LangGraph checkpointing if you want graph-native persistence.
PydanticAI is useful for multi-provider structured outputs, but you don’t need an orchestration framework to score ~2k dialogues; you need concurrency, rate limits, retries, and resumability.

⸻

1) Scoring metric decision matrix (PHQ‑8 vs PHQ‑9 vs alternatives)

You’re building a retrieval corpus with a real-data validator (DAIC‑WOZ PHQ‑8). That dominates the decision.

Key constraints from your project docs
	•	SQPsychConv public HF exports are binary mdd vs control only—no continuous symptom scale included, so you must impute PHQ scores.  ￼
	•	The cross‑dataset plan is explicitly: LLM-score SQPsychConv → evaluate on DAIC‑WOZ ground truth PHQ‑8 (not circular if DAIC‑WOZ labels are the validator).  ￼
	•	Your own notes flag PHQ‑9 item‑9 content is extremely rare (~0.3%) in SQPsychConv, which makes numeric item‑9 severity especially unstable.  ￼

What the literature says (relevant, recent, and actually about transcripts)
	•	There is at least one recent study explicitly evaluating LLMs predicting PHQ‑8 items from DAIC‑WOZ transcripts, with best performance from frontier models and item-level prediction framing.
	•	There are also cautionary results showing LLMs can perform poorly estimating PHQ‑9 from clinical notes when PROM content is removed, i.e., these tasks are not automatically reliable.
	•	If you considered clinician-rated scales (MADRS/HAMD): one recent interview-based modeling paper explicitly noted certain clinician items are hard/invalid from text alone (e.g., “apparent sadness” depends on nonverbal cues), reinforcing your point about observability.

⸻

Decision matrix

Option	Pros	Cons / Risk	Fit for your use-case
PHQ‑8 (recommended)	Directly matches DAIC‑WOZ target; self-report scale; avoids suicidality item; severity cutpoints well-established (0–4, 5–9, 10–14, 15–19, 20–24).	Still inferential from conversation; some items may be undermentioned (psychomotor, appetite).	Best: aligns validation + minimizes ethical/technical failure modes.
PHQ‑9 (not recommended as primary)	Clinically common; adds item‑9 signal	Numeric item‑9 from transcripts is high-stakes + often ambiguous; PHQ‑8 exists partly because follow-up for item‑9 is impractical in some contexts; positives to item‑9 often reflect passive death thoughts rather than active suicidality.	High risk + mismatched to validation target.
PHQ‑8 + “self-harm/death mention” tag (recommended add-on)	Keeps your corpus safe + useful for retrieval filtering; avoids pretending you have a validated suicidality severity label; handles the fact mentions are rare in SQPsychConv.  ￼	Doesn’t produce a PHQ‑9 total; but that’s fine for DAIC‑WOZ validation.	Best add-on: safety + utility.
PHQ‑8 Days (number of days)	More granular	Not realistically inferable from free-form dialogue; DAIC‑WOZ validation uses PHQ‑8 totals anyway.	Not worth it.
Clinician-rated scales (HAMD/MADRS)	Clinical prestige	Requires observation/nonverbal cues; weak epistemic grounding from transcript-only; even recent work excludes some nonverbal-dependent items.	Don’t do this for transcript-only scoring.
BDI-II	Self-report	Not your validation target; mapping between scales isn’t item-to-item validated (your spec calls this out).  ￼	Not ideal for DAIC‑WOZ alignment.

PHQ‑8 ↔ PHQ‑9 crosswalk: does it make the decision moot?

For depression severity (not suicidality), it’s almost moot because:
	•	PHQ‑8 is essentially PHQ‑9 minus item‑9.
	•	The severity cutpoints are treated as identical in key public-health usage; PHQ‑8 is used when item‑9 follow-up isn’t feasible.
So for your DAIC‑WOZ objective, PHQ‑8 is the correct canonical label.

Ethical analysis: is synthetic suicidality scoring acceptable?

Two separate questions:

A) Is it ethically “allowed” to label suicidality content in synthetic text?
Generally yes—because no real patient is endangered—but…

B) Is it ethically/clinically valid to generate a 0–3 PHQ‑9 item‑9 score from therapy dialogue?
I would say no as a primary label, because:
	•	It invites downstream misuse as a “suicide risk score” (even though PHQ‑9 item‑9 isn’t a suicide risk instrument).
	•	Your own corpus barely contains item‑9-like mentions (~0.3%), making numeric calibration essentially impossible.  ￼

Safer move: store a non-clinical binary tag: mentions_self_harm_or_death: true/false, plus evidence quotes, plus a bright-line disclaimer: “not validated for risk.”

⸻

2) Multi-model consensus architecture (reduce stochasticity, increase reliability)

Why a multi-agent panel is justified here

You have three independent sources of noise:
	1.	Model stochasticity (even at low temperature, some nondeterminism persists in practice)
	2.	Ambiguity of dialogue evidence (“frequency over 2 weeks” isn’t always stated)
	3.	Systematic bias (models over/under-score certain symptoms)

Recent evidence supports that a panel/jury of diverse LLMs can improve reliability of judging versus a single model.
Also, your SQPsychConv paper notes LLM-based evaluation can show high inter-model correlation (risk of shared bias), reinforcing the need for diversity and explicit aggregation rather than “one judge to rule them all.”  ￼

Critical preprocessing guardrail (important for DAIC‑WOZ validation)

DAIC‑WOZ has documented issues where using the interviewer’s prompts can inflate depression detection, because models learn shortcuts around mental-health-history prompt blocks instead of participant language.
That means your scoring/prediction stack should default to:
	•	Participant-only scoring (or participant-weighted) for anything validated against DAIC‑WOZ.

Your own notes already flag this exact concern.  ￼

⸻

Recommended consensus pipeline

High-level diagram (architecture)

flowchart LR
  A[Dialogue] --> B[Preprocess: role-split, cleanup, chunk if needed]
  B --> C1[Scorer A: OpenAI frontier model]
  B --> C2[Scorer B: Anthropic frontier model]
  B --> C3[Scorer C: Google frontier model]

  C1 --> D[Aggregate per-item distributions]
  C2 --> D
  C3 --> D

  D --> E{Disagreement / low-evidence?}
  E -- no --> F[Finalize: PHQ-8 items+total + uncertainty]
  E -- yes --> G[Arbitration Judge (strong, different family if possible)]
  G --> H[Final report w/ rationale + uncertainty]
  H --> F

  F --> I[(SQLite/JSONL checkpoint store)]
  F --> J[Vectorize + retrieval corpus export]

Concrete scoring output (what each scorer must emit)

For each dialogue, each scorer outputs:
	•	items: dict of 8 items → {score: 0..3, evidence: [quotes], confidence: 0..1, insuff_evidence: bool}
	•	total_score: sum of 8 item scores
	•	severity_bucket: derived from standard cutpoints (0–4, 5–9, 10–14, 15–19, 20–24)
	•	mentions_self_harm_or_death: boolean + evidence (not numeric item‑9)
	•	notes: brief

Aggregation (this is the “frontier-quality” part)

Instead of “pick one score,” compute a posterior-ish distribution:

For each item i:
	1.	Collect votes from all scorers (and runs): v_{i,1}, v_{i,2}, …
	2.	Convert to a probability over {0,1,2,3} with Dirichlet smoothing:
	•	p_i(s) \propto \alpha + \#\{v_{i,*}=s\}
	3.	Final item score = argmax p_i or expected value (I’d store both)
	4.	Item uncertainty = entropy of p_i

Total score:
	•	total_expected = sum(E[item_i])
	•	total_p50 = sum(median(item_i)) (or median of totals from each scorer-run)

This gives you:
	•	A deterministic label (item_mode, total_mode)
	•	A calibrated-ish uncertainty (entropy, vote_margin)
	•	A principled “confidence” that’s not just the model’s self-report

Disagreement threshold (practical + defensible)

Use item-level thresholds since PHQ‑8 is item-summed:

Trigger arbitration if ANY item meets one of:
	•	range(item_scores) >= 2 (e.g., {0,2,2} or {1,3,2})
	•	No two scorers agree and vote margin is tiny (e.g., counts {1,1,1} if you use 3 runs)
	•	“Insufficient evidence” flagged by ≥2 scorers
	•	Evidence quotes are missing for ≥3 items (low auditability)

Your internal spec already frames “multi-run scoring for stability” and treating variance as a signal.  ￼

Should the judge be stronger or different family?

Answer: both, when you can.
	•	If you keep OpenAI/Anthropic/Google as scorers, pick a judge that is the strongest available and not one of the scorers, to reduce correlated errors.
	•	If you must reuse a family, at least use a stronger tier (e.g., “flagship” vs “fast”).

This matches your own prompt’s open question framing.  ￼

How many runs per model?

Given you’re not cost-constrained, but want implementability:

Recommended default: 2 runs × 3 models = 6 passes per dialogue, then:
	•	If aggregate_uncertainty is low → stop
	•	If high → run +1 pass for each model (now 9) OR go straight to judge

Why not always 5–10 runs? Because you’ll likely find diminishing returns; your spec’s “Run1..Run5 → mean/std” is a good pilot method to empirically determine if >2 adds value.  ￼

⸻

3) Framework recommendation + minimal viable orchestration (Python-only)

Recommendation

MVP (fastest, most robust for batch annotation)

Custom Python pipeline:
	•	asyncio concurrency
	•	tenacity retries + jitter
	•	aiolimiter rate limiting per provider
	•	pydantic schemas for validation
	•	SQLite checkpoint DB (resume-safe)
	•	JSONL exports + a “scoring manifest” recording prompt/model/version hashes

This is the simplest thing that’s still production-grade for ~2k items.

When to use LangGraph

If you want graph-native checkpointing, LangGraph has explicit support for persistence/checkpointers (e.g., Postgres) for fault tolerance and stateful runs.
But for your use-case, it’s optional complexity.

When to use PydanticAI

PydanticAI is attractive specifically because it’s built around typed/validated outputs and supports multiple model providers.
I’d use it if you want one uniform abstraction across OpenAI/Anthropic/Gemini and you’re okay adopting a newer framework.

When not to use Microsoft Agent Framework

Microsoft Agent Framework is aimed at enterprise agent systems; it’s capable and has Python quickstarts, but it’s likely more moving parts than you need for a “score 2k dialogues with retries” job.

⸻

Starter code (drop-in skeleton)

This is “real” orchestration code; you only plug in the vendor call in score_with_model().

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

# ---------- Schema ----------

PHQScore = Literal[0, 1, 2, 3]

class PHQItem(BaseModel):
    score: PHQScore
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    insuff_evidence: bool = False

class PHQ8Report(BaseModel):
    # Required metadata
    file_id: str
    model_id: str
    prompt_version: str

    # PHQ-8 items
    anhedonia: PHQItem
    depressed_mood: PHQItem
    sleep: PHQItem
    fatigue: PHQItem
    appetite: PHQItem
    guilt_worthlessness: PHQItem
    concentration: PHQItem
    psychomotor: PHQItem

    # Derived
    total_score: int = Field(ge=0, le=24)
    severity_bucket: Literal["0-4", "5-9", "10-14", "15-19", "20-24"]

    # Safety tag (NOT PHQ-9 scoring)
    mentions_self_harm_or_death: bool = False
    self_harm_evidence: list[str] = Field(default_factory=list)

class AggregatedPHQ8(BaseModel):
    file_id: str
    prompt_version: str

    # posterior-ish distributions per item (counts)
    item_vote_counts: dict[str, dict[str, int]]

    # final labels
    item_mode: dict[str, int]
    item_expected: dict[str, float]
    total_mode: int
    total_expected: float
    severity_bucket: str

    # uncertainty
    item_entropy: dict[str, float]
    total_vote_std: float
    needs_arbitration: bool

    # audit
    per_model_reports: list[PHQ8Report]

# ---------- Checkpoint DB ----------

def init_db(path: str = "scores.sqlite") -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS results (
            file_id TEXT PRIMARY KEY,
            prompt_version TEXT NOT NULL,
            aggregated_json TEXT NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS raw_reports (
            file_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            report_json TEXT NOT NULL,
            PRIMARY KEY (file_id, model_id, prompt_version)
        )
    """)
    con.commit()
    return con

def already_done(con: sqlite3.Connection, file_id: str, prompt_version: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM results WHERE file_id=? AND prompt_version=?",
        (file_id, prompt_version),
    ).fetchone()
    return row is not None

# ---------- Scoring prompt ----------

def make_prompt(dialogue_text: str) -> str:
    # You should strongly consider feeding only Client/Participant turns here.
    return f"""
You are scoring PHQ-8 from a therapy dialogue.

Rules:
- Base scoring ONLY on what the CLIENT/PARTICIPANT says about THEIR OWN symptoms.
- If frequency over 2 weeks is unclear, infer conservatively and set insuff_evidence=true with low confidence.
- Return JSON that matches the provided schema exactly.
- Provide 1-3 short evidence quotes per item when possible.

PHQ-8 items are scored: 0=not at all, 1=several days, 2=more than half the days, 3=nearly every day.

Dialogue:
{dialogue_text}
""".strip()

# ---------- Provider call (plug-in) ----------

@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    provider: Literal["openai", "anthropic", "google"]

@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(initial=1, max=60))
async def score_with_model(model: ModelSpec, file_id: str, dialogue_text: str, prompt_version: str) -> PHQ8Report:
    prompt = make_prompt(dialogue_text)

    # TODO: implement actual API call for each provider.
    # IMPORTANT: enforce structured JSON output at the API level when possible.
    # OpenAI supports structured outputs / JSON schema.

    raw_json = await fake_llm_call_returning_json(model, prompt)  # <-- replace

    try:
        report = PHQ8Report.model_validate({**raw_json, "file_id": file_id, "model_id": model.model_id, "prompt_version": prompt_version})
    except ValidationError as e:
        raise RuntimeError(f"Invalid JSON from {model.model_id}: {e}") from e

    return report

# ---------- Aggregation math ----------

import math

def entropy_from_counts(counts: dict[int, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c == 0:
            continue
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent

def severity_bucket(total: int) -> str:
    if total <= 4: return "0-4"
    if total <= 9: return "5-9"
    if total <= 14: return "10-14"
    if total <= 19: return "15-19"
    return "20-24"

def aggregate_reports(file_id: str, prompt_version: str, reports: list[PHQ8Report]) -> AggregatedPHQ8:
    item_names = [
        "anhedonia","depressed_mood","sleep","fatigue",
        "appetite","guilt_worthlessness","concentration","psychomotor"
    ]

    # vote counts per item
    vote_counts: dict[str, dict[str, int]] = {}
    item_mode: dict[str, int] = {}
    item_expected: dict[str, float] = {}
    item_entropy: dict[str, float] = {}

    total_scores = [r.total_score for r in reports]

    needs_arbitration = False

    for item in item_names:
        counts = {0:0, 1:0, 2:0, 3:0}
        vals = []
        for r in reports:
            v = getattr(r, item).score
            counts[int(v)] += 1
            vals.append(int(v))

        # Disagreement rule: range >= 2
        if max(vals) - min(vals) >= 2:
            needs_arbitration = True

        # Convert to strings for JSON stability
        vote_counts[item] = {str(k): v for k, v in counts.items()}

        # mode
        mode_score = max(counts.items(), key=lambda kv: kv[1])[0]
        item_mode[item] = int(mode_score)

        # expected value
        tot = sum(counts.values())
        item_expected[item] = sum(k * (c / tot) for k, c in counts.items()) if tot else 0.0

        # entropy
        item_entropy[item] = entropy_from_counts(counts)

    total_mode = round(sum(item_mode.values()))
    total_expected = float(sum(item_expected.values()))

    # std for audit
    mean_total = sum(total_scores)/len(total_scores)
    total_std = math.sqrt(sum((t-mean_total)**2 for t in total_scores)/len(total_scores))

    # also trigger arbitration if totals are very unstable
    if total_std >= 2.0:
        needs_arbitration = True

    return AggregatedPHQ8(
        file_id=file_id,
        prompt_version=prompt_version,
        item_vote_counts=vote_counts,
        item_mode=item_mode,
        item_expected=item_expected,
        total_mode=total_mode,
        total_expected=total_expected,
        severity_bucket=severity_bucket(total_mode),
        item_entropy=item_entropy,
        total_vote_std=total_std,
        needs_arbitration=needs_arbitration,
        per_model_reports=reports
    )

# ---------- Orchestration ----------

async def score_dialogue_ensemble(
    con: sqlite3.Connection,
    file_id: str,
    dialogue_text: str,
    models: list[ModelSpec],
    runs_per_model: int,
    prompt_version: str,
) -> AggregatedPHQ8:
    if already_done(con, file_id, prompt_version):
        return None  # caller can skip

    tasks = []
    for m in models:
        for _ in range(runs_per_model):
            tasks.append(score_with_model(m, file_id, dialogue_text, prompt_version))

    reports: list[PHQ8Report] = await asyncio.gather(*tasks)

    # persist raw reports
    for r in reports:
        con.execute(
            "INSERT OR REPLACE INTO raw_reports(file_id, model_id, prompt_version, report_json) VALUES(?,?,?,?)",
            (file_id, r.model_id, prompt_version, r.model_dump_json()),
        )
    con.commit()

    agg = aggregate_reports(file_id, prompt_version, reports)

    # TODO: arbitration step if agg.needs_arbitration == True

    con.execute(
        "INSERT OR REPLACE INTO results(file_id, prompt_version, aggregated_json) VALUES(?,?,?)",
        (file_id, prompt_version, agg.model_dump_json()),
    )
    con.commit()

    return agg

Notes on model selection (Jan 2026 reality)
	•	OpenAI model naming and deprecations shift quickly; use the current recommended “latest chat” and structured outputs support rather than hardcoding legacy IDs.
	•	Google Gemini API versions also change; rely on the official changelog and model availability.

(Your code should treat model IDs as config, not constants.)

⸻

4) Validation protocol: prove synthetic → real transfer works

Your cross-validation spec already states the core logic: score synthetic → retrieve → predict → validate on DAIC‑WOZ PHQ‑8.  ￼

Here’s the “definitive” protocol I’d run (with ablations that answer the real scientific question, not just “did it work once”).

4.1 Pre-validation: prove your scorer is competent on DAIC‑WOZ

Before scoring SQPsychConv, run your ensemble scorer on DAIC‑WOZ training/dev transcripts (where labels exist) and report:
	•	Total score: MAE, RMSE, Pearson r, Spearman ρ
	•	Binary (PHQ‑8 ≥ 10): AUC, sensitivity, specificity, F1
	•	Item-level: MAE per item, weighted κ per item
	•	Reliability across ensemble: ICC(2,k), Krippendorff’s α (ordinal)

Your internal prompt explicitly calls for ICC/Krippendorff.  ￼

Critical preprocessing requirement: evaluate participant-only vs full dialogue.
Because DAIC‑WOZ prompt bias is real, you should treat “full dialogue” as a stress test, not the default.

Success criterion suggestion (reasonable targets):
	•	Total score MAE: aim for ≤ ~2–3 (depends on transcript-only limits)
	•	Binary AUC: solidly > 0.80
	•	Reliability: ICC ≥ 0.8 / Krippendorff α ≥ 0.7 for item scoring
(You can set these once you see DAIC‑WOZ baseline.)

4.2 Score SQPsychConv (the corpus creation step)
	•	Score all 2,090 dialogues (and treat train/test as one corpus if identical, per spec).  ￼
	•	Store: per-item mode/expected, total mode/expected, uncertainty fields, safety tag.

Also run internal sanity checks:
	•	Score distribution by condition (mdd vs control) should separate clearly.  ￼
	•	Cronbach’s alpha over 8 items should be plausible (internal consistency check).
	•	Flag outliers: control dialogues with PHQ‑8 ≥ 15, etc.

4.3 Build retrieval corpus + test transfer on DAIC‑WOZ

Build a retrieval predictor with three baselines:

Baseline A (no retrieval):
LLM predicts PHQ‑8 directly from DAIC‑WOZ participant text.

Baseline B (kNN label transfer, non-LLM):
Embed DAIC‑WOZ transcript, retrieve k synthetic dialogues, predict total as weighted average of retrieved total_expected.

Baseline C (RAG few-shot, your intended use):
Retrieve k synthetic dialogues and give them as exemplars to a predictor model:
	•	“Here are k similar cases with their PHQ‑8 item labels; now score this new transcript.”

Then evaluate all on DAIC‑WOZ with ground truth.

Ablations that answer the science question:
	•	k ∈ {3, 5, 10, 20}
	•	participant-only vs full transcript embeddings
	•	use total_mode vs total_expected vs “calibrated total”
	•	include uncertainty gating: only retrieve exemplars with low scorer entropy
	•	remove the highest-similarity exemplar (anti-leak test)

Why it’s not circular (and publishable):
	•	Training/knowledge source: synthetic corpus
	•	Validator: DAIC‑WOZ real PHQ‑8 labels
	•	You’re empirically testing whether synthetic labeled exemplars improve real prediction.  ￼

4.4 Human validation slice (optional but high value)

Your own spec suggests 50–100 samples.  ￼

Do this surgically:
	•	Sample high-uncertainty dialogues (where ensemble disagrees) + a matched set of low-uncertainty dialogues.
	•	Have a clinician rater score PHQ‑8 from transcript only.
	•	Report κ / ICC vs ensemble.

This gives you:
	•	A real-world estimate of “how often are we wrong when we say we’re confident?”

⸻

Practical implementation risks & mitigations (the stuff that usually breaks these projects)

1) “Therapist prompt bias” and shortcut learning

Even though SQPsychConv is CBT-structured, DAIC‑WOZ evidence shows models can exploit interviewer prompts as shortcuts.
Mitigation:
	•	Always compute a participant-only representation and treat it as canonical for DAIC‑WOZ-facing validation.

2) LLM-on-LLM style recognition (synthetic circularity)

Your cross-validation spec explicitly flags the risk: the scorer may recognize “LLM style” rather than clinical content.  ￼
Mitigation:
	•	Use cross-vendor frontier scorers (not the generator family)
	•	Penalize evidence that is not symptom-specific
	•	Validate on DAIC‑WOZ

3) SQPsychConv generator variants & quality

SQPsychConv was generated across multiple open models.  ￼
Your own notes suggest gemma/qwen2.5 are highest-quality per expert eval.  ￼
Mitigation:
	•	Score everything, but store client_model as metadata and analyze whether certain generators systematically distort PHQ patterns.

4) Data quirks in your export
	•	Dialogues length range and code-switching issues exist in local exports.  ￼
Mitigation:
	•	Clean CJK artifacts if needed (or ignore; frontier models will usually handle)
	•	Chunk only if you exceed context limits; otherwise score whole

⸻

If you want the “most publishable / defensible” version (one extra upgrade)

Add calibrated weighting of scorers using DAIC‑WOZ dev set:
	•	Run each scorer on DAIC‑WOZ train/dev
	•	Learn per-item confusion tendencies (ordinal)
	•	Weight votes accordingly in aggregation

This turns your ensemble into something closer to “learned rater reliability” instead of “three opinions.”

⸻

Final answers to your three decisions

Decision 1: PHQ‑8 vs PHQ‑9

PHQ‑8 primary. Add a non-clinical “self-harm/death mention” tag instead of PHQ‑9 item‑9 scoring.
This is maximally aligned with DAIC‑WOZ and avoids the worst ethical/technical pitfalls (and your corpus barely contains item‑9 content anyway).  ￼

Decision 2: Multi-model consensus architecture

Your panel + threshold + judge concept is right, but make it:
	•	independent scoring
	•	distributional aggregation
	•	selective arbitration (range ≥ 2 / low evidence)
	•	participant-only for DAIC‑WOZ-facing validation
Use 2 runs per model by default, expand adaptively.  ￼

Decision 3: Framework selection

Python custom orchestrator is the best ROI for 2k dialogues.
	•	Add Pydantic schemas for structured output
	•	Add SQLite checkpoints for resumability
Optionally adopt LangGraph only if you want graph-native persistence.
PydanticAI can be helpful for multi-provider structured outputs, but not strictly required.

⸻

If you want, I can also propose a scoring prompt pack (rubric + anchor examples per 0/1/2/3 for each PHQ item) and a QC dashboard spec (uncertainty heatmaps, disagreement drilldowns). That’s usually what turns this from “it runs” into “it’s actually auditable and paper-ready.”
