# SENIOR REVIEW REQUEST: vibe-check Master Spec (v1)

You are a senior AI/ML architect and applied-statistics reviewer. Please perform an adversarial, accuracy-first review of a specification for a new repository.

## Your Mission

1. **Validate technical accuracy**: Are model IDs, pricing, and API patterns correct *as of today*? If not, propose corrected values.
2. **Find blockers**: Anything missing that prevents implementation, reproducibility, or legal redistribution?
3. **Assess architecture**: Is LangGraph + PydanticAI appropriate, or is a simpler async pipeline sufficient? Justify.
4. **Review consensus algorithm**: Is the Dirichlet posterior + entropy approach sound for **ordinal** PHQ-8 scores? Is `range ≥ 2` a good arbitration trigger?
5. **Validate the validation protocol**: Do Phases 0–3 actually de-risk the system and prove it works?
6. **Suggest improvements**: What would you change to make this production-grade and scientifically defensible?

## Context (Why We’re Doing This)

- `ai-psychiatrist` reproduces PHQ-8 scoring well, but **DAIC-WOZ licensing blocks redistribution/deployment**.
- We want to create a **freely redistributable** reference corpus by scoring **SQPsychConv** (2,090 synthetic therapy dialogues) with **PHQ-8 item scores (0–3)** plus a **self-harm/death mention flag**.
- We will validate scorer competence and downstream usefulness using **DAIC-WOZ locally** (not redistributed).

## Key Decisions To Validate

| Decision | Current Choice | What You Should Check |
|----------|----------------|------------------------|
| Scoring Metric | PHQ-8 + self-harm/death boolean tag | Is PHQ-8 defensible vs PHQ-9 / alternative scales? |
| Jurors | OpenAI + Anthropic + Google (cross-vendor) | Are the chosen “frontier” models real + available + correct IDs? |
| Judge | Stronger model, used only on disagreements | Should it be a different provider/family to reduce correlated errors? |
| Runs | 2 runs/model (6 votes/item) | Is this worth the cost? Would 1 run + more models be better? |
| Aggregation | Dirichlet smoothing + entropy | Is this the right approach for ordinal ratings? Better alternatives? |
| Disagreement | `range ≥ 2` triggers arbitration | Too sensitive / too lenient? What should replace it? |
| Preprocessing | Client-only text (remove therapist turns) | Is this correct for *scoring* and/or for *retrieval*? |
| Data split | Use HF dataset splits | **Important**: our local analysis suggests HF train/test may be byte-identical for some SQPsychConv variants; validate and propose an explicit resplit protocol. |

## Files To Review (SSOT)

Primary:
- `docs/_brainstorming/sqpsychconv/new_repo/SPEC-vibe-check.md`

Supporting context from this repo:
- `_literature/markdown/SQPsychConv/SQPsychConv.md` (paper)
- `docs/research/spec-sqpsychconv-cross-validation.md` (feasibility + dataset realities)
- `docs/research/deep-research-prompt-sqpsychconv-multi-agent-scoring.md` (research questions + constraints)

## Review Rules (Important)

- **No guessing**: if you can’t verify a claim (e.g., pricing/model IDs), mark it **UNVERIFIED** and recommend a verification step + source.
- **Cite sources** for any quantitative claims (pricing, benchmark scores, refusal rates, etc.).
- **Propose concrete edits**: If something is wrong, show the corrected table/value or rewritten paragraph.
- **Be clinically cautious**: PHQ scoring is a mental health measurement task; highlight safety/ethics issues and mitigations.

## Deliverable Format

Please respond with:

1. **APPROVE / NEEDS REVISION / REJECT** (with short rationale)
2. **Critical issues** (must-fix before implementation)
3. **Major improvements** (high leverage, not strictly required)
4. **Minor nits** (style/clarity)
5. **Ready-to-implement assessment** (what’s missing to start coding)
