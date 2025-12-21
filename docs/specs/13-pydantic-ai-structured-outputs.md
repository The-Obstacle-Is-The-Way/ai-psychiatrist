# Spec 13: Structured Output Migration (Pydantic AI)

> **STATUS: DEFERRED**
>
> This spec is deferred until paper replication is fully validated with real
> E2E testing. The current XML-based parsing pipeline is complete and working.
> Pydantic AI migration will provide cleaner, validated outputs but is not
> required for core functionality.
>
> **Tracked by**: [GitHub Issue #28](https://github.com/The-Obstacle-Is-The-Way/ai-psychiatrist/issues/28)
>
> **Last Updated**: 2025-12-21

---

## Objective

Introduce an **optional, post-replication** path that uses Pydantic AI to obtain
validated structured outputs from LLMs, while preserving the paper-aligned XML
path for replication.

## Preconditions

- Spec 11.5 (full pipeline verification) completed and validated
- Spec 12.5 (final cleanup) completed
- Paper replication results documented and stable

## Goals

- Add **validated structured output** parsing using Pydantic AI
- Keep **domain dataclasses** as the authoritative model
- Provide a **feature flag** to switch between XML and structured output modes
- Preserve paper fidelity by default (XML remains the default path)

## Non-Goals

- Do **not** change prompts or output formats during replication
- Do **not** migrate domain entities to Pydantic by default
- Do **not** remove the XML parsing path

## Deliverables

1. `src/ai_psychiatrist/agents/output_models.py`
   - Pydantic models describing LLM output schemas (qualitative, judge, etc.)
2. `src/ai_psychiatrist/infrastructure/llm/pydantic_ai_client.py`
   - Adapter around Pydantic AI `Agent` usage (configurable)
3. `src/ai_psychiatrist/config.py`
   - New setting: `LLM_OUTPUT_MODE = "xml" | "structured"`
   - New setting: `OLLAMA_BASE_URL` (OpenAI-compatible `/v1`)
4. Agent updates (e.g., `agents/qualitative.py`)
   - Use structured output path when `LLM_OUTPUT_MODE == "structured"`
   - Map Pydantic output models to existing domain entities
5. Tests
   - Unit tests for schema validation and mapping
   - Contract tests to ensure XML path remains unchanged

## Implementation Plan

### 1. Output Schemas

- Define Pydantic output models for each agent:
  - `QualitativeAssessmentOutput`
  - `JudgeFeedbackOutput`
  - Future: quantitative and meta-review outputs
- Keep these models **isolated** from domain entities.

### 2. Pydantic AI Adapter

- Use Pydantic AI with Ollama via OpenAI-compatible `/v1` endpoint.
- Provide a thin adapter so agents can call:
  - `client.generate_structured(prompt, output_model)`
- All Pydantic AI usage stays in **infrastructure**.

### 3. Feature Flag

- Default: `LLM_OUTPUT_MODE="xml"` (paper fidelity)
- When `structured`, agents:
  - call Pydantic AI
  - validate output
  - map to domain dataclasses

### 4. Mapping Layer

- Create a small mapping function per agent:
  - `to_domain(output_model) -> DomainEntity`
- Ensure all domain invariants still apply.

## Acceptance Criteria

- XML path unchanged and still the default
- Structured path produces **domain entities** equivalent to XML path
- Pydantic AI usage is **fully optional** and gated by config
- No domain code depends on Pydantic
- Tests cover both XML and structured paths

## Testing

- Unit tests for:
  - output schema validation
  - mapping to domain entities
  - error handling (invalid output -> retry or error)
- Contract tests:
  - XML output parsing unchanged
- Integration tests (mocked LLM responses):
  - verify structured path end-to-end without network

## Rollout Notes

- Keep the structured output path **off by default**
- Use a dedicated branch for experimentation
- Only enable after paper replication is complete and signed off
