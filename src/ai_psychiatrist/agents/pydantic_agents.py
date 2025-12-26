"""Factory functions for Pydantic AI agents used in this project."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai import Agent, TextOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from ai_psychiatrist.agents.extractors import (
    extract_judge_metric,
    extract_meta_review,
    extract_quantitative,
)

if TYPE_CHECKING:
    from ai_psychiatrist.agents.output_models import (
        JudgeMetricOutput,
        MetaReviewOutput,
        QuantitativeOutput,
    )


def _ollama_v1_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    return f"{base_url}/v1"


def create_quantitative_agent(
    *,
    model_name: str,
    base_url: str,
    retries: int,
    system_prompt: str,
) -> Agent[None, QuantitativeOutput]:
    """Create a Pydantic AI agent for quantitative scoring."""
    model = OpenAIChatModel(
        model_name, provider=OllamaProvider(base_url=_ollama_v1_base_url(base_url))
    )
    agent: Agent[None, QuantitativeOutput] = Agent(
        model=model,
        output_type=TextOutput(extract_quantitative),
        retries=retries,
        system_prompt=system_prompt,
    )
    return agent


def create_judge_metric_agent(
    *,
    model_name: str,
    base_url: str,
    retries: int,
    system_prompt: str,
) -> Agent[None, JudgeMetricOutput]:
    """Create a Pydantic AI agent for judge metric scoring."""
    model = OpenAIChatModel(
        model_name, provider=OllamaProvider(base_url=_ollama_v1_base_url(base_url))
    )
    agent: Agent[None, JudgeMetricOutput] = Agent(
        model=model,
        output_type=TextOutput(extract_judge_metric),
        retries=retries,
        system_prompt=system_prompt,
    )
    return agent


def create_meta_review_agent(
    *,
    model_name: str,
    base_url: str,
    retries: int,
    system_prompt: str,
) -> Agent[None, MetaReviewOutput]:
    """Create a Pydantic AI agent for meta-review output validation."""
    model = OpenAIChatModel(
        model_name, provider=OllamaProvider(base_url=_ollama_v1_base_url(base_url))
    )
    agent: Agent[None, MetaReviewOutput] = Agent(
        model=model,
        output_type=TextOutput(extract_meta_review),
        retries=retries,
        system_prompt=system_prompt,
    )
    return agent
