"""Additional coverage tests for QuantitativeAssessmentAgent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from ai_psychiatrist.agents.prompts.quantitative import DOMAIN_KEYWORDS
from ai_psychiatrist.agents.quantitative import QuantitativeAssessmentAgent
from ai_psychiatrist.domain.entities import Transcript
from ai_psychiatrist.domain.enums import AssessmentMode, PHQ8Item
from tests.fixtures.mock_llm import MockLLMClient

SAMPLE_EVIDENCE_RESPONSE = json.dumps({k: ["evidence"] for k in DOMAIN_KEYWORDS})


class TestQuantitativeCoverage:
    """Tests targeting specific code branches for 100% coverage."""

    @pytest.fixture
    def transcript(self) -> Transcript:
        return Transcript(participant_id=1, text="Test transcript.")

    @pytest.mark.asyncio
    async def test_extract_evidence_json_error(self, transcript: Transcript) -> None:
        """Test _extract_evidence handling of completely invalid JSON."""
        # First response is invalid JSON for evidence
        # Second response is valid scoring (to allow assess to finish)
        scoring_response = json.dumps(
            {k: {"evidence": "e", "reason": "r", "score": 0} for k in DOMAIN_KEYWORDS}
        )

        client = MockLLMClient(chat_responses=["NOT JSON AT ALL", scoring_response])

        agent = QuantitativeAssessmentAgent(client, mode=AssessmentMode.ZERO_SHOT)

        # This should log a warning and return empty dict, then backfill
        result = await agent.assess(transcript)
        assert result.total_score is not None

    @pytest.mark.asyncio
    async def test_parse_response_strategy_2_failure(self, transcript: Transcript) -> None:
        """Test Strategy 2 (Answer block) finding a block but failing to parse JSON."""
        # Evidence response (valid)
        evidence_resp = SAMPLE_EVIDENCE_RESPONSE

        # Scoring response: Has <answer> tags but content is invalid JSON
        # This forces it to go to Strategy 3 (LLM Repair)
        bad_answer_block = "<answer>{ invalid json </answer>"

        # Repair response (valid)
        repair_resp = json.dumps(
            {k: {"evidence": "repaired", "reason": "r", "score": 1} for k in DOMAIN_KEYWORDS}
        )

        client = MockLLMClient(chat_responses=[evidence_resp, bad_answer_block, repair_resp])

        agent = QuantitativeAssessmentAgent(client, mode=AssessmentMode.ZERO_SHOT)
        result = await agent.assess(transcript)

        # Should have used the repaired score
        assert result.items[PHQ8Item.DEPRESSED].evidence == "repaired"
        assert result.items[PHQ8Item.DEPRESSED].score == 1

    @pytest.mark.asyncio
    async def test_parse_response_strategy_3_failure(self, transcript: Transcript) -> None:
        """Test Strategy 3 (LLM Repair) failing to produce valid JSON."""
        evidence_resp = SAMPLE_EVIDENCE_RESPONSE
        # Initial response: completely broken
        raw_resp = "COMPLETE GARBAGE"
        # Repair response: ALSO broken
        repair_resp = "STILL GARBAGE"

        client = MockLLMClient(chat_responses=[evidence_resp, raw_resp, repair_resp])

        agent = QuantitativeAssessmentAgent(client, mode=AssessmentMode.ZERO_SHOT)
        result = await agent.assess(transcript)

        # Should fall back to empty skeleton (Strategy 4)
        assert result.items[PHQ8Item.DEPRESSED].score is None
        assert result.items[PHQ8Item.DEPRESSED].evidence == "No relevant evidence found"

    @pytest.mark.asyncio
    async def test_validate_and_normalize_score_types(self, transcript: Transcript) -> None:
        """Test various score formats in _validate_and_normalize."""
        evidence_resp = SAMPLE_EVIDENCE_RESPONSE

        # JSON with mixed score types
        mixed_scores = json.dumps(
            {
                "PHQ8_NoInterest": {"score": 3},  # int 3 -> 3
                "PHQ8_Depressed": {"score": "2"},  # str "2" -> 2
                "PHQ8_Sleep": {"score": "N/A"},  # str "N/A" -> None
                "PHQ8_Tired": {"score": "n/a"},  # str "n/a" -> None (case insensitive check?)
                "PHQ8_Appetite": {"score": "invalid"},  # str "invalid" -> None
                "PHQ8_Failure": {"score": 4},  # int 4 -> None (out of bounds)
                "PHQ8_Concentrating": {"score": -1},  # int -1 -> None (out of bounds)
                "PHQ8_Moving": {"score": None},  # None -> None
            }
        )

        client = MockLLMClient(chat_responses=[evidence_resp, mixed_scores])
        agent = QuantitativeAssessmentAgent(client, mode=AssessmentMode.ZERO_SHOT)
        result = await agent.assess(transcript)

        assert result.items[PHQ8Item.NO_INTEREST].score == 3
        assert result.items[PHQ8Item.DEPRESSED].score == 2
        assert result.items[PHQ8Item.SLEEP].score is None
        assert result.items[PHQ8Item.TIRED].score is None
        assert result.items[PHQ8Item.APPETITE].score is None
        assert result.items[PHQ8Item.FAILURE].score is None
        assert result.items[PHQ8Item.CONCENTRATING].score is None
        assert result.items[PHQ8Item.MOVING].score is None

    @pytest.mark.asyncio
    async def test_strip_json_block_variations(self, transcript: Transcript) -> None:
        """Test _strip_json_block with different formats."""

        # 1. Markdown with "json" language identifier
        resp1 = """```json
{"PHQ8_NoInterest": {"score": 1}}
```"""
        # 2. Markdown without language identifier
        resp2 = """```
{"PHQ8_NoInterest": {"score": 2}}
```"""
        # 3. Block at very start
        resp3 = """```json{"PHQ8_NoInterest": {"score": 3}}```"""

        client = MockLLMClient(
            chat_responses=[
                SAMPLE_EVIDENCE_RESPONSE,
                resp1,
                SAMPLE_EVIDENCE_RESPONSE,
                resp2,
                SAMPLE_EVIDENCE_RESPONSE,
                resp3,
            ]
        )

        agent = QuantitativeAssessmentAgent(client, mode=AssessmentMode.ZERO_SHOT)

        res1 = await agent.assess(transcript)
        assert res1.items[PHQ8Item.NO_INTEREST].score == 1

        res2 = await agent.assess(transcript)
        assert res2.items[PHQ8Item.NO_INTEREST].score == 2

        res3 = await agent.assess(transcript)
        assert res3.items[PHQ8Item.NO_INTEREST].score == 3
