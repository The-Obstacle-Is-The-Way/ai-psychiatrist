from __future__ import annotations

import httpx
import pytest


@pytest.mark.e2e
@pytest.mark.ollama
@pytest.mark.slow
class TestServerRealOllama:
    async def test_server_full_pipeline_transcript_text_zero_shot_real_ollama(
        self,
        ollama_client,
        sample_transcript: str,
    ) -> None:
        import server  # noqa: PLC0415

        # ollama_client fixture provides skip logic; server creates its own client
        _ = ollama_client

        async with server.lifespan(server.app):
            transport = httpx.ASGITransport(app=server.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                health = await client.get("/health")
                assert health.status_code == 200

                response = await client.post(
                    "/full_pipeline",
                    json={
                        "transcript_text": sample_transcript,
                        "mode": 0,  # zero-shot (does not require embeddings artifact)
                    },
                )
                assert response.status_code == 200, response.text

                payload = response.json()
                assert "qualitative" in payload
                assert "quantitative" in payload
                assert "evaluation" in payload
                assert "meta_review" in payload
