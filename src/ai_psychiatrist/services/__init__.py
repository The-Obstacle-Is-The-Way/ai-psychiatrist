"""Business logic services.

This module provides service classes for transcript loading, chunking,
ground truth data access, assessment feedback loops, and embedding-based
similarity search for few-shot prompting.

Public API:
- TranscriptService: Load and manage DAIC-WOZ transcripts
- TranscriptChunker: Create overlapping chunks for embedding
- GroundTruthService: Load PHQ-8 ground truth scores
- FeedbackLoopService: Iterative assessment refinement
- FeedbackLoopResult: Result of feedback loop process
- EmbeddingService: Generate embeddings and find similar chunks
- ReferenceBundle: Bundle of reference examples for prompts
- ReferenceStore: Pre-computed embeddings and scores storage
"""

from ai_psychiatrist.services.chunking import TranscriptChunker
from ai_psychiatrist.services.embedding import EmbeddingService, ReferenceBundle
from ai_psychiatrist.services.feedback_loop import FeedbackLoopResult, FeedbackLoopService
from ai_psychiatrist.services.ground_truth import GroundTruthService
from ai_psychiatrist.services.reference_store import ReferenceStore
from ai_psychiatrist.services.transcript import TranscriptService

__all__ = [
    "EmbeddingService",
    "FeedbackLoopResult",
    "FeedbackLoopService",
    "GroundTruthService",
    "ReferenceBundle",
    "ReferenceStore",
    "TranscriptChunker",
    "TranscriptService",
]
