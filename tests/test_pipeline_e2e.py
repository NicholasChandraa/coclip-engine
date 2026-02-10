"""
End-to-end integration test for full LangGraph pipeline.

Tests the complete flow:
START → Transcription → Analysis → Editing → Finalization → END

This validates:
1. Graph compilation and execution
2. State transitions between nodes
3. Conditional routing (Command API)
4. Progress tracking
5. Error handling
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from app.graphs import run_video_processing_pipeline
from app.utils.logging import logger


async def test_full_pipeline_with_mock():
    """
    Test full pipeline with mocked transcription.

    This test validates the complete graph flow without needing
    an actual video file or WhisperX processing.
    """
    logger.info("=" * 70)
    logger.info("🚀 FULL PIPELINE END-TO-END TEST")
    logger.info("=" * 70)

    # Mock Redis
    mock_redis = AsyncMock()
    mock_redis.set = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)

    # Test parameters
    job_id = "e2e_test_job"
    video_path = "/tmp/mock_video.mp4"

    logger.info(f"📝 Job ID: {job_id}")
    logger.info(f"📹 Video Path: {video_path}")
    logger.info("")

    # Mock transcription result
    mock_segment_1 = MagicMock()
    mock_segment_1.start = 0.0
    mock_segment_1.end = 30.0
    mock_segment_1.text = (
        "This is an absolutely amazing viral moment that will blow your mind!"
    )
    mock_segment_1.speaker = "Speaker A"

    mock_segment_2 = MagicMock()
    mock_segment_2.start = 30.0
    mock_segment_2.end = 60.0
    mock_segment_2.text = (
        "And here's another incredible insight that everyone needs to hear!"
    )
    mock_segment_2.speaker = "Speaker B"

    mock_transcription = MagicMock()
    mock_transcription.segments = [mock_segment_1, mock_segment_2]
    mock_transcription.duration = 60.0
    mock_transcription.language = "en"
    mock_transcription.total_segments = 2
    mock_transcription.model_dump_json = MagicMock(return_value='{"test": "data"}')

    # Patch transcription node to return mock result
    async def mock_transcription_node(state, redis):
        from langgraph.types import Command

        logger.info("🎤 [MOCK] Transcription node executed")
        return Command(
            update={
                "transcription_result": mock_transcription,
                "progress": 25,
                "status": "analyzing",
                "current_phase": "Phase 2: Content Analysis",
            },
            goto="analysis",
        )

    try:
        # Patch the transcription node
        with patch(
            "app.graphs.nodes.transcription_node.transcription_node",
            mock_transcription_node,
        ):
            logger.info("🔧 Starting pipeline execution...")
            logger.info("")

            # Run pipeline
            final_state = await run_video_processing_pipeline(
                redis=mock_redis,
                job_id=job_id,
                video_path=video_path,
                use_streaming=False,
            )

            logger.info("")
            logger.info("=" * 70)
            logger.info("📊 FINAL STATE")
            logger.info("=" * 70)
            logger.info(f"✅ Status: {final_state.get('status')}")
            logger.info(f"✅ Progress: {final_state.get('progress')}%")
            logger.info(f"✅ Current Phase: {final_state.get('current_phase')}")
            logger.info(f"✅ Errors: {final_state.get('errors', [])}")

            # Check transcription
            if final_state.get("transcription_result"):
                logger.info(
                    f"✅ Transcription: {final_state['transcription_result'].total_segments} segments"
                )

            # Check analysis
            if final_state.get("clip_candidates"):
                logger.info(
                    f"✅ Clip Candidates: {len(final_state['clip_candidates'])} found"
                )
                for i, clip in enumerate(final_state["clip_candidates"]):
                    logger.info(
                        f"   Clip {i+1}: {clip.get('title')} (score: {clip.get('viral_score', 0)})"
                    )

            # Check editing
            if final_state.get("clips"):
                logger.info(f"✅ Generated Clips: {len(final_state['clips'])} clips")

            logger.info("=" * 70)

            # Assertions
            assert final_state.get("status") in [
                "completed",
                "completed_with_warnings",
                "failed",
            ]
            assert final_state.get("progress") == 100
            assert final_state.get("transcription_result") is not None

            if final_state.get("status") == "completed":
                logger.info("🎉 END-TO-END TEST PASSED!")
                return True
            else:
                logger.warning(
                    f"⚠️ Test completed with status: {final_state.get('status')}"
                )
                logger.warning(f"   Errors: {final_state.get('errors')}")
                return False

    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}", exc_info=True)
        return False


async def main():
    """Run end-to-end test."""
    logger.info("Starting end-to-end pipeline test...")
    logger.info("This test validates full LangGraph orchestration")
    logger.info("")

    success = await test_full_pipeline_with_mock()

    if success:
        logger.info("")
        logger.info("✅ All end-to-end tests PASSED")
        exit(0)
    else:
        logger.info("")
        logger.error("❌ End-to-end test FAILED")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
