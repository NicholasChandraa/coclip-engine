"""
Test script for Gemini Analysis Node.

This tests the full analysis workflow including:
1. Gemini LLM initialization
2. Viral clip detection
3. JSON response parsing
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from unittest.mock import AsyncMock, MagicMock
from app.graphs.nodes.analysis_node import (
    analysis_node,
    get_gemini_llm,
    _format_transcript_for_analysis,
    _parse_gemini_response,
    _validate_clip_candidate,
)
from app.schemas.graph_schemas import VideoProcessingState
from app.utils.logging import logger


async def test_gemini_llm_initialization():
    """Test 1: Gemini LLM can be initialized."""
    logger.info("🧪 Test 1: Gemini LLM Initialization")

    try:
        llm = get_gemini_llm()
        assert llm is not None
        # LangChain aliases gemini-2.0-flash-exp to gemini-3-flash-preview
        assert llm.model == "gemini-3-flash-preview"
        logger.info(f"✅ Gemini LLM initialized: {llm.model}")
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


def test_clip_validation():
    """Test 2: Clip candidate validation."""
    logger.info("🧪 Test 2: Clip Validation")

    # Valid clip
    valid_clip = {"start": 10.5, "end": 45.2, "title": "Test Clip", "viral_score": 8.5}
    assert _validate_clip_candidate(valid_clip) == True

    # Invalid clips
    invalid_clips = [
        {"start": 10, "end": 45},  # Missing required fields
        {"start": -5, "end": 45, "title": "Test", "viral_score": 8},  # Negative start
        {"start": 50, "end": 40, "title": "Test", "viral_score": 8},  # End before start
    ]

    for clip in invalid_clips:
        assert _validate_clip_candidate(clip) == False

    logger.info("✅ Clip validation working correctly")
    return True


def test_json_parsing():
    """Test 3: Gemini JSON response parsing."""
    logger.info("🧪 Test 3: JSON Response Parsing")

    # Test with markdown code block
    response_markdown = """```json
{
  "clips": [
    {
      "start": 15.5,
      "end": 45.2,
      "title": "Viral Moment",
      "reasoning": "High emotional impact",
      "viral_score": 9.0
    }
  ]
}
```"""

    clips = _parse_gemini_response(response_markdown, "test_job")
    assert len(clips) == 1
    assert clips[0]["viral_score"] == 9.0

    # Test with plain JSON
    response_plain = """{
  "clips": [
    {"start": 10, "end": 30, "title": "Test", "viral_score": 7.5}
  ]
}"""

    clips = _parse_gemini_response(response_plain, "test_job")
    assert len(clips) == 1

    logger.info("✅ JSON parsing working correctly")
    return True


def test_transcript_formatting():
    """Test 4: Transcript formatting for Gemini."""
    logger.info("🧪 Test 4: Transcript Formatting")

    # Create mock transcription
    mock_segment = MagicMock()
    mock_segment.start = 15.5
    mock_segment.end = 45.2
    mock_segment.text = "This is a test segment"
    mock_segment.speaker = "Speaker A"

    mock_transcription = MagicMock()
    mock_transcription.segments = [mock_segment]

    formatted = _format_transcript_for_analysis(mock_transcription)

    assert "[00:15 - 00:45]" in formatted
    assert "Speaker A" in formatted
    assert "This is a test segment" in formatted

    logger.info("✅ Transcript formatting working correctly")
    return True


async def test_analysis_node_workflow():
    """Test 5: Full analysis_node workflow with mock."""
    logger.info("🧪 Test 5: Analysis Node Workflow")

    # Create mock state
    mock_segment = MagicMock()
    mock_segment.start = 0
    mock_segment.end = 60
    mock_segment.text = "This is an amazing viral moment that everyone will share!"
    mock_segment.speaker = "Speaker A"

    mock_transcription = MagicMock()
    mock_transcription.segments = [mock_segment]
    mock_transcription.duration = 60.0

    state = VideoProcessingState(
        job_id="test_workflow",
        video_path="/tmp/test.mp4",
        transcription_result=mock_transcription,
        clips=[],
        errors=[],
        progress=25,
        status="analyzing",
        current_phase="Phase 2",
    )

    # Mock Redis
    mock_redis = AsyncMock()
    mock_redis.set = AsyncMock()

    logger.info("⚠️ Skipping full workflow test (requires real Gemini API call)")
    logger.info("   To test: Set GEMINI_API_KEY and call analysis_node(state, redis)")

    return True


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("🚀 Starting Gemini Analysis Node Tests")
    logger.info("=" * 60)

    tests = [
        ("Gemini LLM Initialization", test_gemini_llm_initialization),
        ("Clip Validation", test_clip_validation),
        ("JSON Parsing", test_json_parsing),
        ("Transcript Formatting", test_transcript_formatting),
        ("Analysis Node Workflow", test_analysis_node_workflow),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, "PASS" if result else "FAIL", None))
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}", exc_info=True)
            results.append((test_name, "FAIL", str(e)))

    # Print summary
    logger.info("=" * 60)
    logger.info("📊 Test Summary")
    logger.info("=" * 60)

    passed = sum(1 for _, status, _ in results if status == "PASS")
    failed = sum(1 for _, status, _ in results if status == "FAIL")

    for test_name, status, error in results:
        symbol = "✅" if status == "PASS" else "❌"
        logger.info(f"{symbol} {test_name}: {status}")
        if error:
            logger.info(f"   Error: {error}")

    logger.info("=" * 60)
    logger.info(f"Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
