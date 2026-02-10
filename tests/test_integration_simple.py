"""
Simple integration test to validate graph can compile and execute.

Tests basic functionality without complex mocking.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from unittest.mock import AsyncMock
from app.graphs.video_processing_graph import create_video_processing_graph
from app.schemas.graph_schemas import create_initial_state
from app.utils.logging import logger


async def test_graph_compilation():
    """Test that graph compiles successfully."""
    logger.info("🧪 Test: Graph Compilation")

    # Mock Redis
    mock_redis = AsyncMock()
    mock_redis.set = AsyncMock()
    mock_redis.get = AsyncMock()

    # Create graph
    graph = create_video_processing_graph(mock_redis, "test_job")

    assert graph is not None
    logger.info("✅ Graph compiled successfully")
    return True


async def test_initial_state_creation():
    """Test initial state can be created."""
    logger.info("🧪 Test: Initial State Creation")

    state = create_initial_state("test_job", "/tmp/video.mp4")

    assert state["job_id"] == "test_job"
    assert state["video_path"] == "/tmp/video.mp4"
    assert state["progress"] == 0
    assert state["status"] == "queued"

    logger.info("✅ Initial state created correctly")
    return True


async def main():
    """Run simple integration tests."""
    logger.info("=" * 60)
    logger.info("🚀 Simple Integration Test")
    logger.info("=" * 60)

    tests = [
        test_graph_compilation,
        test_initial_state_creation,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            logger.error(f"❌ Test failed: {e}", exc_info=True)
            failed += 1

    logger.info("=" * 60)
    logger.info(f"Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)

    if failed == 0:
        logger.info("✅ All integration tests PASSED")
        return True
    else:
        logger.error(f"❌ {failed} test(s) FAILED")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
