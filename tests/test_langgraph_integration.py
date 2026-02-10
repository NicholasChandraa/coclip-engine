"""
Test script for LangGraph video processing pipeline.

This script tests:
1. State schema creation
2. Graph compilation
3. Basic state flow through nodes
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from app.schemas.graph_schemas import create_initial_state
from app.graphs import create_video_processing_graph
from unittest.mock import AsyncMock
from app.utils.logging import logger


async def test_graph_compilation():
    """Test that graph compiles without errors."""
    logger.info("🧪 Test 1: Graph Compilation")

    # Mock Redis connection
    mock_redis = AsyncMock()
    mock_redis.set = AsyncMock()
    mock_redis.get = AsyncMock()

    # Create graph
    job_id = "test_job_123"
    graph = create_video_processing_graph(mock_redis, job_id)

    logger.info("✅ Graph compiled successfully!")
    return True


async def test_state_creation():
    """Test state schema creation."""
    logger.info("🧪 Test 2: State Creation")

    # Create initial state
    state = create_initial_state(
        job_id="test_job_456", video_path="/tmp/test_video.mp4"
    )

    # Validate state structure
    assert state["job_id"] == "test_job_456"
    assert state["video_path"] == "/tmp/test_video.mp4"
    assert state["progress"] == 0
    assert state["status"] == "queued"
    assert state["clips"] == []
    assert state["errors"] == []

    logger.info("✅ State creation successful!")
    return True


async def test_graph_visualization():
    """Test graph structure visualization."""
    logger.info("🧪 Test 3: Graph Structure")

    # Mock Redis connection
    mock_redis = AsyncMock()

    # Create graph
    job_id = "test_job_viz"
    graph = create_video_processing_graph(mock_redis, job_id)

    # Try to get graph structure (if LangGraph supports it)
    try:
        # Get nodes
        nodes = graph.nodes if hasattr(graph, "nodes") else "N/A"
        logger.info(f"Graph nodes: {nodes}")

        logger.info("✅ Graph structure validated!")
    except Exception as e:
        logger.warning(f"⚠️ Could not visualize graph: {e}")

    return True


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("🚀 Starting LangGraph Integration Tests")
    logger.info("=" * 60)

    tests = [
        ("Graph Compilation", test_graph_compilation),
        ("State Creation", test_state_creation),
        ("Graph Structure", test_graph_visualization),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, "PASS", None))
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
