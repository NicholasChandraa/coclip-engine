"""
LangGraph video processing graph builder.

Uses 2026 best practices:
- StateGraph with START/END constants
- MemorySaver checkpointer for development
- Async node execution with ainvoke/astream support
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from redis import asyncio as aioredis

from app.schemas.graph_schemas import VideoProcessingState
from app.graphs.nodes import (
    transcription_node,
    analysis_node,
    editing_node,
    finalization_node,
    hook_generation_node,
)
from app.utils.logging import logger


def create_video_processing_graph(redis: aioredis.Redis, job_id: str):
    """
    Create and compile LangGraph for video processing pipeline.

    Graph flow:
    START → transcription → analysis → hook_generation → editing → finalization → END

    With conditional routing:
    - If transcription fails or has no content → skip to finalization
    - If analysis finds no clips → skip editing

    Args:
        redis: Async Redis connection for progress tracking
        job_id: Unique job identifier for logging/tracking

    Returns:
        Compiled LangGraph ready for execution
    """
    logger.info(f"🔧 [Job {job_id}] Building LangGraph processing pipeline...")

    # Initialize StateGraph with our state schema
    graph = StateGraph(VideoProcessingState)

    # Create async wrapper functions with redis injection
    # LangGraph nodes need to be proper async functions, not lambdas
    async def transcription_wrapper(state):
        return await transcription_node(state, redis)

    async def analysis_wrapper(state):
        return await analysis_node(state, redis)

    async def hook_generation_wrapper(state):
        return await hook_generation_node(state, redis)

    async def editing_wrapper(state):
        return await editing_node(state, redis)

    async def finalization_wrapper(state):
        return await finalization_node(state, redis)

    # Add nodes with async wrappers
    graph.add_node("transcription", transcription_wrapper)
    graph.add_node("analysis", analysis_wrapper)
    graph.add_node("hook_generation", hook_generation_wrapper)
    graph.add_node("editing", editing_wrapper)
    graph.add_node("finalization", finalization_wrapper)

    # Define edges using START/END constants
    graph.add_edge(START, "transcription")

    # Conditional edges are handled by Command API in nodes
    # transcription_node returns Command(goto="analysis" or "finalization")
    # analysis_node returns Command(goto="hook_generation" or "finalization")
    # hook_generation_node returns Command(goto="editing")
    # editing_node returns Command(goto="finalization")

    # finalization is terminal → END
    graph.add_edge("finalization", END)

    # Compile with MemorySaver checkpointer (2026 pattern)
    # For production, replace with AsyncPostgresSaver or AsyncRedisSaver
    checkpointer = MemorySaver()
    compiled_graph = graph.compile(checkpointer=checkpointer)

    logger.info(f"✅ [Job {job_id}] LangGraph pipeline compiled successfully")

    return compiled_graph


async def run_video_processing_pipeline(
    redis: aioredis.Redis,
    job_id: str,
    video_path: str,
    source: str = "upload",
    source_url: str = "",
    user_id: str = "",
):
    """
    Execute video processing pipeline with LangGraph.

    Args:
        redis: Async Redis connection
        job_id: Unique job identifier
        video_path: Path to video file
        source: "upload" or "youtube"
        source_url: Original YouTube URL (if source="youtube")

    Returns:
        Final state after pipeline completion
    """
    from app.schemas.graph_schemas import create_initial_state

    # Create graph
    graph = create_video_processing_graph(redis, job_id)

    # Create initial state
    initial_state = create_initial_state(job_id, video_path)
    initial_state["source"] = source
    if source_url:
        initial_state["source_url"] = source_url
    if user_id:
        initial_state["user_id"] = user_id

    # Config with thread_id for checkpointing
    config = {"configurable": {"thread_id": job_id}}

    logger.info(f"🚀 [Job {job_id}] Starting LangGraph pipeline execution...")

    # Standard execution mode
    final_state = await graph.ainvoke(initial_state, config=config)

    final_status = final_state.get("status", "unknown")
    final_clips = len(final_state.get("clips", []))
    final_errors = len(final_state.get("errors", []))
    logger.info(
        f"🏁 [Job {job_id}] Pipeline finished: "
        f"status={final_status}, clips={final_clips}, errors={final_errors}"
    )

    return final_state
