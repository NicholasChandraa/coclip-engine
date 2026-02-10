"""
Transcription node for LangGraph video processing pipeline.

Implements Phase 1: WhisperX transcription (0% → 25%)
Uses 2026 Command API pattern for combined state update + routing.
"""

from typing import Literal
from langgraph.types import Command
from app.schemas.graph_schemas import VideoProcessingState
from app.tools.transcriber import transcriber
from app.utils.progress_tracker import create_progress_tracker
from app.utils.logging import logger
from redis import asyncio as aioredis


async def transcription_node(
    state: VideoProcessingState, redis: aioredis.Redis
) -> Command[Literal["analysis", "finalization"]]:
    """
    Phase 1: Transcription node using WhisperX.

    Pipeline steps (0% → 25%):
    1. Load audio (0% → 2%)
    2. WhisperX transcription (2% → 10%)
    3. Alignment for word-level timestamps (10% → 18%)
    4. Diarization for speaker detection (18% → 23%) [if enabled]
    5. Format result (23% → 25%)

    Args:
        state: Current LangGraph state
        redis: Async Redis connection for progress tracking

    Returns:
        Command with updated state and routing decision
    """
    job_id = state["job_id"]
    video_path = state["video_path"]

    # Initialize progress tracker
    tracker = create_progress_tracker(redis, job_id)

    try:
        logger.info(f"🎬 [Job {job_id}] Starting Phase 1: Transcription")
        await tracker.update_progress(0, "transcribing", "Phase 1: Transcription")

        # Step 1: Load audio (0% → 2%)
        logger.info(f"📂 [Job {job_id}] Loading audio from video...")
        audio = transcriber.load_audio(video_path)
        await tracker.update_progress(2, phase="Audio loaded")

        # Step 2: WhisperX Transcription (2% → 10%)
        logger.info(f"🎤 [Job {job_id}] Running WhisperX transcription...")
        raw_result = transcriber.step_transcribe(audio)
        language = raw_result["language"]
        await tracker.update_progress(
            10, phase=f"Transcription complete (lang: {language})"
        )

        # Step 3: Alignment (10% → 18%)
        logger.info(f"🎯 [Job {job_id}] Aligning for word-level timestamps...")
        aligned_result = transcriber.step_align(raw_result["segments"], audio, language)
        await tracker.update_progress(18, phase="Word alignment complete")

        # Step 4: Diarization (18% → 23%) [conditional]
        from app.core.config import settings

        if settings.ENABLE_DIARIZATION:
            logger.info(f"👥 [Job {job_id}] Running speaker diarization...")
            final_result = transcriber.step_diarize(audio, aligned_result)
            await tracker.update_progress(23, phase="Diarization complete")
        else:
            logger.info(f"⏭️ [Job {job_id}] Skipping diarization (disabled)")
            final_result = aligned_result
            await tracker.update_progress(23, phase="Diarization skipped")

        # Step 5: Format result (23% → 25%)
        logger.info(f"📝 [Job {job_id}] Formatting transcription result...")
        transcription_result = transcriber.format_result(final_result, language)
        await tracker.update_progress(25, "analyzing", "Phase 1 complete")

        logger.info(
            f"✅ [Job {job_id}] Phase 1 COMPLETE! "
            f"Language: {language}, Segments: {len(transcription_result.segments)}"
        )

        # Check if we have valid transcription to proceed
        if transcription_result.segments and len(transcription_result.segments) > 0:
            # Command API: Update state + route to analysis
            return Command(
                update={
                    "transcription_result": transcription_result,
                    "progress": 25,
                    "status": "analyzing",
                    "current_phase": "Phase 2: Content Analysis",
                },
                goto="analysis",
            )
        else:
            # No content found, skip to finalization
            logger.warning(
                f"⚠️ [Job {job_id}] No transcription segments found, skipping analysis"
            )
            return Command(
                update={
                    "transcription_result": transcription_result,
                    "progress": 80,
                    "status": "finalizing",
                    "current_phase": "Phase 4: Finalization",
                    "errors": ["No transcription segments found"],
                },
                goto="finalization",
            )

    except Exception as e:
        error_msg = f"Transcription failed: {str(e)}"
        logger.error(f"❌ [Job {job_id}] {error_msg}", exc_info=True)
        await tracker.set_error(error_msg)

        # Return error state and skip to finalization
        return Command(
            update={
                "progress": 0,
                "status": "failed",
                "current_phase": "Failed",
                "errors": [error_msg],
            },
            goto="finalization",  # Go to finalization for cleanup
        )
