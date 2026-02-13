from arq import create_pool
from arq.connections import RedisSettings
from app.core.config import settings
from app.utils.logging import logger
from typing import Optional
import os
import json


async def process_video_task(ctx, job_id: str, video_path: str):
    """
    Background task for full pipeline processing using LangGraph.

    This is now a thin wrapper around LangGraph orchestration.
    The actual pipeline logic is in app.graphs.video_processing_graph.

    Full Pipeline Progress:
      Phase 1: Transcription (WhisperX)  →  0% - 25%
      Phase 2: Content Analysis (Gemini) → 25% - 50%  [TODO]
      Phase 3: Video Editing (FFmpeg)    → 50% - 80%  [TODO]
      Phase 4: Finalization              → 80% - 100%

    Args:
        ctx: ARQ context (can access Redis connection)
        job_id: Unique job identifier
        video_path: Path to video file to process
    """
    # Import LangGraph pipeline
    from app.graphs import run_video_processing_pipeline

    try:
        logger.info(f"🎬 [Job {job_id}] Starting LangGraph pipeline: {video_path}")

        # Execute LangGraph pipeline
        # Use non-streaming mode (ainvoke) for simplicity
        # For real-time updates, switch to use_streaming=True
        final_state = await run_video_processing_pipeline(
            redis=ctx["redis"],
            job_id=job_id,
            video_path=video_path,
        )

        # Check final status
        final_status = final_state.get("status", "unknown")
        errors = final_state.get("errors", [])

        if final_status == "failed":
            error_msg = errors[0] if errors else "Pipeline failed with unknown error"
            logger.error(f"❌ [Job {job_id}] Pipeline failed: {error_msg}")
            raise Exception(error_msg)
        else:
            if errors:
                logger.warning(
                    f"⚠️ [Job {job_id}] Pipeline completed with {len(errors)} clip errors "
                    f"(some clips may have failed)"
                )
            else:
                logger.info(f"✅ [Job {job_id}] Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"❌ [Job {job_id}] Pipeline failed: {e}", exc_info=True)
        await ctx["redis"].set(f"job:{job_id}:status", "failed")
        await ctx["redis"].set(f"job:{job_id}:error", str(e))
        raise

    finally:
        # Cleanup temp file
        # Note: finalization_node also does cleanup, but we keep this as fallback
        if os.path.exists(video_path):
            try:
                # Only delete if it's in temp directory
                if "temp" in video_path.lower() or "tmp" in video_path.lower():
                    os.remove(video_path)
                    logger.info(f"🗑️ [Job {job_id}] Cleaned up temp file: {video_path}")
                else:
                    logger.info(f"⏭️ [Job {job_id}] Skipping cleanup (not a temp file)")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ [Job {job_id}] Failed to cleanup: {cleanup_error}")


async def startup(ctx):
    """
    ARQ Worker startup hook - preload WhisperX model.

    Dipanggil sekali saat worker process start.
    """
    try:
        from app.tools.transcriber import transcriber

        logger.info("🔧 ARQ Worker starting up...")
        transcriber.load_model()
        logger.info(
            f"✅ Worker ready with WhisperX model loaded! "
            f"Diarization: {'enabled' if settings.ENABLE_DIARIZATION else 'disabled'}"
        )
    except Exception as e:
        logger.warning(f"⚠️ Failed to preload model at startup: {e}")
        logger.info("📌 Model will be loaded on first job instead (lazy loading)")


class WorkerSettings:
    """
    ARQ Worker Configuration.

    Dipakai saat run command: arq app.workers.transcription_worker.WorkerSettings
    """

    # List of tasks yang bisa di-run oleh worker
    functions = [process_video_task]

    # Startup hook - load model saat worker start
    on_startup = startup

    # Redis connection settings
    redis_settings = RedisSettings(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        database=settings.REDIS_DB,
        conn_timeout=30,  # Connection timeout (seconds)
        conn_retries=5,  # Retry connect on failure
        conn_retry_delay=1,  # Delay between retries (seconds)
    )

    # Worker performance settings
    max_jobs = 2  # Max concurrent jobs (adjust based on GPU memory)
    job_timeout = 7200  # 2 hours timeout untuk video panjang (1-2 jam)
    keep_result = 3600  # Keep result in Redis for 1 hour

    # Queue name
    queue_name = "arq:queue"
