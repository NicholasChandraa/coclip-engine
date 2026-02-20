from arq import create_pool
from arq.connections import RedisSettings
from app.core.config import settings
from app.utils.logging import logger
from typing import Optional
import os
import json
import asyncio


async def process_video_task(
    ctx, job_id: str, video_path: str, source: str = "upload", youtube_url: str = "", user_id: str = ""
):
    """
    Background task for full pipeline processing using LangGraph.

    This is now a thin wrapper around LangGraph orchestration.
    The actual pipeline logic is in app.graphs.video_processing_graph.

    Full Pipeline Progress:
      Phase 0: Download (yt-dlp, YouTube only)
      Phase 1: Transcription (WhisperX)  →  0% - 25%
      Phase 2: Content Analysis (Gemini) → 25% - 50%
      Phase 3: Video Editing (FFmpeg)    → 50% - 80%
      Phase 4: Finalization              → 80% - 100%

    Args:
        ctx: ARQ context (can access Redis connection)
        job_id: Unique job identifier
        video_path: Path to video file to process (empty if YouTube)
        source: "upload" or "youtube"
        youtube_url: YouTube URL (only when source="youtube")
    """
    # Import LangGraph pipeline
    from app.graphs import run_video_processing_pipeline

    # Check if job was aborted before starting (e.g. retry of cancelled job)
    job_status = await ctx["redis"].get(f"job:{job_id}:status")
    if job_status and job_status.decode() == "aborted":
        logger.info(f"⏭️ [Job {job_id}] Skipping aborted job")
        return

    try:
        # Phase 0: Download YouTube video if needed
        if source == "youtube" and youtube_url:
            logger.info(f"📥 [Job {job_id}] Downloading YouTube video: {youtube_url}")
            await ctx["redis"].set(f"job:{job_id}:status", "downloading")

            from app.utils.downloader import download_video

            result = await download_video(
                url=youtube_url,
                output_dir=settings.TEMP_DIR,
                job_id=job_id,
            )
            video_path = result.file_path
            logger.info(
                f"✅ [Job {job_id}] YouTube download complete: "
                f"'{result.title}' ({result.file_size / 1024 / 1024:.1f} MB)"
            )

        logger.info(f"🎬 [Job {job_id}] Starting LangGraph pipeline: {video_path}")
        file_size = os.path.getsize(video_path) / 1024 / 1024 if os.path.exists(video_path) else 0
        logger.info(f"📁 [Job {job_id}] Video file: {file_size:.1f} MB")

        # Execute LangGraph pipeline
        import time as _time
        _pipeline_start = _time.time()

        final_state = await run_video_processing_pipeline(
            redis=ctx["redis"],
            job_id=job_id,
            video_path=video_path,
            source=source,
            source_url=youtube_url,
            user_id=user_id,
        )

        _pipeline_elapsed = _time.time() - _pipeline_start

        # Check final status
        final_status = final_state.get("status", "unknown")
        errors = final_state.get("errors", [])
        clips_count = len(final_state.get("clips", []))

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
                logger.info(
                    f"✅ [Job {job_id}] Pipeline completed in {_pipeline_elapsed:.1f}s, "
                    f"{clips_count} clips generated"
                )

    except Exception as e:
        logger.error(f"❌ [Job {job_id}] Pipeline failed: {e}", exc_info=True)
        await ctx["redis"].set(f"job:{job_id}:status", "failed")
        await ctx["redis"].set(f"job:{job_id}:error", str(e))
        raise

    finally:
        # Ensure all GPU models are freed even if pipeline fails
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"[Job {job_id}] CUDA cache cleared")
        except Exception:
            pass

        # Unload any remaining models to free VRAM for next task
        try:
            from app.tools.transcriber import transcriber
            transcriber.unload_all()
        except Exception:
            pass

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
    ARQ Worker startup hook.

    Dipanggil sekali saat worker process start.
    Models are loaded on-demand per job (lazy loading) to save VRAM.
    """
    logger.info(
        f"🔧 ARQ Worker ready! "
        f"Diarization: {'enabled' if settings.ENABLE_DIARIZATION else 'disabled'}, "
        f"Models: lazy loading (on-demand)"
    )


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
