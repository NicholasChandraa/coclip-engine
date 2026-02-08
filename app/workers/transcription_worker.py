from arq import create_pool
from arq.connections import RedisSettings
from app.core.config import settings
from app.utils.logging import logger
import os
import json


async def process_video_task(ctx, job_id: str, video_path: str):
    """
    Background task untuk full pipeline processing.

    Full Pipeline Progress:
      Phase 1: Transcription (WhisperX)  →  0% - 25%
      Phase 2: Content Analysis (Gemini) → 25% - 50%  [TODO]
      Phase 3: Video Editing (FFmpeg)    → 50% - 80%  [TODO]
      Phase 4: Finalization              → 80% - 100% [TODO]

    Args:
        ctx: ARQ context (bisa akses Redis connection)
        job_id: Unique job identifier
        video_path: Path ke file video yang akan di-process
    """
    # LAZY IMPORT
    from app.tools.transcriber import transcriber

    # Progress update helper
    async def update_progress(progress: int, status: str = None):
        """Helper untuk update progress dan optional status di Redis."""
        await ctx['redis'].set(f"job:{job_id}:progress", str(progress))
        if status:
            await ctx['redis'].set(f"job:{job_id}:status", status)

    try:
        logger.info(f"🎬 [Job {job_id}] Starting pipeline: {video_path}")

        # ================================================================
        # PHASE 1: TRANSCRIPTION (WhisperX) — 0% → 25%
        # ================================================================
        await update_progress(0, "transcribing")
        logger.info(f"📊 [Job {job_id}] Progress: 0% — Phase 1: Transcription started")

        # Load audio
        audio = transcriber.load_audio(video_path)
        await update_progress(2)
        logger.info(f"📊 [Job {job_id}] Progress: 2% — Audio loaded")

        # WhisperX Step 1: Transcription
        raw_result = transcriber.step_transcribe(audio)
        language = raw_result["language"]
        await update_progress(10)
        logger.info(f"📊 [Job {job_id}] Progress: 10% — Transcription done (lang: {language})")

        # WhisperX Step 2: Alignment
        aligned_result = transcriber.step_align(raw_result["segments"], audio, language)
        await update_progress(18)
        logger.info(f"📊 [Job {job_id}] Progress: 18% — Alignment done")

        # WhisperX Step 3: Diarization
        if settings.ENABLE_DIARIZATION:
            final_result = transcriber.step_diarize(audio, aligned_result)
            await update_progress(23)
            logger.info(f"📊 [Job {job_id}] Progress: 23% — Diarization done")
        else:
            final_result = aligned_result
            await update_progress(23)
            logger.info(f"📊 [Job {job_id}] Progress: 23% — Diarization skipped (disabled)")

        # Format & save transcription result
        transcription_result = transcriber.format_result(final_result, language)
        await ctx['redis'].set(
            f"job:{job_id}:transcription",
            transcription_result.model_dump_json(),
            ex=3600
        )
        await update_progress(25)
        logger.info(
            f"📊 [Job {job_id}] Progress: 25% — Phase 1 COMPLETE! "
            f"Language: {language}, Segments: {len(transcription_result.segments)}"
        )

        # ================================================================
        # PHASE 2: CONTENT ANALYSIS (Gemini) — 25% → 50%  [TODO]
        # ================================================================
        await update_progress(25, "analyzing")
        logger.info(f"📊 [Job {job_id}] Progress: 25% — Phase 2: Content Analysis [TODO]")

        # TODO: Implement Gemini content analysis
        # clips = analyzer.analyze(transcription_result)
        await update_progress(50)
        logger.info(f"📊 [Job {job_id}] Progress: 50% — Phase 2 SKIPPED (not implemented)")

        # ================================================================
        # PHASE 3: VIDEO EDITING (FFmpeg) — 50% → 80%  [TODO]
        # ================================================================
        await update_progress(50, "editing")
        logger.info(f"📊 [Job {job_id}] Progress: 50% — Phase 3: Video Editing [TODO]")

        # TODO: Implement FFmpeg video editing
        # editor.cut_clips(video_path, clips)
        # editor.burn_subtitles(clips, transcription_result)
        await update_progress(80)
        logger.info(f"📊 [Job {job_id}] Progress: 80% — Phase 3 SKIPPED (not implemented)")

        # ================================================================
        # PHASE 4: FINALIZATION — 80% → 100%  [TODO]
        # ================================================================
        await update_progress(80, "finalizing")
        logger.info(f"📊 [Job {job_id}] Progress: 80% — Phase 4: Finalization")

        # Save final result (untuk sekarang = transcription result saja)
        await ctx['redis'].set(
            f"job:{job_id}:result",
            transcription_result.model_dump_json(),
            ex=3600
        )

        # TODO: Generate thumbnails
        # TODO: Save clip metadata to DB
        # TODO: Notify Golang (webhook/callback)

        await update_progress(100, "completed")
        logger.info(f"📊 [Job {job_id}] Progress: 100% — Pipeline COMPLETE!")

    except Exception as e:
        logger.error(f"❌ [Job {job_id}] Pipeline failed: {e}")
        await ctx['redis'].set(f"job:{job_id}:status", "failed")
        await ctx['redis'].set(f"job:{job_id}:error", str(e))
        raise

    finally:
        # Cleanup temp file
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                logger.info(f"🗑️ [Job {job_id}] Cleaned up temp file: {video_path}")
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
        conn_timeout=30,       # Connection timeout (seconds)
        conn_retries=5,        # Retry connect on failure
        conn_retry_delay=1,    # Delay between retries (seconds)
    )

    # Worker performance settings
    max_jobs = 2  # Max concurrent jobs (adjust based on GPU memory)
    job_timeout = 7200  # 2 hours timeout untuk video panjang (1-2 jam)
    keep_result = 3600  # Keep result in Redis for 1 hour

    # Queue name
    queue_name = "arq:queue"
