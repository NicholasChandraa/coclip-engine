from arq import create_pool
from arq.connections import RedisSettings
from app.core.config import settings
from app.utils.logging import logger
import os
import json

async def transcribe_video_task(ctx, job_id: str, video_path: str):
    """
    Background task untuk transcribe video.

    Dipanggil oleh ARQ worker, bukan langsung dari endpoint.

    Args:
        ctx: ARQ context (bisa akses Redis connection)
        job_id: Unique job identifier
        video_path: Path ke file video yang akan di-transcribe
    """
    # LAZY IMPORT - import transcriber di sini, bukan di top level
    # Ini fix issue ctranslate2 di Windows saat ARQ worker startup
    from app.tools.transcriber import transcriber

    try:
        logger.info(f"🎬 [Job {job_id} Starting transcription: {video_path}]")

        # Update status jadi 'processing'
        await ctx['redis'].set(
            f"job:{job_id}:status",
            "processing"
        )

        await ctx['redis'].set(
            f"job:{job_id}:progress",
            "0"
        )

        # TRANSCRIBE (blocking operation, tapi di worker thread OK)
        # Worker process terpisah, jadi ga block FastAPI
        logger.info(f"🎙️[Job {job_id} Transcribing audio...]")
        result = transcriber.transcribe(video_path)

        logger.info(
            f"✅ [Job {job_id}] Transcription completed! "
            f"Language: {result['language']}, "
            f"Segments: {len(result['segments'])}"
        )

        # Save result ke redis (expire after 1 jam)
        await ctx['redis'].set(
            f"job:{job_id}:result",
            json.dumps(result),
            ex=3600 # Expire setelah 1 jam
        )
        await ctx['redis'].set(
            f"job:{job_id}:status",
            "completed"
        )
        await ctx['redis'].set(
            f"job:{job_id}:progress",
            "100"
        )

        logger.info(f"💾 [Job {job_id}] Result saved to Redis")
    
    except Exception as e:
        logger.error(f"❌ [Job {job_id}] Transcription failed: {e}")
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
    ARQ Worker startup hook.

    Dipanggil sekali saat worker process start.
    Load Whisper model di sini supaya:
    1. Model sudah ready saat job pertama masuk (no cold start)
    2. Tidak perlu load ulang untuk setiap job (hemat waktu)

    Note: Worker process TERPISAH dari FastAPI process.
    Model yang di-load di main.py TIDAK di-share ke worker.
    """
    try:
        # LAZY IMPORT - import transcriber di sini juga
        from app.tools.transcriber import transcriber

        logger.info("🔧 ARQ Worker starting up...")
        transcriber.load_model()
        logger.info("✅ Worker ready with Whisper model loaded!")
    except Exception as e:
        logger.warning(f"⚠️ Failed to preload model at startup: {e}")
        logger.info("📌 Model will be loaded on first job instead (lazy loading)")

class WorkerSettings:
    """
    ARQ Worker Configuration.

    Ini config untuk ARQ worker process.
    Dipakai saat run command: arq app.workers.transcription_worker.WorkerSettings
    """
    # List of tasks yang bisa di run oleh worker
    functions = [transcribe_video_task]

    # Startup hook - load model saat worker start
    on_startup = startup

    # Redis connection settings
    redis_settings = RedisSettings(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        database=settings.REDIS_DB
    )

    # Worker performance settings
    max_jobs = 2 # Max concurrent jobs (bisa disesuaikan dengan CPU/GPU yang dimiliki)
    job_timeout = 3600  # 1 jam timeout per job
    keep_result = 3600 # keep resultnya di redis selama 1 jam

    # Queue name (defaul queue)
    queue_name = "arq:queue"
