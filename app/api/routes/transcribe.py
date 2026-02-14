from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from app.core.config import settings
from app.utils.logging import logger
from app.schemas.transcription import (
    SegmentPreview,
    YouTubeRequest,
    TranscribeAsyncResponse,
    JobStatusResponse,
    TranscriptionResult,
)
from arq import create_pool
from arq.connections import RedisSettings
from redis import asyncio as aioredis
import os
import uuid
import json


# ============= Router Setup =============
router = APIRouter()


# Helper Function
async def get_redis_connection():
    """Create Redis connection with retry settings."""
    return await aioredis.from_url(
        f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
        socket_timeout=10,
        socket_keepalive=True,
        retry_on_timeout=True,
    )


# Endpoint
@router.post("/transcribe-async", response_model=TranscribeAsyncResponse)
async def transcribe_async(file: UploadFile = File(...)):
    """
    Async transcription endpoint - Upload file dan return job_id instantly.

    Flow:
    1. Validate file type
    2. Save file to temp storage (streaming)
    3. Enqueue job to ARQ
    4. Return job_id immediately (no waiting!)

    Frontend bisa polling ke /transcribe/status/{job_id} untuk cek progress.

    Args:
        file: Upload file (audio/video) via multipart/form-data

    Returns:
        TranscribeAsyncResponse dengan job_id dan status "queued"

    Raises:
        HTTPException 400: Invalid file type or filename missing
        HTTPException 500: Failed to enqueue job
    """

    # Step 1: Validasi File Type
    logger.info(f"📤 Upload request: {file.filename} ({file.content_type})")

    if not file.content_type or (
        not file.content_type.startswith("audio/")
        and not file.content_type.startswith("video/")
    ):
        logger.warning(f"❌ Upload rejected: invalid type {file.content_type}")
        raise HTTPException(
            status_code=400, detail="File must be audio or video format"
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Step 2: Generate Job ID & Save File
    job_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[
        1
    ]  # split file name misal audio.mp3 maka akan diambil ".mp3"
    temp_path = os.path.join(settings.TEMP_DIR, f"{job_id}{file_ext}")

    # Memastikan temp directory ada
    os.makedirs(settings.TEMP_DIR, exist_ok=True)

    try:
        # Streaming write untuk file besar (8kb chunks)
        logger.info(f"📁 [Job {job_id}] Saving uploaded file: {file.filename}")
        with open(temp_path, "wb") as f:
            while chunk := await file.read(8192):  # 8kb chunks
                f.write(chunk)

        file_size = os.path.getsize(temp_path)
        logger.info(f"✅ [Job {job_id} File saved: {temp_path} ({file_size} bytes)]")

        # Step 3: Enqueue Job ke ARQ
        redis_pool = await create_pool(
            RedisSettings(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                database=settings.REDIS_DB,
            )
        )

        # Enqueue job to ARQ
        job = await redis_pool.enqueue_job(
            "process_video_task", job_id, temp_path  # Function name di worker
        )

        # Set initial status di redis
        redis = await get_redis_connection()
        await redis.set(f"job:{job_id}:status", "queued")
        await redis.set(f"job:{job_id}:progress", "0")
        await redis.set(f"job:{job_id}:filename", file.filename)
        await redis.close()

        logger.info(f"✅ [Job {job_id}] Enqueued to ARQ worker")

        # Step 4: Return Response
        return TranscribeAsyncResponse(
            job_id=job_id,
            status="queued",
            message=f"Transcription job queued successfully for {file.filename}",
        )

    except Exception as e:
        logger.error(f"❌ [Job {job_id}] Failed to enqueue: {e}")
        # Cleanup file kalau gagal enqueue
        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise HTTPException(
            status_code=500, detail=f"Failed to enqueue transcription job: {str(e)}"
        )


@router.post("/transcribe-youtube", response_model=TranscribeAsyncResponse)
async def transcribe_youtube(request: YouTubeRequest):
    """
    Submit YouTube URL for async video processing.

    Flow:
    1. Validate YouTube URL format
    2. Generate job_id
    3. Enqueue job to ARQ (download happens in worker)
    4. Return job_id immediately

    Args:
        request: YouTubeRequest with url field

    Returns:
        TranscribeAsyncResponse dengan job_id dan status "queued"
    """
    from app.utils.downloader import validate_youtube_url

    url = request.url.strip()
    logger.info(f"🔗 YouTube request: {url}")

    # Validate URL
    if not validate_youtube_url(url):
        logger.warning(f"❌ Invalid YouTube URL: {url}")
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL. Supported: youtube.com/watch, youtu.be, youtube.com/shorts",
        )

    job_id = str(uuid.uuid4())

    try:
        # Enqueue job to ARQ — download will happen in worker
        redis_pool = await create_pool(
            RedisSettings(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                database=settings.REDIS_DB,
            )
        )

        # video_path is empty string — worker will download first
        await redis_pool.enqueue_job(
            "process_video_task",
            job_id,
            "",  # no video_path yet, worker downloads it
            "youtube",  # source
            url,  # youtube_url
        )

        # Set initial status
        redis = await get_redis_connection()
        await redis.set(f"job:{job_id}:status", "queued")
        await redis.set(f"job:{job_id}:progress", "0")
        await redis.set(f"job:{job_id}:filename", url)
        await redis.close()

        logger.info(f"✅ [Job {job_id}] YouTube job enqueued: {url}")

        return TranscribeAsyncResponse(
            job_id=job_id,
            status="queued",
            message=f"YouTube video queued for processing: {url}",
        )

    except Exception as e:
        logger.error(f"❌ [Job {job_id}] Failed to enqueue YouTube job: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to enqueue YouTube job: {str(e)}",
        )


@router.get("/transcribe/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Check job status endpoint - untuk polling dari frontend.

    Frontend bisa call endpoint ini setiap 2-5 detik untuk cek progress.

    Args:
        job_id: Job ID yang di-return dari /transcribe-async

    Returns:
        JobStatusResponse dengan status, progress, dan result (kalau completed)

    Raises:
        HTTPException 404: Job not found

    Status values:
        - queued: Job in queue, waiting for worker
        - processing: Worker sedang process
        - completed: Processing selesai, result available
        - failed: Error occurred
    """
    redis = await get_redis_connection()

    try:
        # Get status from Redis
        status = await redis.get(f"job:{job_id}:status")

        if not status:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        status = status.decode()

        # Get progress
        progress = await redis.get(f"job:{job_id}:progress")
        progress = int(progress.decode()) if progress else 0

        # Build response
        response = JobStatusResponse(job_id=job_id, status=status, progress=progress)

        # Kalau completed, include result
        if status == "completed":
            result = await redis.get(f"job:{job_id}:result")
            if result:
                result_data = json.loads(result.decode())

                # New LangGraph structure has clips, not segments
                response.result = {
                    "language": result_data.get("language", "unknown"),
                    "duration": result_data.get("duration", 0),
                    "total_segments": result_data.get("total_segments", 0),
                    "clips_count": result_data.get("clips_count", 0),
                    "clips": result_data.get("clips", []),
                    "status": result_data.get("status", "completed"),
                }

        # Kalau failed, include error message
        if status == "failed":
            error = await redis.get(f"job:{job_id}:error")
            if error:
                response.error = error.decode()

        return response

    finally:
        await redis.close()


@router.get("/transcribe/result/{job_id}")
async def get_full_result(job_id: str):
    """
    Get full transcription result (all segments).

    Call endpoint ini kalau mau ambil SEMUA segments, bukan cuma preview.

    Args:
    job_id: Job ID

    Returns:
    TranscriptionResult dengan semua segments

    Raises:
    HTTPException 404: Job not found or not completed
    """

    redis = await get_redis_connection()

    try:
        status = await redis.get(f"job:{job_id}:status")

        if not status:
            raise HTTPException(status_code=404, detail="Job not found")

        status = status.decode()

        if status != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Job is not completed yet. Current status: {status}",
            )

        # Get result from Redis
        result = await redis.get(f"job:{job_id}:result")

        if not result:
            raise HTTPException(status_code=404, detail="Result not found")

        result_data = json.loads(result.decode())

        # Get full transcription from separate key
        transcription = await redis.get(f"job:{job_id}:transcription")
        transcription_data = None
        if transcription:
            transcription_data = json.loads(transcription.decode())

        # Return comprehensive result
        return {
            "job_id": result_data.get("job_id"),
            "language": result_data.get("language", "unknown"),
            "duration": result_data.get("duration", 0),
            "total_segments": result_data.get("total_segments", 0),
            "clips_count": result_data.get("clips_count", 0),
            "clips": result_data.get("clips", []),
            "transcription": transcription_data,
            "status": result_data.get("status", "completed"),
        }

    finally:
        await redis.close()


@router.get("/transcribe/clips/{job_id}/{clip_number}")
async def download_clip(job_id: str, clip_number: int):
    """
    Download a generated clip file.

    Args:
        job_id: Job ID
        clip_number: Clip number (1-based)

    Returns:
        MP4 file as download
    """
    clip_path = os.path.join(settings.CLIPS_DIR, job_id, f"clip_{clip_number}.mp4")

    if not os.path.exists(clip_path):
        logger.warning(f"⚠️ Clip download 404: job={job_id}, clip={clip_number}")
        raise HTTPException(
            status_code=404,
            detail=f"Clip {clip_number} for job {job_id} not found",
        )

    logger.info(f"📥 Clip download: job={job_id}, clip={clip_number}")
    return FileResponse(
        path=clip_path,
        media_type="video/mp4",
        filename=f"{job_id}_clip_{clip_number}.mp4",
    )


# ============= Database Endpoints =============


@router.get("/jobs")
async def list_jobs(limit: int = 20, offset: int = 0):
    """
    List all processed jobs from database (persistent).

    Args:
        limit: Max results (default 20)
        offset: Pagination offset

    Returns:
        List of job records with clip counts
    """
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    from app.core.database import async_session
    from app.models import Job

    try:
        async with async_session() as session:
            stmt = (
                select(Job)
                .options(selectinload(Job.clips))
                .order_by(Job.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(stmt)
            jobs = result.scalars().all()

            return {
                "total": len(jobs),
                "jobs": [job.to_dict() for job in jobs],
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/jobs/{job_id}")
async def get_job_detail(job_id: str):
    """
    Get job detail with all clips from database.

    Args:
        job_id: Job ID

    Returns:
        Job record with full clip metadata
    """
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    from app.core.database import async_session
    from app.models import Job

    try:
        async with async_session() as session:
            stmt = select(Job).options(selectinload(Job.clips)).where(Job.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()

            if not job:
                raise HTTPException(status_code=404, detail="Job not found in database")

            job_data = job.to_dict()
            job_data["clips"] = [clip.to_dict() for clip in job.clips]
            return job_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
