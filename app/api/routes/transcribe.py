from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.core.config import settings
from app.utils.logging import logger
from arq import create_pool
from arq.connections import RedisSettings
from redis import asyncio as aioredis
import os
import uuid
import json
from typing import List, Optional

# Pydantic Model untuk Type Safety
class SegmentPreview(BaseModel):
    """
    Model untuk preview segment hasil transcription.
    Berisi timing (start/end) dan text dari satu segment audio.
    """
    start: float  # Waktu mulai segment dalam detik
    end: float # waktu akhir segment dalam detik
    text: str # text hasil transcribe untuk segment ini

class TranscribeAsyncResponse(BaseModel):
    """Response untuk async transcription endpoint."""
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    """Response untuk job status check."""
    job_id: str
    status: str  # queued/processing/completed/failed
    progress: int  # 0-100
    result: Optional[dict] = None
    error: Optional[str] = None

class TranscriptionResult(BaseModel):
    """Full transcription result with all segments."""
    language: str
    duration: float
    total_segments: int
    segments: List[SegmentPreview]


# Router Setup
router = APIRouter()


# Helper Function
async def get_redis_connection():
    """Create Redis connection."""
    return await aioredis.from_url(
        f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"
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
    if not file.content_type or (
        not file.content_type.startswith("audio/") and
        not file.content_type.startswith("video/")
    ):
        raise HTTPException(
            status_code=400,
            detail="File must be audio or video format"
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )
    
    # Step 2: Generate Job ID & Save File
    job_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1] # split file name misal audio.mp3 maka akan diambil ".mp3"
    temp_path = os.path.join(settings.TEMP_DIR, f"{job_id}{file_ext}")

    # Memastikan temp directory ada
    os.makedirs(settings.TEMP_DIR, exist_ok=True)

    try:
        # Streaming write untuk file besar (8kb chunks)
        logger.info(f"📁 [Job {job_id}] Saving uploaded file: {file.filename}")
        with open(temp_path, "wb") as f:
            while chunk := await file.read(8192): # 8kb chunks
                f.write(chunk)
        
        file_size = os.path.getsize(temp_path)
        logger.info(f"✅ [Job {job_id} File saved: {temp_path} ({file_size} bytes)]")

        # Step 3: Enqueue Job ke ARQ
        redis_pool = await create_pool(
            RedisSettings(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                database=settings.REDIS_DB
            )
        )

        # Enqueue job to ARQ
        job = await redis_pool.enqueue_job(
            'transcribe_video_task', # Function name di worker
            job_id,
            temp_path
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
            message=f"Transcription job queued successfully for {file.filename}"
        )
    
    except Exception as e:
        logger.error(f"❌ [Job {job_id}] Failed to enqueue: {e}")
        # Cleanup file kalau gagal enqueue
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to enqueue transcription job: {str(e)}"
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
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
        
        status = status.decode()

        # Get progress
        progress = await redis.get(f"job:{job_id}:progress")
        progress = int(progress.decode()) if progress else 0

        # Build response
        response = JobStatusResponse(
            job_id=job_id,
            status=status,
            progress=progress
        )

        # Kalau completed, include result
        if status == "completed":
            result = await redis.get(f"job:{job_id}:result")
            if result:
                result_data = json.loads(result.decode())

                # Convert to TranscriptionResult format
                segments_preview = [
                    SegmentPreview(
                        start=seg["start"],
                        end=seg["end"],
                        text=seg["text"]
                    )
                    for seg in result_data["segments"][:3]  # First 3 segments preview
                ]

                response.result = {
                    "language": result_data["language"],
                    "duration": result_data["duration"],
                    "total_segments": len(result_data["segments"]),
                    "preview": [seg.model_dump() for seg in segments_preview]
                }
        
        # Kalau failed, include error message
        if status == "failed":
            error = await redis.get(f"job:{job_id}:error")
            if error:
                response.error = error.decode()
        
        return response

    finally:
        await redis.close()

@router.get("/transcribe/result/{job_id}", response_model=TranscriptionResult)
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
                detail=f"Job is not completed yet. Current status: {status}"
            )

        result = await redis.get(f"job:{job_id}:result")

        if not result:
            raise HTTPException(status_code=404, detail="Result not found")
        
        result_data = json.loads(result.decode())

        # Convert all segments
        segments = [
            SegmentPreview(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"]
            )
            for seg in result_data["segments"]
        ]

        return TranscriptionResult(
            language=result_data["language"],
            duration=result_data["duration"],
            total_segments=len(segments),
            segments=segments
        )

    finally:
        await redis.close()