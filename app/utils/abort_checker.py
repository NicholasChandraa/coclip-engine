"""
Utility untuk cek apakah job telah di-abort.

Dipanggil di awal setiap node DAN di checkpoint-checkpoint penting
di dalam node yang long-running (WhisperX, Gemini, FFmpeg).

Usage:
    from app.utils.abort_checker import AbortError, check_aborted, raise_if_aborted

    # Pattern 1: return Command langsung (di awal node)
    if await check_aborted(redis, job_id):
        return Command(update={"status": "aborted"}, goto="finalization")

    # Pattern 2: raise exception (di tengah node, dalam try/except)
    await raise_if_aborted(redis, job_id)
"""

from redis import asyncio as aioredis
from app.utils.logging import logger


class AbortError(Exception):
    """Raised ketika job di-abort di tengah processing."""
    pass


async def check_aborted(redis: aioredis.Redis, job_id: str) -> bool:
    """Returns True jika job sudah di-abort."""
    try:
        status = await redis.get(f"job:{job_id}:status")
        return status is not None and status.decode() == "aborted"
    except Exception:
        return False


async def raise_if_aborted(redis: aioredis.Redis, job_id: str, checkpoint: str = "") -> None:
    """
    Raise AbortError jika job sudah di-abort.
    Dipanggil di checkpoint-checkpoint di dalam node.

    Args:
        redis: Async Redis connection
        job_id: Job ID
        checkpoint: Nama checkpoint untuk logging (opsional)
    """
    if await check_aborted(redis, job_id):
        msg = f"[Job {job_id}] Aborted"
        if checkpoint:
            msg += f" at checkpoint: {checkpoint}"
        logger.info(f"⏭️ {msg}")
        raise AbortError(msg)
