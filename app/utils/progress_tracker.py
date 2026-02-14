"""
Progress Tracker utility for LangGraph nodes.

Provides async Redis integration for real-time progress updates
that can be consumed by frontend via polling.
"""

from redis import asyncio as aioredis
from app.utils.logging import logger
from typing import Optional


class ProgressTracker:
    """
    Async progress tracker for LangGraph video processing jobs.

    Integrates with Redis to provide real-time progress updates
    that frontend can poll via /job/{job_id}/status endpoint.
    """

    def __init__(self, redis: aioredis.Redis, job_id: str):
        """
        Initialize progress tracker.

        Args:
            redis: Async Redis connection
            job_id: Unique job identifier
        """
        self.redis = redis
        self.job_id = job_id
        self._progress_key = f"job:{job_id}:progress"
        self._status_key = f"job:{job_id}:status"
        self._error_key = f"job:{job_id}:error"

    async def update_progress(
        self, progress: int, status: Optional[str] = None, phase: Optional[str] = None
    ) -> None:
        """
        Update job progress in Redis.

        Args:
            progress: Progress percentage (0-100)
            status: Optional status update (e.g., "transcribing", "analyzing")
            phase: Optional human-readable phase description
        """
        # Update progress
        await self.redis.set(self._progress_key, str(progress))

        # Update status if provided
        if status:
            await self.redis.set(self._status_key, status)

        # Log the update
        log_msg = f"[Job {self.job_id}] Progress: {progress}%"
        if status:
            log_msg += f" | Status: {status}"
        if phase:
            log_msg += f" | Phase: {phase}"
        logger.info(log_msg)

    async def get_progress(self) -> int:
        """
        Get current progress percentage.

        Returns:
            Current progress (0-100)
        """
        try:
            progress = await self.redis.get(self._progress_key)
            return int(progress) if progress else 0
        except Exception as e:
            logger.warning(f"[Job {self.job_id}] Redis get_progress failed: {e}")
            return 0

    async def get_status(self) -> str:
        """
        Get current status.

        Returns:
            Current status string
        """
        try:
            status = await self.redis.get(self._status_key)
            return status.decode() if status else "unknown"
        except Exception as e:
            logger.warning(f"[Job {self.job_id}] Redis get_status failed: {e}")
            return "unknown"

    async def set_error(self, error: str) -> None:
        """
        Mark job as failed with error message.

        Args:
            error: Error message
        """
        await self.redis.set(self._status_key, "failed")
        await self.redis.set(self._error_key, error)
        logger.error(f"[Job {self.job_id}] Failed: {error}")

    async def set_completed(self) -> None:
        """Mark job as completed."""
        await self.redis.set(self._status_key, "completed")
        await self.redis.set(self._progress_key, "100")
        logger.info(f"✅ [Job {self.job_id}] Completed successfully")


def create_progress_tracker(redis: aioredis.Redis, job_id: str) -> ProgressTracker:
    """
    Factory function to create ProgressTracker instance.

    Args:
        redis: Async Redis connection
        job_id: Unique job identifier

    Returns:
        ProgressTracker instance
    """
    return ProgressTracker(redis, job_id)
