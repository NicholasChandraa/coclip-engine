from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, desc
from app.middleware.auth import CurrentUser, get_current_user
from app.core.database import async_session
from app.models import Job, Clip
from pydantic import BaseModel
from app.utils.logging import logger

router = APIRouter()

class ClipResponse(BaseModel):
    clip_id: str | int
    clip_number: int
    job_id: str
    title: str
    start_time: float
    end_time: float
    viral_score: float | None
    suggested_caption: str | None
    hook_text: str | None
    transcript_text: str | None
    reasoning: str | None
    tags: List[str] | None
    status: str
    file_path: str | None
    file_size: int | None
    created_at: str

    class Config:
        from_attributes = True

@router.get("/jobs/clips/top", response_model=List[ClipResponse])
async def get_top_clips(
    limit: int = Query(6, ge=1, le=20),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Get the top rated clips across all jobs for the current user.
    Ordered by viral_score descending.
    """
    try:
        async with async_session() as session:
            # Join Job and Clip to filter by user_id and only get completed clips
            stmt = (
                select(Clip)
                .join(Job, Clip.job_id == Job.id)
                .where(
                    Job.user_id == current_user.user_id,
                    Clip.status.in_(["ready", "completed"]),
                    Clip.viral_score.isnot(None)
                )
                .order_by(desc(Clip.viral_score))
                .limit(limit)
            )

            result = await session.execute(stmt)
            clips = result.scalars().all()

            response_clips = []
            for clip in clips:
                response_clips.append(
                    ClipResponse(
                        clip_id=clip.id,  # Note: clip.id is an integer, clip.clip_id is string
                        clip_number=clip.clip_number,
                        job_id=clip.job_id,
                        title=clip.title,
                        start_time=clip.start,
                        end_time=clip.end,
                        viral_score=clip.viral_score,
                        suggested_caption=clip.suggested_caption,
                        hook_text=clip.hook_text,
                        transcript_text=clip.transcript_text,
                        reasoning=clip.reasoning,
                        tags=clip.tags or [],
                        status=clip.status,
                        file_path=clip.file_path,
                        file_size=clip.file_size,
                        created_at=str(clip.created_at)
                    )
                )

            return response_clips

    except Exception as e:
        logger.error(f"❌ Failed to fetch top clips: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch top clips: {str(e)}"
        )
