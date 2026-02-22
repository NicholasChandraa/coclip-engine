"""Social media upload endpoints."""

import asyncio
import uuid
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from app.core.config import settings
from app.core.database import async_session
from app.middleware.auth import CurrentUser, get_current_user
from app.models import Clip, ClipUpload
from app.utils.youtube_uploader import upload_to_youtube
from app.utils.logging import logger

router = APIRouter()


class UploadRequest(BaseModel):
    clip_id: str
    platform: str  # "youtube"
    title: str
    description: str | None = None
    tags: list[str] | None = None
    privacy: str = "private"  # public | unlisted | private


@router.post("/social/upload")
async def start_upload(
    req: UploadRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Trigger a clip upload to a social platform. Returns upload_id for polling."""
    async with async_session() as session:
        # Verify clip exists
        result = await session.execute(
            select(Clip).where(Clip.clip_id == req.clip_id)
        )
        clip = result.scalar_one_or_none()
        if not clip:
            raise HTTPException(status_code=404, detail="Clip not found")

        # Get valid access token from auth-service
        access_token = await _get_token(current_user.user_id, req.platform)

        # Create upload record
        upload_id = str(uuid.uuid4())
        upload = ClipUpload(
            id=upload_id,
            clip_id=req.clip_id,
            user_id=current_user.user_id,
            platform=req.platform,
            status="uploading",
            title=req.title,
            description=req.description,
            tags=req.tags,
            privacy=req.privacy,
        )
        session.add(upload)
        await session.commit()
        clip_path = clip.file_path

    # Fire background task (don't await — returns immediately)
    asyncio.create_task(
        _run_upload(upload_id, access_token, clip_path, req)
    )

    return {"upload_id": upload_id, "status": "uploading"}


@router.get("/social/upload/{upload_id}")
async def get_upload_status(
    upload_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Poll upload status by upload_id."""
    async with async_session() as session:
        upload = await session.get(ClipUpload, upload_id)
        if not upload or upload.user_id != current_user.user_id:
            raise HTTPException(status_code=404, detail="Upload not found")
        return {
            "upload_id": upload.id,
            "status": upload.status,
            "platform_video_id": upload.platform_video_id,
            "platform_url": upload.platform_url,
            "error": upload.error,
        }


@router.get("/social/uploads")
async def list_uploads(
    current_user: CurrentUser = Depends(get_current_user),
):
    """List all upload history for the current user."""
    async with async_session() as session:
        result = await session.execute(
            select(ClipUpload)
            .where(ClipUpload.user_id == current_user.user_id)
            .order_by(ClipUpload.created_at.desc())
            .limit(50)
        )
        uploads = result.scalars().all()
        return [
            {
                "upload_id": u.id,
                "clip_id": u.clip_id,
                "platform": u.platform,
                "status": u.status,
                "platform_url": u.platform_url,
                "title": u.title,
                "created_at": u.created_at.isoformat() if u.created_at else None,
            }
            for u in uploads
        ]


async def _get_token(user_id: str, platform: str) -> str:
    """Fetch a valid access token from auth-service internal endpoint."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{settings.AUTH_SERVICE_URL}/api/v1/internal/social/token/{user_id}/{platform}",
            headers={"X-Service-Token": settings.AUTH_SERVICE_TOKEN},
        )
        if resp.status_code == 404:
            raise HTTPException(
                status_code=400,
                detail=f"{platform} account not connected. Go to Settings to connect.",
            )
        resp.raise_for_status()
        return resp.json()["access_token"]


async def _run_upload(
    upload_id: str,
    access_token: str,
    clip_path: str,
    req: UploadRequest,
) -> None:
    """Background task: upload video and update ClipUpload status."""
    async with async_session() as session:
        try:
            logger.info(f"Starting YouTube upload for upload_id={upload_id}")
            result = await upload_to_youtube(
                access_token=access_token,
                clip_path=clip_path,
                title=req.title,
                description=req.description or "",
                tags=req.tags or [],
                privacy=req.privacy,
            )
            upload = await session.get(ClipUpload, upload_id)
            if upload:
                upload.status = "completed"
                upload.platform_video_id = result["video_id"]
                upload.platform_url = result["url"]
                upload.completed_at = datetime.now(timezone.utc)
                await session.commit()
            logger.info(f"Upload completed: {result['url']}")
        except Exception as e:
            logger.error(f"Upload failed for {upload_id}: {e}")
            async with async_session() as err_session:
                upload = await err_session.get(ClipUpload, upload_id)
                if upload:
                    upload.status = "failed"
                    upload.error = str(e)
                    await err_session.commit()
