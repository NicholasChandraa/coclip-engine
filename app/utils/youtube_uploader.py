# engine/app/utils/youtube_uploader.py
"""YouTube Data API v3 — resumable upload utility."""

import asyncio
from pathlib import Path

import httpx

from datetime import datetime, timezone

YOUTUBE_UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
CHUNK_SIZE = 5 * 1024 * 1024  # 5 MB chunks


async def upload_to_youtube(
    access_token: str,
    clip_path: str,
    title: str,
    description: str,
    tags: list[str],
    privacy: str,  # "public" | "unlisted" | "private"
    scheduled_time: datetime | None = None,
) -> dict:
    """Upload a video file to YouTube via resumable upload.

    Returns {"video_id": str, "url": str}.
    Raises httpx.HTTPStatusError or Exception on failure.
    """
    file_path = Path(clip_path)
    file_size = file_path.stat().st_size

    status_payload = {
        "privacyStatus": privacy,
        "selfDeclaredMadeForKids": False,
    }

    if scheduled_time:
        # YouTube requires scheduled videos to have privacyStatus declared as private.
        status_payload["privacyStatus"] = "private"
        # Ensure the datetime is UTC and format it exactly how YouTube expects: YYYY-MM-DDThh:mm:ss.000Z
        utc_time = scheduled_time.astimezone(timezone.utc)
        status_payload["publishAt"] = utc_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    metadata = {
        "snippet": {
            "title": title,
            "description": description or "",
            "tags": [t.lstrip("#") for t in (tags or [])],
            "categoryId": "22",  # People & Blogs
        },
        "status": status_payload,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Initiate resumable upload session
        init_resp = await client.post(
            f"{YOUTUBE_UPLOAD_URL}?uploadType=resumable&part=snippet,status",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "X-Upload-Content-Type": "video/mp4",
                "X-Upload-Content-Length": str(file_size),
            },
            json=metadata,
        )
        if not init_resp.is_success:
            print(f"YouTube API Error Response: {init_resp.text}")
        init_resp.raise_for_status()
        upload_url = init_resp.headers["Location"]

    # Step 2: Upload file in chunks (new client for long-running upload)
    video_id = await _upload_chunks(upload_url, file_path, file_size)
    return {
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
    }


async def _upload_chunks(upload_url: str, file_path: Path, file_size: int) -> str:
    """Upload file in 5MB chunks using YouTube resumable upload protocol."""
    offset = 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        with open(file_path, "rb") as f:
            while offset < file_size:
                chunk = f.read(CHUNK_SIZE)
                chunk_len = len(chunk)
                end = offset + chunk_len - 1

                resp = await client.put(
                    upload_url,
                    content=chunk,
                    headers={
                        "Content-Range": f"bytes {offset}-{end}/{file_size}",
                        "Content-Type": "video/mp4",
                    },
                )

                if resp.status_code in (200, 201):
                    data = resp.json()
                    return data["id"]
                elif resp.status_code == 308:
                    # Resume Incomplete — continue with next chunk
                    range_header = resp.headers.get("Range", "")
                    if range_header:
                        offset = int(range_header.split("-")[1]) + 1
                    else:
                        offset += chunk_len
                else:
                    raise Exception(
                        f"YouTube upload failed: {resp.status_code} {resp.text}"
                    )

    raise Exception("Upload ended without receiving video ID")
