"""
YouTube Video Downloader - yt-dlp integration for Coclip.

Downloads YouTube videos to local storage for pipeline processing.
"""

import os
import re
import asyncio
import signal
from typing import Optional, Callable
from dataclasses import dataclass

from app.utils.logging import logger

# Global reference to active download process for cancellation
_active_downloads: dict = {}  # job_id -> yt_dlp.YoutubeDL instance


@dataclass
class DownloadResult:
    """Result of a YouTube video download."""

    file_path: str
    title: str
    duration: float  # seconds
    uploader: str
    file_size: int  # bytes


# Regex patterns for YouTube URL validation
YOUTUBE_PATTERNS = [
    r"^https?://(www\.)?youtube\.com/watch\?v=[\w-]+",
    r"^https?://youtu\.be/[\w-]+",
    r"^https?://(www\.)?youtube\.com/shorts/[\w-]+",
]


def validate_youtube_url(url: str) -> bool:
    """Check if URL is a valid YouTube URL."""
    return any(re.match(pattern, url) for pattern in YOUTUBE_PATTERNS)


async def get_video_info(url: str) -> Optional[dict]:
    """
    Fetch video metadata without downloading.

    Returns:
        dict with title, duration, uploader, resolution, or None on error
    """
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    def _extract():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=False)

    try:
        info = await asyncio.to_thread(_extract)
        result = {
            "title": info.get("title", "Unknown"),
            "duration": info.get("duration", 0),
            "uploader": info.get("uploader", "Unknown"),
            "width": info.get("width"),
            "height": info.get("height"),
        }

        logger.info(f"[VIDEO YOUTUBE INFO]: {result}")

        return result
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return None


async def download_video(
    url: str,
    output_dir: str,
    job_id: str,
    progress_callback: Optional[Callable] = None,
) -> DownloadResult:
    """
    Download YouTube video using yt-dlp.

    Args:
        url: YouTube video URL
        output_dir: Directory to save the video
        job_id: Job ID used as filename
        progress_callback: Optional async callback(percent: float) for progress updates

    Returns:
        DownloadResult with file path and metadata

    Raises:
        Exception if download fails
    """
    import yt_dlp

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{job_id}.mp4")
    output_template = os.path.join(output_dir, f"{job_id}.%(ext)s")

    last_percent = [0.0]

    def _progress_hook(d):
        if d["status"] == "downloading":
            pct_str = d.get("_percent_str", "0%").strip().replace("%", "")
            try:
                pct = float(pct_str)
                if pct - last_percent[0] >= 5:  # update every 5%
                    last_percent[0] = pct
                    logger.info(
                        f"  [Job {job_id}] Downloading: {pct:.0f}% "
                        f"| Speed: {d.get('_speed_str', '?')} "
                        f"| ETA: {d.get('_eta_str', '?')}"
                    )
            except ValueError:
                pass
        elif d["status"] == "finished":
            logger.info(f"  [Job {job_id}] Download finished, merging...")

    ydl_opts = {
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_progress_hook],
    }

    def _download():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            _active_downloads[job_id] = ydl
            try:
                info = ydl.extract_info(url, download=True)
                return info
            finally:
                _active_downloads.pop(job_id, None)

    logger.info(f"[Job {job_id}] Starting YouTube download: {url}")

    info = await asyncio.to_thread(_download)

    # yt-dlp may save with different extension before merging
    if not os.path.exists(output_path):
        # Check for webm or other format
        for ext in ["mp4", "mkv", "webm"]:
            candidate = os.path.join(output_dir, f"{job_id}.{ext}")
            if os.path.exists(candidate):
                output_path = candidate
                break

    if not os.path.exists(output_path):
        raise FileNotFoundError(
            f"Downloaded file not found at {output_path}"
        )

    file_size = os.path.getsize(output_path)
    title = info.get("title", "Unknown")
    duration = info.get("duration", 0)
    uploader = info.get("uploader", "Unknown")

    logger.info(
        f"[Job {job_id}] Download complete: "
        f"'{title}' ({duration:.0f}s, {file_size / 1024 / 1024:.1f} MB)"
    )

    return DownloadResult(
        file_path=output_path,
        title=title,
        duration=duration,
        uploader=uploader,
        file_size=file_size,
    )


def cancel_download(job_id: str) -> bool:
    """Cancel an active download for a job.

    Returns True if a download was found and cancelled.
    """
    ydl = _active_downloads.pop(job_id, None)
    if ydl is not None:
        try:
            # yt-dlp checks _download_retcode to abort
            ydl._download_retcode = 1
            logger.info(f"[Job {job_id}] Download cancelled")
            return True
        except Exception as e:
            logger.warning(f"[Job {job_id}] Failed to cancel download: {e}")
    return False
