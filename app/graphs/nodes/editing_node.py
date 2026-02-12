"""
Video editing node for LangGraph video processing pipeline.

Implements Phase 3: FFmpeg video editing (50% → 80%)
Cuts video clips and burns word-level subtitles using WhisperX timestamps.
"""

import os
import json
import asyncio
from typing import Literal, Optional, List, Tuple
from langgraph.types import Command
from app.schemas.graph_schemas import VideoProcessingState
from app.schemas.transcription import TranscriptionResultDetailed
from app.utils.progress_tracker import create_progress_tracker
from app.utils.subtitle_generator import generate_ass_subtitle
from app.utils.video_formats import get_format, VideoFormat
from app.utils.logging import logger
from app.core.config import settings
from redis import asyncio as aioredis


async def _detect_video_resolution(video_path: str) -> Tuple[int, int]:
    """
    Detect video resolution using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (width, height)
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        video_path,
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            data = json.loads(stdout.decode())
            stream = data.get("streams", [{}])[0]
            width = stream.get("width", 1920)
            height = stream.get("height", 1080)
            return (width, height)
        else:
            logger.warning(f"ffprobe failed, using default 1920x1080")
            return (1920, 1080)
    except Exception as e:
        logger.warning(f"ffprobe error: {e}, using default 1920x1080")
        return (1920, 1080)


def _calculate_crop_filter(
    input_width: int,
    input_height: int,
    target_format: VideoFormat,
) -> Optional[str]:
    """
    Calculate FFmpeg crop+scale filter to convert video to target format.

    Args:
        input_width: Input video width
        input_height: Input video height
        target_format: Target video format

    Returns:
        FFmpeg filter string or None if no crop needed
    """
    target_w = target_format.width
    target_h = target_format.height
    target_ratio = target_w / target_h
    input_ratio = input_width / input_height

    # If already correct ratio, just scale
    if abs(input_ratio - target_ratio) < 0.01:
        if input_width != target_w or input_height != target_h:
            return f"scale={target_w}:{target_h}"
        return None

    # Need to crop
    if input_ratio > target_ratio:
        # Input is wider → crop width (landscape → portrait)
        crop_h = input_height
        crop_w = int(input_height * target_ratio)
        crop_x = (input_width - crop_w) // 2
        crop_y = 0
    else:
        # Input is taller → crop height (portrait → landscape)
        crop_w = input_width
        crop_h = int(input_width / target_ratio)
        crop_x = 0
        crop_y = (input_height - crop_h) // 2

    # Build filter: crop then scale
    filter_str = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={target_w}:{target_h}"
    logger.info(
        f"📐 Crop {input_width}x{input_height} → {target_w}x{target_h}: {filter_str}"
    )
    return filter_str


async def _cut_clip_ffmpeg(
    input_path: str,
    output_path: str,
    start: float,
    end: float,
    job_id: str,
    clip_index: int,
    subtitle_path: Optional[str] = None,
    crop_filter: Optional[str] = None,
) -> dict:
    """
    Cut a single clip using FFmpeg async subprocess, optionally burning subtitles.

    Args:
        input_path: Source video file path
        output_path: Destination clip file path
        start: Start timestamp in seconds
        end: End timestamp in seconds
        job_id: Job ID for logging
        clip_index: Clip number for logging
        subtitle_path: Optional path to .ass subtitle file to burn
        crop_filter: Optional FFmpeg crop/scale filter

    Returns:
        Dict with clip info or error details
    """
    duration = end - start

    # Use input seeking (faster, less accurate but subtitles actually render)
    cmd = [
        "ffmpeg",
        "-ss",
        str(start),
        "-i",
        input_path,
        "-t",
        str(duration),
    ]

    # Build video filter chain: crop + subtitle
    video_filters = []

    if crop_filter:
        video_filters.append(crop_filter)

    if subtitle_path and os.path.exists(subtitle_path):
        # Escape backslashes and colons for FFmpeg filter on Windows
        escaped_path = subtitle_path.replace("\\", "/").replace(":", "\\:")
        video_filters.append(f"ass='{escaped_path}'")
        logger.info(f"🔤 [Job {job_id}] Burning subtitles for clip {clip_index}")

    if video_filters:
        filter_chain = ",".join(video_filters)
        cmd.extend(["-vf", filter_chain])

    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            "-y",
            output_path,
        ]
    )

    logger.info(
        f"✂️ [Job {job_id}] Cutting clip {clip_index}: "
        f"{start:.1f}s → {end:.1f}s ({duration:.1f}s)"
    )

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode()[-500:]
            logger.error(
                f"❌ [Job {job_id}] FFmpeg failed for clip {clip_index}: {error_msg}"
            )
            return {"success": False, "error": error_msg}

        file_size = os.path.getsize(output_path)
        logger.info(
            f"✅ [Job {job_id}] Clip {clip_index} saved: "
            f"{output_path} ({file_size / 1024 / 1024:.1f} MB)"
        )

        return {
            "success": True,
            "file_path": output_path,
            "file_size": file_size,
            "duration": duration,
        }

    except FileNotFoundError:
        logger.error(f"❌ [Job {job_id}] FFmpeg not found! Is it installed?")
        return {"success": False, "error": "FFmpeg not found"}
    except Exception as e:
        logger.error(f"❌ [Job {job_id}] FFmpeg error: {e}")
        return {"success": False, "error": str(e)}


async def editing_node(
    state: VideoProcessingState, redis: aioredis.Redis
) -> Command[Literal["finalization"]]:
    """
    Phase 3: Video editing using FFmpeg with subtitle burning.

    Pipeline steps (50% → 80%):
    1. Create output directory for job clips
    2. Generate ASS subtitles per clip (word-level from WhisperX)
    3. Cut each clip + burn subtitles using FFmpeg
    4. Collect metadata and route to finalization

    Args:
        state: Current LangGraph state
        redis: Async Redis connection for progress tracking

    Returns:
        Command with generated clips routing to finalization
    """
    job_id = state["job_id"]
    video_path = state.get("video_path", "")
    clip_candidates = state.get("clip_candidates", [])
    transcription: Optional[TranscriptionResultDetailed] = state.get(
        "transcription_result"
    )

    # Initialize progress tracker
    tracker = create_progress_tracker(redis, job_id)

    try:
        logger.info(f"🎬 [Job {job_id}] Starting Phase 3: Video Editing + Subtitles")
        await tracker.update_progress(50, "editing", "Phase 3: Video Editing")

        # Validate input
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if not clip_candidates:
            logger.warning(f"⚠️ [Job {job_id}] No clip candidates to edit")
            return Command(
                update={
                    "clips": [],
                    "progress": 80,
                    "status": "finalizing",
                    "current_phase": "Phase 4: Finalization",
                },
                goto="finalization",
            )

        # Create output directory: clips/{job_id}/
        job_clips_dir = os.path.join(settings.CLIPS_DIR, job_id)
        os.makedirs(job_clips_dir, exist_ok=True)

        # Get transcription segments for subtitle generation
        segments = transcription.segments if transcription else []

        # Detect video resolution and get target format
        logger.info(f"🔍 [Job {job_id}] Detecting video resolution...")
        input_width, input_height = await _detect_video_resolution(video_path)
        logger.info(f"📐 [Job {job_id}] Input: {input_width}x{input_height}")

        # Get target output format from config
        try:
            target_format = get_format(settings.OUTPUT_FORMAT)
            logger.info(
                f"🎯 [Job {job_id}] Target format: {target_format.description} "
                f"({target_format.width}x{target_format.height})"
            )
        except ValueError:
            # Fallback to TikTok if invalid format
            logger.warning(
                f"⚠️ [Job {job_id}] Invalid OUTPUT_FORMAT '{settings.OUTPUT_FORMAT}', "
                "using TikTok (1080x1920)"
            )
            target_format = get_format("tiktok")

        # Calculate crop filter if needed
        crop_filter = _calculate_crop_filter(input_width, input_height, target_format)

        # Cut each clip
        generated_clips = []
        total_clips = len(clip_candidates)
        errors = []

        for i, candidate in enumerate(clip_candidates):
            clip_num = i + 1
            clip_start = candidate["start"]
            clip_end = candidate["end"]

            # Progress: distribute 50→78% across clips
            clip_progress = 50 + int((clip_num / total_clips) * 28)
            await tracker.update_progress(
                clip_progress,
                phase=f"Cutting clip {clip_num}/{total_clips}: {candidate.get('title', '')}",
            )

            # Step A: Generate ASS subtitle for this clip
            subtitle_path = None
            if segments:
                ass_filename = f"clip_{clip_num}.ass"
                ass_path = os.path.join(job_clips_dir, ass_filename)

                subtitle_path = generate_ass_subtitle(
                    segments=segments,
                    clip_start=clip_start,
                    clip_end=clip_end,
                    output_path=ass_path,
                    job_id=job_id,
                    video_width=target_format.width,
                    video_height=target_format.height,
                    font_size=target_format.subtitle_size,
                    margin_bottom=target_format.subtitle_margin_bottom,
                )

            # Step B: Cut clip + burn subtitles with FFmpeg
            clip_filename = f"clip_{clip_num}.mp4"
            output_path = os.path.join(job_clips_dir, clip_filename)

            result = await _cut_clip_ffmpeg(
                input_path=video_path,
                output_path=output_path,
                start=clip_start,
                end=clip_end,
                job_id=job_id,
                clip_index=clip_num,
                subtitle_path=subtitle_path,
                crop_filter=crop_filter,
            )

            if result["success"]:
                clip_metadata = {
                    "clip_id": f"{job_id}_clip_{clip_num}",
                    "clip_number": clip_num,
                    "start": clip_start,
                    "end": clip_end,
                    "duration": result["duration"],
                    "title": candidate.get("title", f"Clip {clip_num}"),
                    "reasoning": candidate.get("reasoning", ""),
                    "viral_score": candidate.get("viral_score", 0),
                    "suggested_caption": candidate.get("suggested_caption", ""),
                    "file_path": output_path,
                    "file_size": result["file_size"],
                    "has_subtitles": subtitle_path is not None,
                    "subtitle_path": subtitle_path,
                    "status": "ready",
                }
                generated_clips.append(clip_metadata)
            else:
                error_msg = f"Clip {clip_num} failed: {result['error']}"
                errors.append(error_msg)
                logger.error(f"❌ [Job {job_id}] {error_msg}")

        await tracker.update_progress(80, "finalizing", "Phase 3 complete")

        logger.info(
            f"✅ [Job {job_id}] Phase 3 COMPLETE! "
            f"Cut {len(generated_clips)}/{total_clips} clips with subtitles"
        )

        # Build state update
        update = {
            "clips": generated_clips,
            "progress": 80,
            "status": "finalizing",
            "current_phase": "Phase 4: Finalization",
        }

        if errors:
            update["errors"] = errors

        return Command(update=update, goto="finalization")

    except Exception as e:
        error_msg = f"Video editing failed: {str(e)}"
        logger.error(f"❌ [Job {job_id}] {error_msg}", exc_info=True)
        await tracker.set_error(error_msg)

        return Command(
            update={
                "progress": 50,
                "status": "failed",
                "current_phase": "Failed",
                "errors": [error_msg],
            },
            goto="finalization",
        )
