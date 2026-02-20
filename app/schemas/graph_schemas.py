"""
LangGraph State Schemas for Video Processing Pipeline.

Uses 2026 best practices:
- Annotated reducers for safe state merging
- TypedDict for clear type hints
- Operator-based reducers for list accumulation
"""

from typing import TypedDict, Optional, Annotated, List
from typing_extensions import NotRequired
import operator
from app.schemas.transcription import TranscriptionResultDetailed


class VideoProcessingState(TypedDict):
    """
    LangGraph state for video processing pipeline.

    Uses Annotated reducers to prevent state overwrites in parallel execution.

    State Flow:
    1. Input: job_id, video_path
    2. Phase 1: audio_data, transcription_result
    3. Phase 2: analysis_result, clip_candidates
    4. Phase 3: clips (accumulated)
    5. Phase 4: final_result
    6. Tracking: progress, status, current_phase, errors
    """

    # ===== Input Fields =====
    job_id: str  # Unique job identifier
    user_id: NotRequired[Optional[str]]  # User ID dari auth-service
    video_path: str  # Path to video file
    source: str  # "upload" or "youtube"
    source_url: NotRequired[Optional[str]]  # YouTube URL (if source="youtube")

    # ===== Processing Data (Optional) =====
    audio_data: NotRequired[
        Optional[bytes]
    ]  # Audio extracted from video (memory intensive, optional)

    # ===== Phase Results =====
    # Phase 1: Transcription (WhisperX)
    transcription_result: NotRequired[Optional[TranscriptionResultDetailed]]

    # Phase 2: Content Analysis (Gemini) - TODO
    analysis_result: NotRequired[Optional[dict]]  # Gemini analysis output
    clip_candidates: NotRequired[List[dict]]  # Suggested clips with timestamps

    # Phase 3A: Hook Generation (Gemini + TTS)
    # Each hook: {"clip_index": 0, "hook_text": "...", "caption": "...", "audio_path": "...", "language": "id"}
    hooks: NotRequired[Optional[List[dict]]]

    # Phase 3B: Video Editing (FFmpeg)
    # Use Annotated reducer to accumulate clips from parallel editing
    clips: Annotated[List[dict], operator.add]  # Generated clip files

    # Phase 4: Finalization
    final_result: NotRequired[Optional[dict]]  # Final metadata and thumbnails

    # ===== Progress Tracking =====
    progress: int  # 0-100 percentage
    status: str  # queued/processing/transcribing/analyzing/editing/finalizing/completed/failed
    current_phase: str  # Human-readable current phase name

    # ===== Error Handling =====
    # Use Annotated reducer to collect errors from all phases
    errors: Annotated[List[str], operator.add]  # Accumulated error messages


def create_initial_state(job_id: str, video_path: str) -> VideoProcessingState:
    """
    Create initial state for video processing job.

    Args:
        job_id: Unique job identifier
        video_path: Path to video file

    Returns:
        VideoProcessingState with initial values
    """
    return VideoProcessingState(
        job_id=job_id,
        video_path=video_path,
        source="upload",
        clips=[],  # Initialize empty list for Annotated[List, operator.add]
        errors=[],  # Initialize empty list for error accumulation
        progress=0,
        status="queued",
        current_phase="initialization",
    )
