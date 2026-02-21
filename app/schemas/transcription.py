from pydantic import BaseModel
from typing import List, Optional


class SegmentPreview(BaseModel):
    """
    Model untuk preview segment hasil transcription.
    Berisi timing (start/end) dan text dari satu segment audio.
    """
    start: float  # Waktu mulai segment dalam detik
    end: float    # Waktu akhir segment dalam detik
    text: str     # Text hasil transcribe untuk segment ini


class WordTimestamp(BaseModel):
    """
    Word-level timestamp dari WhisperX alignment.
    Setiap word punya start/end time dan confidence score.
    """
    word: str
    start: float
    end: float
    score: float


class TranscriptionSegment(BaseModel):
    """
    Segment dengan WhisperX features (word timestamps + speaker).
    """
    start: float
    end: float
    text: str
    words: Optional[List[WordTimestamp]] = None  # Word-level timestamps
    speaker: Optional[str] = None  # Speaker label dari diarization


class YouTubeRequest(BaseModel):
    """Request body untuk YouTube URL transcription."""
    url: str  # YouTube video URL
    job_name: Optional[str] = None  # Optional custom name (fallback: YouTube title)


class TranscribeAsyncResponse(BaseModel):
    """Response untuk async transcription endpoint."""
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Response untuk job status check endpoint."""
    job_id: str
    status: str  # queued/processing/completed/failed
    progress: int  # 0-100
    result: Optional[dict] = None
    error: Optional[str] = None


class TranscriptionResult(BaseModel):
    """
    Full transcription result dengan semua segments.
    Include language detection, duration, dan list of segments.
    """
    language: str
    duration: float
    total_segments: int
    segments: List[SegmentPreview]


class TranscriptionResultDetailed(BaseModel):
    """
    Detailed transcription result dengan word timestamps dan speaker info.
    Untuk advanced use cases yang butuh word-level precision.
    """
    language: str
    duration: float
    total_segments: int
    segments: List[TranscriptionSegment]
