"""
SQLAlchemy ORM models for CoClip engine.

Tables:
- jobs: Video processing job records
- clips: Generated clip metadata (1:N with jobs)
"""

from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import String, Float, Integer, Boolean, Text, DateTime, ForeignKey, Index, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Optional, List
from app.core.database import Base


class Job(Base):
    """Video processing job record."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID job_id
    user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)  # UUID dari auth-service
    video_name: Mapped[str] = mapped_column(String(255))
    language: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_segments: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String(20), default="upload")  # "upload" or "youtube"
    source_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # YouTube URL
    status: Mapped[str] = mapped_column(String(50), default="queued")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationship
    clips: Mapped[List["Clip"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "video_name": self.video_name,
            "source": self.source,
            "source_url": self.source_url,
            "language": self.language,
            "duration": self.duration,
            "total_segments": self.total_segments,
            "status": self.status,
            "error": self.error,
            "clips_count": len(self.clips) if self.clips else 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
        }


class Clip(Base):
    """Generated video clip metadata."""

    __tablename__ = "clips"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    clip_id: Mapped[str] = mapped_column(String(100), unique=True)  # job_id_clip_N
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    clip_number: Mapped[int] = mapped_column(Integer)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    duration: Mapped[float] = mapped_column(Float)
    title: Mapped[str] = mapped_column(String(255))
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    viral_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    suggested_caption: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    hook_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    transcript_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    file_path: Mapped[str] = mapped_column(String(500))
    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    has_subtitles: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(50), default="ready")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationship
    job: Mapped["Job"] = relationship(back_populates="clips")
    uploads: Mapped[List["ClipUpload"]] = relationship(
        primaryjoin="Clip.clip_id == ClipUpload.clip_id",
        foreign_keys="[ClipUpload.clip_id]",
        viewonly=True
    )

    def to_dict(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "job_id": self.job_id,
            "clip_number": self.clip_number,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "title": self.title,
            "reasoning": self.reasoning,
            "viral_score": self.viral_score,
            "suggested_caption": self.suggested_caption,
            "hook_text": self.hook_text,
            "transcript_text": self.transcript_text,
            "tags": self.tags,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "has_subtitles": self.has_subtitles,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "uploads": [{"platform": u.platform, "status": u.status, "url": u.platform_url} for u in self.uploads] if getattr(self, "uploads", None) else [],
        }


class ClipUpload(Base):
    """Upload record for a clip to a social media platform."""

    __tablename__ = "clip_uploads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    clip_id: Mapped[str] = mapped_column(String(255), ForeignKey("clips.clip_id", ondelete="CASCADE"))
    user_id: Mapped[str] = mapped_column(String(36), nullable=False)
    platform: Mapped[str] = mapped_column(String(50), nullable=False)  # "youtube"
    status: Mapped[str] = mapped_column(String(50), default="uploading")  # uploading|completed|failed
    platform_video_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    platform_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    privacy: Mapped[str] = mapped_column(String(20), default="private")  # public|unlisted|private
    scheduled_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
