"""
Video format definitions for different platforms.

Supports TikTok/Reels (9:16), Instagram Square (1:1), YouTube Shorts/Landscape, and custom formats.
Each format includes resolution, aspect ratio, and optimized subtitle styling.
"""

from typing import Dict, Any


class VideoFormat:
    """Video format configuration."""

    def __init__(
        self,
        width: int,
        height: int,
        ratio: str,
        subtitle_size: int,
        subtitle_margin_bottom: int,
        description: str = "",
    ):
        self.width = width
        self.height = height
        self.ratio = ratio
        self.subtitle_size = subtitle_size
        self.subtitle_margin_bottom = subtitle_margin_bottom
        self.description = description

    @property
    def aspect_ratio(self) -> float:
        """Calculate numeric aspect ratio (width/height)."""
        return self.width / self.height

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "ratio": self.ratio,
            "subtitle_size": self.subtitle_size,
            "subtitle_margin_bottom": self.subtitle_margin_bottom,
            "description": self.description,
        }


# Predefined platform formats
FORMATS: Dict[str, VideoFormat] = {
    "tiktok": VideoFormat(
        width=1080,
        height=1920,
        ratio="9:16",
        subtitle_size=85,
        subtitle_margin_bottom=300,
        description="TikTok vertical (1080x1920)",
    ),
    "reels": VideoFormat(
        width=1080,
        height=1920,
        ratio="9:16",
        subtitle_size=85,
        subtitle_margin_bottom=280,
        description="Instagram Reels vertical (1080x1920)",
    ),
    "shorts": VideoFormat(
        width=1080,
        height=1920,
        ratio="9:16",
        subtitle_size=85,
        subtitle_margin_bottom=280,
        description="YouTube Shorts vertical (1080x1920)",
    ),
    "square": VideoFormat(
        width=1080,
        height=1080,
        ratio="1:1",
        subtitle_size=75,
        subtitle_margin_bottom=200,
        description="Instagram Square (1080x1080)",
    ),
    "landscape": VideoFormat(
        width=1920,
        height=1080,
        ratio="16:9",
        subtitle_size=65,
        subtitle_margin_bottom=150,
        description="YouTube Landscape (1920x1080)",
    ),
}


def get_format(format_name: str) -> VideoFormat:
    """
    Get video format by name.

    Args:
        format_name: Format identifier (tiktok, reels, shorts, square, landscape)

    Returns:
        VideoFormat instance

    Raises:
        ValueError: If format not found
    """
    if format_name not in FORMATS:
        raise ValueError(
            f"Unknown format '{format_name}'. "
            f"Available: {', '.join(FORMATS.keys())}"
        )
    return FORMATS[format_name]


def create_custom_format(
    width: int,
    height: int,
    subtitle_size: int = 24,
    subtitle_margin_bottom: int = 100,
) -> VideoFormat:
    """
    Create a custom video format.

    Args:
        width: Video width in pixels
        height: Video height in pixels
        subtitle_size: Font size for subtitles
        subtitle_margin_bottom: Bottom margin for subtitles in pixels

    Returns:
        VideoFormat instance
    """
    ratio = f"{width}:{height}"
    return VideoFormat(
        width=width,
        height=height,
        ratio=ratio,
        subtitle_size=subtitle_size,
        subtitle_margin_bottom=subtitle_margin_bottom,
        description=f"Custom {width}x{height}",
    )
