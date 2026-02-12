"""
Subtitle generator for video clips.

Generates ASS (Advanced SubStation Alpha) subtitle files
with TikTok/Reels style word-by-word highlighting using
WhisperX word-level timestamps.
"""

import os
from typing import List, Optional
from app.schemas.transcription import TranscriptionSegment, WordTimestamp
from app.utils.logging import logger
from app.core.config import settings


def _seconds_to_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp format (H:MM:SS.CC)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    """Escape special characters for ASS format."""
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


# ===== ASS Style Presets =====


def _generate_ass_header(
    width: int = 1080,
    height: int = 1920,
    font_size: int = 28,
    margin_bottom: int = 180,
) -> str:
    """
    Generate ASS header with format-specific styling.

    Args:
        width: PlayResX (video width)
        height: PlayResY (video height)
        font_size: Font size for subtitles
        margin_bottom: Bottom margin in pixels

    Returns:
        ASS header string
    """
    # Alignment 8 = Top-center (text aligns from top of subtitle area)
    # This makes 1-line subtitles show at "line 1" not "line 3"
    # MarginV for alignment 8 = distance from TOP of screen
    margin_from_top = height - margin_bottom

    return f"""[Script Info]
Title: CoClip Subtitles
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2.5,0,8,10,10,{margin_from_top},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _get_words_for_clip(
    segments: List[TranscriptionSegment],
    clip_start: float,
    clip_end: float,
) -> List[dict]:
    """
    Extract words that fall within a clip's time range.

    Args:
        segments: Full transcription segments with word timestamps
        clip_start: Clip start time in seconds (absolute)
        clip_end: Clip end time in seconds (absolute)

    Returns:
        List of word dicts with timestamps adjusted to clip-relative time
    """
    clip_words = []

    for segment in segments:
        # Skip segments outside clip range
        if segment.end < clip_start or segment.start > clip_end:
            continue

        if not segment.words:
            # Fallback: use segment text if no word-level timestamps
            clip_words.append(
                {
                    "word": segment.text,
                    "start": max(0, segment.start - clip_start),
                    "end": min(clip_end - clip_start, segment.end - clip_start),
                }
            )
            continue

        for word in segment.words:
            # Check if word is within clip range
            if word.end < clip_start or word.start > clip_end:
                continue

            # Adjust timestamp to be clip-relative (0-based)
            clip_words.append(
                {
                    "word": word.word,
                    "start": max(0, word.start - clip_start),
                    "end": min(clip_end - clip_start, word.end - clip_start),
                }
            )

    return clip_words


def _group_words_into_lines(
    words: List[dict],
    max_words_per_line: int = 5,
    max_gap_seconds: float = 1.5,
) -> List[List[dict]]:
    """
    Group words into display lines for natural-looking subtitles.

    Rules:
    - Max N words per line
    - Break on long pauses (>1.5s gap)

    Args:
        words: List of word dicts with start/end times
        max_words_per_line: Maximum words per subtitle line
        max_gap_seconds: Maximum gap before forcing line break

    Returns:
        List of word groups (lines)
    """
    if not words:
        return []

    lines = []
    current_line = [words[0]]

    for i in range(1, len(words)):
        word = words[i]
        prev_word = words[i - 1]

        # Check if we need a line break
        gap = word["start"] - prev_word["end"]
        should_break = len(current_line) >= max_words_per_line or gap > max_gap_seconds

        if should_break:
            lines.append(current_line)
            current_line = [word]
        else:
            current_line.append(word)

    if current_line:
        lines.append(current_line)

    return lines


def generate_ass_subtitle(
    segments: List[TranscriptionSegment],
    clip_start: float,
    clip_end: float,
    output_path: str,
    job_id: str = "",
    style: str = "word_highlight",
    video_width: int = 1080,
    video_height: int = 1920,
    font_size: int = 28,
    margin_bottom: int = 180,
) -> Optional[str]:
    """
    Generate ASS subtitle file for a clip with word-by-word highlighting.

    Creates TikTok/Reels style animated captions where each word
    lights up as it's spoken.

    Args:
        segments: Full transcription segments from WhisperX
        clip_start: Clip start time (absolute seconds)
        clip_end: Clip end time (absolute seconds)
        output_path: Path to save .ass file
        job_id: Job ID for logging
        style: Subtitle style ("word_highlight" or "simple")

    Returns:
        Path to generated .ass file, or None if failed
    """
    try:
        # Extract words for this clip
        words = _get_words_for_clip(segments, clip_start, clip_end)

        if not words:
            logger.warning(
                f"⚠️ [Job {job_id}] No words found for clip {clip_start:.1f}-{clip_end:.1f}s"
            )
            return None

        # Group words into display lines
        lines = _group_words_into_lines(words)

        # Build ASS dialogue events
        events = []

        if style == "word_highlight":
            # Word-by-word highlight style (TikTok-like)
            for line_words in lines:
                line_start = line_words[0]["start"]
                line_end = line_words[-1]["end"]

                # Build line with per-word karaoke timing
                # Using \kf (fill effect) for smooth word highlighting
                text_parts = []
                for w in line_words:
                    word_duration_cs = int((w["end"] - w["start"]) * 100)
                    # \kf = karaoke fill, duration in centiseconds
                    word_text = (
                        w["word"].upper() if settings.SUBTITLE_UPPERCASE else w["word"]
                    )
                    escaped = _escape_ass_text(word_text)
                    text_parts.append(f"{{\\kf{word_duration_cs}}}{escaped}")

                line_text = " ".join(text_parts)

                event = (
                    f"Dialogue: 0,"
                    f"{_seconds_to_ass_time(line_start)},"
                    f"{_seconds_to_ass_time(line_end)},"
                    f"Default,,0,0,0,,"
                    f"{line_text}"
                )
                events.append(event)
        else:
            # Simple style - just show text
            for line_words in lines:
                line_start = line_words[0]["start"]
                line_end = line_words[-1]["end"]

                # Apply uppercase if enabled
                if settings.SUBTITLE_UPPERCASE:
                    line_text = " ".join(
                        _escape_ass_text(w["word"].upper()) for w in line_words
                    )
                else:
                    line_text = " ".join(
                        _escape_ass_text(w["word"]) for w in line_words
                    )

                event = (
                    f"Dialogue: 0,"
                    f"{_seconds_to_ass_time(line_start)},"
                    f"{_seconds_to_ass_time(line_end)},"
                    f"Default,,0,0,0,,"
                    f"{line_text}"
                )
                events.append(event)

        # Write ASS file
        ass_header = _generate_ass_header(
            width=video_width,
            height=video_height,
            font_size=font_size,
            margin_bottom=margin_bottom,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ass_header)
            for event in events:
                f.write(event + "\n")

        logger.info(
            f"📝 [Job {job_id}] Generated subtitle: {output_path} "
            f"({len(events)} lines, {len(words)} words)"
        )
        return output_path

    except Exception as e:
        logger.error(
            f"❌ [Job {job_id}] Subtitle generation failed: {e}", exc_info=True
        )
        return None
