"""
Subtitle Preview Tool

Quick test tool to preview subtitle styling without processing full video.
Generates test ASS file and renders on a sample clip.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.subtitle_generator import _generate_ass_header


def generate_test_subtitle(
    output_path: str = "test_subtitle.ass",
    style: str = "modern",  # modern, tiktok, minimal, classic
):
    """
    Generate test subtitle file with sample text.

    Styles:
    - modern: All caps, bold white, strong shadow
    - tiktok: White text, black outline, no karaoke
    - minimal: Simple white, subtle outline
    - classic: Current yellow karaoke style
    """

    # Style definitions
    styles = {
        "modern": {
            "header": """[Script Info]
Title: Modern Subtitle Preview
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,32,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,3,2,10,10,180,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""",
            "sample_text": "THIS IS A MODERN SUBTITLE\nCLEAN AND BOLD",
        },
        "tiktok": {
            "header": """[Script Info]
Title: TikTok Subtitle Preview
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,36,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,180,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""",
            "sample_text": "This is TikTok style subtitle\nClean white text",
        },
        "minimal": {
            "header": """[Script Info]
Title: Minimal Subtitle Preview
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,30,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,180,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""",
            "sample_text": "Minimal subtitle style\nSubtle and clean",
        },
        "classic": {
            "header": """[Script Info]
Title: Classic Karaoke Preview
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,28,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2.5,0,2,10,10,180,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""",
            "sample_text": "{\\kf50}Classic {\\kf50}yellow {\\kf50}karaoke",
        },
        "clipper": {
            "header": """[Script Info]
Title: Video Clipper Style
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,85,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2.5,0,2,10,10,280,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""",
            "sample_text": "VIDEO CLIPPER STYLE\nCLEAN AND PROFESSIONAL",
        },
    }

    style_config = styles.get(style, styles["modern"])

    # Write ASS file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(style_config["header"])
        # Add 3 sample dialogue lines at different times
        f.write(
            "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,"
            + style_config["sample_text"]
            + "\n"
        )
        f.write(
            "Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,SAMPLE SUBTITLE TEXT\n"
        )
        f.write(
            "Dialogue: 0,0:00:07.00,0:00:09.00,Default,,0,0,0,,Preview how it looks here\n"
        )

    print(f"✅ Generated test subtitle: {output_path}")
    print(f"   Style: {style}")
    return output_path


def render_preview(
    video_path: str,
    subtitle_path: str,
    output_path: str = "preview_output.mp4",
):
    """
    Render video with subtitle preview using FFmpeg.

    Args:
        video_path: Input video file (any existing clip)
        subtitle_path: ASS subtitle file to test
        output_path: Output preview file
    """
    import subprocess

    # Escape paths for FFmpeg on Windows
    escaped_subtitle = subtitle_path.replace("\\", "/").replace(":", "\\:")

    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-t",
        "10",  # Only first 10 seconds
        "-vf",
        f"ass='{escaped_subtitle}'",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-y",
        output_path,
    ]

    print(f"🎬 Rendering preview...")
    print(f"   Input: {video_path}")
    print(f"   Subtitle: {subtitle_path}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Preview rendered: {output_path}")
        print(f"   Open this file to see subtitle styling!")
        return output_path
    else:
        print(f"❌ FFmpeg failed:")
        print(result.stderr[-500:])
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview subtitle styling")
    parser.add_argument(
        "--style",
        choices=["modern", "tiktok", "minimal", "classic", "clipper"],
        default="modern",
        help="Subtitle style to preview",
    )
    parser.add_argument(
        "--video",
        required=False,
        help="Video file to use for preview (optional, uses latest clip if not specified)",
    )
    parser.add_argument(
        "--output",
        default="subtitle_preview.mp4",
        help="Output preview file name",
    )

    args = parser.parse_args()

    # Generate test subtitle
    subtitle_file = generate_test_subtitle(
        output_path=f"test_subtitle_{args.style}.ass",
        style=args.style,
    )

    # Find video to use
    if args.video:
        video_input = args.video
    else:
        # Try to find a recent clip
        clips_dir = Path("clips")
        if clips_dir.exists():
            clip_files = list(clips_dir.glob("*/clip_1.mp4"))
            if clip_files:
                video_input = str(clip_files[-1])  # Latest clip
                print(f"📹 Using clip: {video_input}")
            else:
                print("❌ No clips found. Please specify --video path")
                sys.exit(1)
        else:
            print("❌ No clips directory. Please specify --video path")
            sys.exit(1)

    # Render preview
    render_preview(video_input, subtitle_file, args.output)

    print("\n" + "=" * 60)
    print("🎨 SUBTITLE STYLE GUIDE")
    print("=" * 60)
    print("\nColor Codes (ASS format, &HAABBGGRR):")
    print("  White:  &H00FFFFFF")
    print("  Black:  &H00000000")
    print("  Yellow: &H0000FFFF")
    print("  Red:    &H000000FF")
    print("  Green:  &H0000FF00")
    print("  Blue:   &H00FF0000")
    print("\nFont Parameters:")
    print("  Bold=-1 (bold), 0 (normal)")
    print("  Fontsize=28-40 (bigger = easier to read)")
    print("  Outline=2-4 (text outline thickness)")
    print("  Shadow=0-3 (drop shadow)")
    print("\nTry different styles:")
    print("  python test_subtitle_preview.py --style modern")
    print("  python test_subtitle_preview.py --style tiktok")
    print("  python test_subtitle_preview.py --style minimal")
