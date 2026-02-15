
import os
import sys
import subprocess
from app.utils.subtitle_generator import generate_ass_subtitle
from app.schemas.transcription import TranscriptionSegment, WordTimestamp

# Dummy data: Short Text
SHORT_TEXT = "Halo dunia ini tes pendek"
SHORT_WORDS = [
    WordTimestamp(word="Halo", start=0.5, end=1.0, score=0.9),
    WordTimestamp(word="dunia", start=1.0, end=1.5, score=0.9),
    WordTimestamp(word="ini", start=1.5, end=2.0, score=0.9),
    WordTimestamp(word="tes", start=2.0, end=2.5, score=0.9),
    WordTimestamp(word="pendek", start=2.5, end=3.0, score=0.9),
]
SEGMENTS_SHORT = [
    TranscriptionSegment(
        start=0.5,
        end=3.0,
        text=SHORT_TEXT,
        words=SHORT_WORDS
    )
]

# Dummy data: Single Line (Should show at bottom margin)
SINGLE_TEXT = "Ini cuma satu baris"
SINGLE_WORDS = [
    WordTimestamp(word="Ini", start=0.5, end=1.0, score=0.9),
    WordTimestamp(word="cuma", start=1.0, end=1.5, score=0.9),
    WordTimestamp(word="satu", start=1.5, end=2.0, score=0.9),
    WordTimestamp(word="baris", start=2.0, end=2.5, score=0.9),
]
SEGMENTS_SINGLE = [
    TranscriptionSegment(
        start=0.5,
        end=3.0,
        text=SINGLE_TEXT,
        words=SINGLE_WORDS
    )
]

# Dummy data: Long Text
LONG_TEXT = "Ini adalah contoh kalimat yang sangat panjang untuk menguji apakah generator subtitle bisa memenggal baris dengan benar dan rapi sesuai format TikTok yang standar"
LONG_WORDS = [
    WordTimestamp(word="Ini", start=0.2, end=0.5, score=0.9),
    WordTimestamp(word="adalah", start=0.5, end=0.8, score=0.9),
    WordTimestamp(word="contoh", start=0.8, end=1.2, score=0.9),
    WordTimestamp(word="kalimat", start=1.2, end=1.6, score=0.9),
    WordTimestamp(word="yang", start=1.6, end=1.8, score=0.9),
    WordTimestamp(word="sangat", start=1.8, end=2.2, score=0.9),
    WordTimestamp(word="panjang", start=2.2, end=2.6, score=0.9),
    WordTimestamp(word="untuk", start=2.6, end=2.9, score=0.9),
    WordTimestamp(word="menguji", start=2.9, end=3.3, score=0.9),
    WordTimestamp(word="apakah", start=3.3, end=3.7, score=0.9),
    WordTimestamp(word="generator", start=3.7, end=4.1, score=0.9),
    WordTimestamp(word="subtitle", start=4.1, end=4.5, score=0.9),
    WordTimestamp(word="bisa", start=4.5, end=4.7, score=0.9),
    WordTimestamp(word="memenggal", start=4.7, end=5.2, score=0.9),
    WordTimestamp(word="baris", start=5.2, end=5.5, score=0.9),
    WordTimestamp(word="dengan", start=5.5, end=5.8, score=0.9),
    WordTimestamp(word="benar", start=5.8, end=6.2, score=0.9),
    WordTimestamp(word="dan", start=6.2, end=6.4, score=0.9),
    WordTimestamp(word="rapi", start=6.4, end=6.8, score=0.9),
    WordTimestamp(word="sesuai", start=6.8, end=7.2, score=0.9),
    WordTimestamp(word="format", start=7.2, end=7.6, score=0.9),
    WordTimestamp(word="TikTok", start=7.6, end=8.0, score=0.9),
    WordTimestamp(word="yang", start=8.0, end=8.2, score=0.9),
    WordTimestamp(word="standar", start=8.2, end=8.6, score=0.9),
]
SEGMENTS_LONG = [
    TranscriptionSegment(
        start=0.2,
        end=8.6,
        text=LONG_TEXT,
        words=LONG_WORDS
    )
]

def render_preview(ass_file, output_mp4, duration=5):
    """Render video preview using FFmpeg with black background."""
    # Convert path to absolute and handle specialized ffmpeg escaping for filter
    # FFmpeg filter requires escaping inside the filter string: \ -> / or \\
    # Best to use forward slashes for paths in filters
    ass_path_filter = os.path.abspath(ass_file).replace("\\", "/").replace(":", "\\:")
    
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", f"color=c=0xD3D3D3:s=1080x1920:d={duration}",
        "-vf", f"ass='{ass_path_filter}'",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        output_mp4
    ]
    
    print(f"🎬 Rendering {output_mp4}...")
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        print(f"✅ Generated: {output_mp4}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to render {output_mp4}")
        print(e.stderr.decode() if e.stderr else "No error output")

def main():
    os.makedirs("tests/output", exist_ok=True)
    
    # 1. Short Text (TikTok style)
    ass_short = "tests/output/preview_short.ass"
    generate_ass_subtitle(
        SEGMENTS_SHORT, 0.0, 5.0, ass_short, 
        job_id="test_short",
        font_size=85,      # Default for TikTok 1080p
        margin_bottom=300  # Default for TikTok
    )
    render_preview(ass_short, "tests/output/preview_short.mp4", duration=5)

    # 2. Single Line (Test bottom alignment)
    ass_single = "tests/output/preview_single.ass"
    generate_ass_subtitle(
        SEGMENTS_SINGLE, 0.0, 5.0, ass_single,
        job_id="test_single",
        font_size=85,
        margin_bottom=300
    )
    render_preview(ass_single, "tests/output/preview_single.mp4", duration=5)
    
    # 3. Long Text (TikTok style)
    ass_long = "tests/output/preview_long.ass"
    generate_ass_subtitle(
        SEGMENTS_LONG, 0.0, 10.0, ass_long, 
        job_id="test_long",
        font_size=85,
        margin_bottom=300
    )
    render_preview(ass_long, "tests/output/preview_long.mp4", duration=10)

if __name__ == "__main__":
    main()
