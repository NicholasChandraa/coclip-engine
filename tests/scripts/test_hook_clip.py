"""
Hook Clip Test - Test hook intro creation + concat with main clip.

Tests the full hook flow:
1. Cut a short test clip from source video
2. Generate TTS audio (or use silent audio)
3. Create blurred hook intro with text overlay
4. Concat intro + main clip
5. Verify output plays correctly (no slow motion, audio sync)

Usage:
    python -m tests.scripts.test_hook_clip
    python -m tests.scripts.test_hook_clip --video tests/video/sample_video_1.mp4
    python -m tests.scripts.test_hook_clip --video tests/video/sample_video_1.mp4 --tts
    python -m tests.scripts.test_hook_clip --skip-cut --clip path/to/existing_clip.mp4
"""

import os
import sys
import json
import asyncio
import argparse
import time

# ── Config ──────────────────────────────────────────────────────
SOURCE_VIDEO = "./tests/smart_crop_output/sidebyside.mp4"
OUTPUT_DIR = "tests/hook_clip_output"

TEST_START = 0.0  # 0:00
TEST_END = 30.0  # 0:30
TARGET_W = 1080
TARGET_H = 1920

HOOK_TEXT = "Kamu harus tau ini! Rahasia yang jarang orang bahas."


def print_header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def print_step(num, msg):
    print(f"\n  [{num}] {msg}")


async def detect_video_info(video_path: str) -> dict:
    """Detect fps, resolution, audio sample rate from video."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "json",
        video_path,
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()
    data = json.loads(stdout.decode())
    stream = data.get("streams", [{}])[0]

    # Parse fps
    r_str = stream.get("r_frame_rate", "30/1")
    num, den = r_str.split("/")
    fps = float(num) / float(den)

    # Audio sample rate
    cmd_audio = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,codec_name",
        "-of",
        "json",
        video_path,
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd_audio,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_a, _ = await process.communicate()
    data_a = json.loads(stdout_a.decode())
    a_stream = data_a.get("streams", [{}])[0]

    return {
        "width": stream.get("width", 0),
        "height": stream.get("height", 0),
        "fps": fps,
        "audio_sample_rate": a_stream.get("sample_rate", "N/A"),
        "audio_codec": a_stream.get("codec_name", "N/A"),
    }


async def cut_test_clip(source: str, output: str, start: float, end: float) -> bool:
    """Cut a short clip from source video (same as editing_node)."""
    duration = end - start
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        source,
        "-t",
        str(duration),
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-cq",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        output,
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        print(f"    ERROR: {stderr.decode()[-300:]}")
        return False
    return True


async def create_hook_intro(
    clip_path: str,
    hook_text: str,
    output_dir: str,
    audio_path: str = "",
    volume: float = 1.0,
) -> str:
    """
    Create hook intro - mirrors editing_node._create_hook_intro logic.
    Returns path to intro video.
    """
    frame_path = os.path.join(output_dir, "hook_frame_test.png")
    intro_path = os.path.join(output_dir, "hook_intro_test.mp4")

    # Step 1: Extract first frame + blur
    print_step("3a", "Extracting blurred frame from clip...")
    extract_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        clip_path,
        "-vframes",
        "1",
        "-vf",
        "boxblur=20:5",
        frame_path,
    ]
    process = await asyncio.create_subprocess_exec(
        *extract_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await process.communicate()
    if process.returncode != 0 or not os.path.exists(frame_path):
        print("    ERROR: Failed to extract frame")
        return ""

    # Step 2: Detect clip fps
    info = await detect_video_info(clip_path)
    clip_fps = info["fps"]
    print(
        f"       Main clip: {info['width']}x{info['height']} @ {clip_fps:.2f}fps, audio={info['audio_sample_rate']}Hz"
    )

    # Step 3: Determine duration
    if audio_path and os.path.exists(audio_path):
        import wave

        with wave.open(audio_path, "rb") as wf:
            tts_duration = wf.getnframes() / wf.getframerate()
        hook_duration = 1.0 + tts_duration + 1.0  # 1.0s padding before + after
        print(
            f"       TTS audio: {audio_path} (tts={tts_duration:.1f}s, total={hook_duration:.1f}s)"
        )
    else:
        hook_duration = 4.0
        print(f"       No TTS audio, using silent {hook_duration:.1f}s")

    # Step 4: Text wrapping
    safe_text = hook_text.upper().replace("'", "\u2019").replace(":", "\\:")
    # Arial Black is wide -> 15 chars fits in safe area (162px margins)
    max_chars = 15
    words = safe_text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    wrapped = "\n".join(lines)

    # Match production: 85px / 1920px ~= 0.044
    font_size = int(TARGET_H * 0.044)
    drawtext = (
        f"drawtext=text='{wrapped}'"
        f":fontfile='C\\:/Windows/Fonts/ariblk.ttf'"
        f":fontsize={font_size}"
        f":fontcolor=white"
        f":borderw=4.5:bordercolor=black"
        f":x=(w-text_w)/2:y=(h-text_h)/2"
    )

    # Step 5: Build FFmpeg command
    has_audio = audio_path and os.path.exists(audio_path)

    # Sound effect file
    sfx_path = os.path.join(os.getcwd(), "music", "sound-effect-1.mp3")
    has_sfx = os.path.exists(sfx_path)
    if has_sfx:
        print(f"       SFX: {sfx_path} (trimmed to 2s)")

    # Inputs: [0]=image, [1]=TTS or silence, [2]=SFX (optional)
    intro_cmd = ["ffmpeg", "-y", "-loop", "1", "-i", frame_path]
    if has_audio:
        intro_cmd.extend(["-i", audio_path])
    else:
        intro_cmd.extend(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"])

    if has_sfx:
        intro_cmd.extend(["-i", sfx_path])

    # Audio mixing
    if has_sfx and has_audio:
        audio_filter = (
            f"[1:a]adelay=1000|1000,volume={volume},aresample=48000[tts];"
            "[2:a]atrim=0:2,volume=0.7,aresample=48000[sfx];"
            "[tts][sfx]amix=inputs=2:duration=longest[aout]"
        )
        intro_cmd.extend(
            ["-filter_complex", audio_filter, "-map", "0:v", "-map", "[aout]"]
        )
    elif has_sfx:
        audio_filter = (
            "[2:a]atrim=0:2,volume=0.7,aresample=48000[sfx];"
            f"[1:a]volume={volume}[tts_v];"
            "[tts_v][sfx]amix=inputs=2:duration=longest[aout]"
        )
        intro_cmd.extend(
            ["-filter_complex", audio_filter, "-map", "0:v", "-map", "[aout]"]
        )
    elif has_audio:
        intro_cmd.extend(["-af", f"adelay=1000|1000,volume={volume}"])

    intro_cmd.extend(
        [
            "-vf",
            drawtext,
            "-t",
            str(hook_duration),
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-cq",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ar",
            "48000",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(clip_fps),
            "-shortest",
            "-movflags",
            "+faststart",
            intro_path,
        ]
    )

    print_step(
        "3b", f"Creating intro video ({hook_duration:.1f}s, {clip_fps:.2f}fps)..."
    )
    process = await asyncio.create_subprocess_exec(
        *intro_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        print(f"    ERROR: {stderr.decode()[-500:]}")
        return ""

    print(f"       Intro saved: {intro_path}")
    return intro_path


async def concat_intro_and_clip(
    intro_path: str, clip_path: str, output_path: str
) -> bool:
    """Concat hook intro + main clip - mirrors editing_node._concat_intro_and_clip."""
    concat_list = output_path + ".concat.txt"

    with open(concat_list, "w", encoding="utf-8") as f:
        f.write(f"file '{os.path.abspath(intro_path).replace(os.sep, '/')}'\n")
        f.write(f"file '{os.path.abspath(clip_path).replace(os.sep, '/')}'\n")

    concat_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_list,
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-cq",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "48000",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]

    process = await asyncio.create_subprocess_exec(
        *concat_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()

    # Cleanup
    if os.path.exists(concat_list):
        os.remove(concat_list)

    if process.returncode != 0:
        print(f"    ERROR: {stderr.decode()[-500:]}")
        return False
    return True


async def generate_tts_audio(text: str, output_dir: str) -> str:
    """Generate TTS audio using Piper or Coqui (same as pipeline)."""
    try:
        sys.path.insert(0, os.getcwd())
        from app.utils.tts_engine import TTSEngine

        audio_path = os.path.join(output_dir, "hook_tts_test.wav")
        tts = TTSEngine()
        result = tts.synthesize(text, "id", audio_path)
        tts.unload()
        if result:
            print(f"       TTS generated: {result}")
            return result

        else:
            print("       TTS returned empty (unsupported language?)")
            return ""
    except Exception as e:
        print(f"       TTS failed: {e}")
        return ""


async def main():
    parser = argparse.ArgumentParser(description="Test hook intro + clip concat")
    parser.add_argument("--video", default=SOURCE_VIDEO, help="Source video path")
    parser.add_argument(
        "--start", type=float, default=TEST_START, help="Clip start time (seconds)"
    )
    parser.add_argument(
        "--end", type=float, default=TEST_END, help="Clip end time (seconds)"
    )
    parser.add_argument("--text", default=HOOK_TEXT, help="Hook text to display")
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Generate TTS voiceover (requires piper/coqui)",
    )
    parser.add_argument(
        "--skip-cut", action="store_true", help="Skip cutting, use existing clip"
    )
    parser.add_argument(
        "--clip", default="", help="Path to existing clip (with --skip-cut)"
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--volume", type=float, default=1.0, help="TTS volume multiplier"
    )
    parser.add_argument(
        "--output-name", default="test_final_with_hook.mp4", help="Final filename"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    clip_path = os.path.join(args.output_dir, "test_main_clip.mp4")
    final_path = os.path.join(args.output_dir, args.output_name)

    # ── Step 1: Cut test clip ─────────────────────────────────────
    if args.skip_cut and args.clip:
        clip_path = args.clip
        print_header("SKIP CUT - Using existing clip")
        print(f"  Clip: {clip_path}")
    else:
        print_header("Step 1: Cut test clip from source")
        if not os.path.exists(args.video):
            print(f"  ERROR: Source video not found: {args.video}")
            return

        t0 = time.time()
        print_step(
            1, f"Cutting {args.start:.0f}s - {args.end:.0f}s from {args.video}..."
        )
        success = await cut_test_clip(args.video, clip_path, args.start, args.end)
        if not success:
            print("  FAILED to cut clip")
            return
        elapsed = time.time() - t0
        size_mb = os.path.getsize(clip_path) / 1024 / 1024
        print(f"       Clip saved: {clip_path} ({size_mb:.1f} MB, {elapsed:.1f}s)")

    # ── Step 2: Info main clip ────────────────────────────────────
    print_header("Step 2: Analyze main clip")
    clip_info = await detect_video_info(clip_path)
    print(f"  Resolution: {clip_info['width']}x{clip_info['height']}")
    print(f"  FPS:        {clip_info['fps']:.2f}")
    print(
        f"  Audio:      {clip_info['audio_codec']} @ {clip_info['audio_sample_rate']}Hz"
    )

    # ── Step 3: TTS (optional) ────────────────────────────────────
    audio_path = ""
    if args.tts:
        print_header("Step 3: Generate TTS voiceover")
        print_step(2, f'Synthesizing: "{args.text[:50]}..."')
        t0 = time.time()
        audio_path = await generate_tts_audio(args.text, args.output_dir)
        elapsed = time.time() - t0
        if audio_path:
            print(f"       TTS done in {elapsed:.1f}s")
        else:
            print(f"       TTS failed, will use silent audio")
    else:
        print_header("Step 3: TTS skipped (use --tts to enable)")

    # ── Step 4: Create hook intro ─────────────────────────────────
    print_header("Step 4: Create hook intro")
    t0 = time.time()
    intro_path = await create_hook_intro(
        clip_path, args.text, args.output_dir, audio_path, args.volume
    )
    elapsed = time.time() - t0
    if not intro_path:
        print("  FAILED to create hook intro")
        return
    intro_info = await detect_video_info(intro_path)
    intro_size = os.path.getsize(intro_path) / 1024 / 1024
    print(
        f"       Intro: {intro_info['width']}x{intro_info['height']} @ {intro_info['fps']:.2f}fps"
    )
    print(
        f"       Audio: {intro_info['audio_codec']} @ {intro_info['audio_sample_rate']}Hz"
    )
    print(f"       Size:  {intro_size:.1f} MB ({elapsed:.1f}s)")

    # ── Step 5: Concat ────────────────────────────────────────────
    print_header("Step 5: Concat intro + main clip")
    t0 = time.time()
    print_step(5, "Concatenating...")
    success = await concat_intro_and_clip(intro_path, clip_path, final_path)
    elapsed = time.time() - t0
    if not success:
        print("  FAILED to concat")
        return

    final_info = await detect_video_info(final_path)
    final_size = os.path.getsize(final_path) / 1024 / 1024
    print(
        f"       Final: {final_info['width']}x{final_info['height']} @ {final_info['fps']:.2f}fps"
    )
    print(
        f"       Audio: {final_info['audio_codec']} @ {final_info['audio_sample_rate']}Hz"
    )
    print(f"       Size:  {final_size:.1f} MB ({elapsed:.1f}s)")

    # ── Summary ───────────────────────────────────────────────────
    print_header("SUMMARY")
    print(f'  Hook text:   "{args.text[:60]}..."')
    print(f"  TTS:         {'yes' if audio_path else 'silent'}")
    print(f"  Main clip:   {clip_path}")
    print(f"  Hook intro:  {intro_path}")
    print(f"  Final:       {final_path}")
    print()
    print(f"  >> Open {final_path} to verify:")
    print(f"     - Hook intro plays first (blurred bg + text)")
    print(f"     - No freeze/loading between intro and clip")
    print(f"     - Main clip plays at normal speed (not slow motion)")
    print(f"     - Audio plays throughout (voiceover then original)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
