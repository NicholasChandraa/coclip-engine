"""
Smart Crop Test - Dynamic Face Tracking with Smooth Transitions
Compare center crop vs dynamic smart crop that follows faces smoothly.
"""

import warnings
warnings.filterwarnings("ignore", message=".*An output with one or more elements was resized.*")

import os
import json
import asyncio
import time

SOURCE_VIDEO = "tests/video/sample_video_1.mp4"
OUTPUT_DIR = "tests/smart_crop_output"

TEST_START = 300.0  # 5:00
TEST_END = 360.0    # 6:00
TEST_DURATION = TEST_END - TEST_START

TARGET_W = 1080
TARGET_H = 1920

from app.utils.speaker_detector import TRANSITION_DURATION


def print_header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


async def detect_video_resolution(video_path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", video_path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    data = json.loads(stdout.decode())
    stream = data.get("streams", [{}])[0]
    return stream.get("width", 0), stream.get("height", 0)


def build_crop_filter(input_w, input_h, target_w, target_h, crop_x=None):
    target_ratio = target_w / target_h
    crop_h = input_h
    crop_w = int(input_h * target_ratio)
    if crop_w > input_w:
        crop_w = input_w
        crop_h = int(input_w / target_ratio)
    x = crop_x if crop_x is not None else (input_w - crop_w) // 2
    x = max(0, min(x, input_w - crop_w))
    y = (input_h - crop_h) // 2
    return f"crop={crop_w}:{crop_h}:{x}:{y},scale={target_w}:{target_h}"


def build_keyframe_crop_expr(keyframes, input_w, input_h, target_w, target_h):
    """
    Build FFmpeg crop filter with animated x-position from keyframes.
    Each keyframe is a CropKeyframe(time, crop_x).
    Smoothly interpolates between consecutive keyframes over TRANSITION_DURATION.
    """
    target_ratio = target_w / target_h
    crop_h = input_h
    crop_w = int(input_h * target_ratio)
    if crop_w > input_w:
        crop_w = input_w
        crop_h = int(input_w / target_ratio)
    max_x = input_w - crop_w

    if not keyframes:
        x = (input_w - crop_w) // 2
        return f"crop={crop_w}:{crop_h}:{x}:0,scale={target_w}:{target_h}"

    if len(keyframes) == 1:
        x = max(0, min(keyframes[0].crop_x, max_x))
        return f"crop={crop_w}:{crop_h}:{x}:0,scale={target_w}:{target_h}"

    # Reduce keyframes: only keep ones where crop_x changes by >= 5px
    # FFmpeg crashes with deeply nested expressions, so cap at 60
    MAX_EXPR_KEYFRAMES = 60

    reduced = [keyframes[0]]
    for kf in keyframes[1:]:
        if abs(kf.crop_x - reduced[-1].crop_x) >= 5:
            reduced.append(kf)
    if reduced[-1].time != keyframes[-1].time:
        reduced.append(keyframes[-1])

    # Subsample evenly if still too many
    if len(reduced) > MAX_EXPR_KEYFRAMES:
        step = len(reduced) / (MAX_EXPR_KEYFRAMES - 1)
        subsampled = [reduced[int(i * step)] for i in range(MAX_EXPR_KEYFRAMES - 1)]
        subsampled.append(reduced[-1])
        reduced = subsampled

    if len(reduced) == 1:
        x = max(0, min(reduced[0].crop_x, max_x))
        return f"crop={crop_w}:{crop_h}:{x}:0,scale={target_w}:{target_h}"

    print(f"    Expression keyframes: {len(reduced)} (reduced from {len(keyframes)})")

    # Build nested FFmpeg expression from reduced keyframes
    x_expr = str(max(0, min(reduced[0].crop_x, max_x)))

    for i in range(1, len(reduced)):
        kf = reduced[i]
        prev_kf = reduced[i - 1]
        x_cur = max(0, min(kf.crop_x, max_x))
        x_prev = max(0, min(prev_kf.crop_x, max_x))
        t_start = kf.time

        if x_cur == x_prev:
            continue

        t_trans_end = t_start + TRANSITION_DURATION
        progress = f"(t-{t_start:.3f})/{TRANSITION_DURATION:.3f}"
        lerp = f"{x_prev}+({x_cur}-{x_prev})*min(1\\,max(0\\,{progress}))"

        x_expr = (
            f"if(gte(t\\,{t_start:.3f})\\,"
            f"if(gte(t\\,{t_trans_end:.3f})\\,{x_cur}\\,{lerp})\\,"
            f"{x_expr})"
        )

    crop_filter = f"crop={crop_w}:{crop_h}:'{x_expr}':0,scale={target_w}:{target_h}"
    return crop_filter


async def cut_clip(input_path, output_path, start, duration, crop_filter=None, label=""):
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", input_path, "-t", str(duration)]
    if crop_filter:
        cmd.extend(["-vf", crop_filter])
    cmd.extend([
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", output_path,
    ])
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        print(f"  [FAIL] {label}: {stderr.decode()[-300:]}")
        return False
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  [OK] {label}: {output_path} ({size_mb:.1f} MB)")
    return True


async def main():
    print_header("Dynamic Smart Crop Test (Keyframe-based Smooth Transitions)")

    if not os.path.exists(SOURCE_VIDEO):
        print(f"[ERROR] Video not found: {SOURCE_VIDEO}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Video info
    print_header("Step 1: Video Info")
    input_w, input_h = await detect_video_resolution(SOURCE_VIDEO)
    target_ratio = TARGET_W / TARGET_H
    crop_w_in_source = int(input_h * target_ratio)
    center_x = (input_w - crop_w_in_source) // 2
    print(f"  Resolution : {input_w}x{input_h}")
    print(f"  Test window: {TEST_START}s - {TEST_END}s ({TEST_DURATION}s)")
    print(f"  Crop width : {crop_w_in_source}px (center_x={center_x})")

    # Step 2: Dynamic face detection
    print_header("Step 2: Dynamic Face Detection (Keyframe-based)")
    from app.utils.speaker_detector import SpeakerDetector, TRANSITION_DURATION

    detector = SpeakerDetector(device="cuda")
    t0 = time.time()
    detector.load()
    print(f"  Detector: {'S3FD' if detector.face_detector else 'OpenCV'} ({time.time()-t0:.2f}s)")

    test_clips = [{"start_time": TEST_START, "end_time": TEST_END}]

    t0 = time.time()
    positions = detector.detect_active_speakers(
        video_path=SOURCE_VIDEO,
        clip_candidates=test_clips,
        frame_width=input_w,
        frame_height=input_h,
        target_width=crop_w_in_source,
        target_height=input_h,
        sample_interval=0.5,
    )
    detect_time = time.time() - t0

    pos = positions[0]
    print(f"  Detection: {detect_time:.2f}s")
    print(f"  Keyframes: {len(pos.keyframes)}")
    print(f"  Primary crop_x: {pos.crop_x} (center={center_x})")
    print(f"  Confidence: {pos.confidence:.3f}")
    print(f"  Fallback: {pos.is_fallback}")
    print()

    # Show keyframe details
    for i, kf in enumerate(pos.keyframes):
        diff = abs(kf.crop_x - center_x)
        direction = "RIGHT" if kf.crop_x > center_x else ("LEFT" if kf.crop_x < center_x else "CENTER")
        print(f"  KF {i+1:3d}: t={kf.time:5.1f}s  crop_x={kf.crop_x:3d} ({direction:>6} {diff:3d}px)")

    detector.unload()

    # Step 3: Generate clips
    print_header("Step 3: Generate Clips")

    # 3a: Center crop (static)
    center_filter = build_crop_filter(input_w, input_h, TARGET_W, TARGET_H)
    center_output = os.path.join(OUTPUT_DIR, "center_crop.mp4")
    print(f"\n  Center: {center_filter}")
    await cut_clip(SOURCE_VIDEO, center_output, TEST_START, TEST_DURATION,
                   crop_filter=center_filter, label="Center Crop")

    # 3b: Dynamic smart crop with keyframe-based smooth transitions
    smart_output = os.path.join(OUTPUT_DIR, "smart_crop.mp4")
    smart_filter = build_keyframe_crop_expr(
        pos.keyframes, input_w, input_h, TARGET_W, TARGET_H
    )
    print(f"\n  Smart: {len(pos.keyframes)} keyframes, transition={TRANSITION_DURATION}s")
    await cut_clip(SOURCE_VIDEO, smart_output, TEST_START, TEST_DURATION,
                   crop_filter=smart_filter, label="Smart Crop (keyframes)")

    # 3c: Reference (no crop)
    nocrop_output = os.path.join(OUTPUT_DIR, "no_crop_reference.mp4")
    await cut_clip(SOURCE_VIDEO, nocrop_output, TEST_START, TEST_DURATION,
                   label="No Crop (ref)")

    # 3d: Side-by-side
    if os.path.exists(center_output) and os.path.exists(smart_output):
        print(f"\n  Generating side-by-side...")
        sbs_output = os.path.join(OUTPUT_DIR, "sidebyside.mp4")
        center_abs = os.path.abspath(center_output).replace("\\", "/")
        smart_abs = os.path.abspath(smart_output).replace("\\", "/")
        sbs_cmd = [
            "ffmpeg", "-y",
            "-i", center_abs,
            "-i", smart_abs,
            "-filter_complex",
            "[0:v]pad=iw+4:ih:0:0:red[left];"
            "[1:v]pad=iw+4:ih:4:0:lime[right];"
            "[left][right]hstack=inputs=2[v]",
            "-map", "[v]", "-map", "0:a",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            sbs_output,
        ]
        proc = await asyncio.create_subprocess_exec(
            *sbs_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode == 0:
            size_mb = os.path.getsize(sbs_output) / 1024 / 1024
            print(f"  [OK] Side-by-side: {sbs_output} ({size_mb:.1f} MB)")
        else:
            print(f"  [FAIL] Side-by-side: {stderr.decode()[-200:]}")

    # Summary
    print_header("Done!")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".mp4"):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024 / 1024
            print(f"    {f} ({size:.1f} MB)")
    print(f"\n  RED = CENTER, GREEN = SMART (keyframe smooth transitions)")
    print(f"  Open sidebyside.mp4 to compare!")


if __name__ == "__main__":
    asyncio.run(main())
