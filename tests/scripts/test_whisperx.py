"""
WhisperX Benchmark Test - Test transcription pipeline steps individually.

Measures timing and VRAM usage for each step:
  1. Model loading
  2. Audio loading
  3. Transcription (with different batch_sizes)
  4. Alignment
  5. Diarization (optional)

Usage:
  python test_whisperx.py
  python test_whisperx.py --no-diarize
  python test_whisperx.py --batch-sizes 2,4,6,8
  python test_whisperx.py --compute-type int8
  python test_whisperx.py --video path/to/video.mp4
"""

import warnings
warnings.filterwarnings("ignore", message=".*An output with one or more elements was resized.*")

import os
import sys
import time
import argparse

# --- PyTorch 2.6+ Workaround (sama seperti transcriber.py) ---
import torch
import torch.serialization
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    try:
        from omegaconf import DictConfig, ListConfig
        torch.serialization.add_safe_globals([DictConfig, ListConfig])
    except:
        pass
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# --- Patch huggingface_hub deprecated 'use_auth_token' ---
import huggingface_hub
_original_hf_hub_download = huggingface_hub.hf_hub_download
def _patched_hf_hub_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    return _original_hf_hub_download(*args, **kwargs)
huggingface_hub.hf_hub_download = _patched_hf_hub_download
# --------------------------------------------------------------

SOURCE_VIDEO = "tests/video/sample_video_1.mp4"


def get_vram_info():
    """Get current GPU VRAM usage."""
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            used = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            return {
                "total": total,
                "used": used,
                "reserved": reserved,
                "free": total - reserved,
            }
    except Exception:
        pass
    return None


def print_vram(label=""):
    info = get_vram_info()
    if info:
        print(f"  VRAM {label}: {info['used']:.2f}GB used / {info['reserved']:.2f}GB reserved / {info['total']:.1f}GB total")


def print_header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="WhisperX Benchmark Test")
    parser.add_argument("--video", default=SOURCE_VIDEO, help="Video file path")
    parser.add_argument("--no-diarize", action="store_true", help="Skip diarization")
    parser.add_argument("--batch-sizes", default="4,6,8", help="Comma-separated batch sizes to test")
    parser.add_argument("--compute-type", default=None, help="Override compute type (float16, int8)")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    video_path = args.video
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    run_diarize = not args.no_diarize

    print_header("WhisperX Benchmark Test")

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return

    file_size = os.path.getsize(video_path) / 1024 / 1024
    print(f"  Video: {video_path} ({file_size:.1f} MB)")
    print(f"  Device: {args.device}")
    print(f"  Batch sizes to test: {batch_sizes}")
    print(f"  Diarization: {'enabled' if run_diarize else 'disabled'}")

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            print(f"  GPU: {gpu.name}, VRAM: {gpu.total_memory / 1024**3:.1f}GB")
        else:
            print("  GPU: Not available (using CPU)")
    except ImportError:
        print("  GPU: PyTorch not installed")

    import whisperx

    # ── Step 1: Load Audio ─────────────────────────────────────────
    print_header("Step 1: Load Audio")
    print_vram("before")

    t0 = time.time()
    audio = whisperx.load_audio(video_path)
    audio_time = time.time() - t0

    duration_sec = len(audio) / 16000  # whisperx uses 16kHz
    print(f"  Audio loaded in {audio_time:.2f}s")
    print(f"  Audio duration: {duration_sec:.1f}s ({duration_sec/60:.1f} min)")
    print_vram("after")

    # ── Step 2: Load Model ─────────────────────────────────────────
    print_header("Step 2: Load WhisperX Model")

    from app.core.config import settings
    model_name = settings.WHISPER_MODEL
    compute_type = args.compute_type or settings.WHISPER_COMPUTE_TYPE

    print(f"  Model: {model_name}")
    print(f"  Compute type: {compute_type}")
    print_vram("before")

    t0 = time.time()
    model = whisperx.load_model(
        model_name,
        args.device,
        compute_type=compute_type,
    )
    model_time = time.time() - t0

    print(f"  Model loaded in {model_time:.2f}s")
    print_vram("after")

    # ── Step 3: Transcription Benchmark ────────────────────────────
    print_header("Step 3: Transcription Benchmark")

    results = {}
    best_batch = None
    best_time = float("inf")

    for batch_size in batch_sizes:
        print(f"\n  --- batch_size={batch_size} ---")
        print_vram("before")

        try:
            import torch, gc
            gc.collect()
            torch.cuda.empty_cache()

            t0 = time.time()
            result = model.transcribe(audio, batch_size=batch_size)
            elapsed = time.time() - t0

            lang = result["language"]
            segments = len(result["segments"])
            speed_ratio = duration_sec / elapsed

            results[batch_size] = {
                "time": elapsed,
                "language": lang,
                "segments": segments,
                "speed_ratio": speed_ratio,
            }

            print(f"  Time: {elapsed:.2f}s ({speed_ratio:.1f}x realtime)")
            print(f"  Language: {lang}, Segments: {segments}")
            print_vram("after")

            if elapsed < best_time:
                best_time = elapsed
                best_batch = batch_size

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM! batch_size={batch_size} too large for this GPU")
                import torch, gc
                gc.collect()
                torch.cuda.empty_cache()
                results[batch_size] = {"time": None, "error": "OOM"}
            else:
                raise

    # Keep last successful result for alignment
    transcription_result = result

    # ── Step 4: Alignment ──────────────────────────────────────────
    print_header("Step 4: Word-Level Alignment")
    print_vram("before")

    lang = transcription_result["language"]
    print(f"  Language: {lang}")

    # Handle Indonesian alignment model
    align_kwargs = {"language_code": lang, "device": args.device}
    if lang == "id":
        align_kwargs["model_name"] = "indonesian-nlp/wav2vec2-large-xlsr-indonesian"
        print(f"  Using Indonesian alignment model")

    t0 = time.time()
    align_model, align_metadata = whisperx.load_align_model(**align_kwargs)
    align_load_time = time.time() - t0
    print(f"  Align model loaded in {align_load_time:.2f}s")
    print_vram("after align model load")

    t0 = time.time()
    aligned_result = whisperx.align(
        transcription_result["segments"],
        align_model,
        align_metadata,
        audio,
        args.device,
        return_char_alignments=False,
    )
    align_time = time.time() - t0

    total_words = sum(
        len(seg.get("words", []))
        for seg in aligned_result.get("segments", [])
    )
    print(f"  Alignment done in {align_time:.2f}s")
    print(f"  Words aligned: {total_words}")
    print_vram("after alignment")

    # Free align model
    del align_model, align_metadata
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    # ── Step 5: Diarization ────────────────────────────────────────
    diarize_time = None
    if run_diarize:
        print_header("Step 5: Speaker Diarization")
        print_vram("before")

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("  SKIPPED: HF_TOKEN not set")
        else:
            try:
                from whisperx.diarize import DiarizationPipeline
                import torch

                t0 = time.time()
                diarize_model = DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=torch.device(args.device),
                )
                diarize_load_time = time.time() - t0
                print(f"  Diarize model loaded in {diarize_load_time:.2f}s")

                print_vram("after diarize model load")

                t0 = time.time()
                diarize_segments = diarize_model(audio)
                diarize_result = whisperx.assign_word_speakers(
                    diarize_segments, aligned_result
                )
                diarize_time = time.time() - t0

                speakers = set(
                    seg.get("speaker", "?")
                    for seg in diarize_result.get("segments", [])
                )
                print(f"  Diarization done in {diarize_time:.2f}s")
                print(f"  Speakers detected: {speakers}")
                print_vram("after diarization")

                del diarize_model
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  Diarization failed: {e}")
    else:
        print_header("Step 5: Speaker Diarization (SKIPPED)")

    # ── Summary ────────────────────────────────────────────────────
    print_header("Summary")

    print(f"\n  Video: {duration_sec:.0f}s ({duration_sec/60:.1f} min)")
    print(f"  Compute type: {compute_type}")
    print(f"  Device: {args.device}")
    print()

    print("  Transcription benchmarks:")
    print(f"  {'Batch':>6} | {'Time':>8} | {'Speed':>10} | {'Status':>6}")
    print(f"  {'-'*6} | {'-'*8} | {'-'*10} | {'-'*6}")
    for bs in batch_sizes:
        r = results.get(bs, {})
        if r.get("error"):
            print(f"  {bs:>6} | {'---':>8} | {'---':>10} | {'OOM':>6}")
        elif r.get("time"):
            print(f"  {bs:>6} | {r['time']:>7.1f}s | {r['speed_ratio']:>8.1f}x | {'OK':>6}")

    if best_batch:
        print(f"\n  Best batch_size: {best_batch} ({best_time:.1f}s)")

    print(f"\n  Pipeline timing:")
    print(f"    Audio load:     {audio_time:.1f}s")
    print(f"    Model load:     {model_time:.1f}s")
    print(f"    Transcription:  {best_time:.1f}s (batch={best_batch})")
    print(f"    Align load:     {align_load_time:.1f}s")
    print(f"    Alignment:      {align_time:.1f}s")
    if diarize_time:
        print(f"    Diarization:    {diarize_time:.1f}s")
        total = audio_time + model_time + best_time + align_load_time + align_time + diarize_time
    else:
        print(f"    Diarization:    skipped")
        total = audio_time + model_time + best_time + align_load_time + align_time

    print(f"    ─────────────────────")
    print(f"    Total:          {total:.1f}s")

    print(f"\n  Recommendations for this GPU:")
    vram = get_vram_info()
    if vram:
        if vram["total"] <= 4:
            print(f"    - compute_type: int8 (save VRAM)")
            print(f"    - batch_size: {best_batch or 4}")
            print(f"    - diarization: disable if not needed (saves ~{diarize_time:.0f}s)" if diarize_time else "")
        elif vram["total"] <= 8:
            print(f"    - compute_type: float16 or int8")
            print(f"    - batch_size: {best_batch or 8}")
        else:
            print(f"    - compute_type: float16")
            print(f"    - batch_size: {best_batch or 16}")


if __name__ == "__main__":
    main()
