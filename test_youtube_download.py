"""
Test YouTube Download - Verify yt-dlp can download videos from YouTube URL.
"""

import os
import sys
import time

OUTPUT_DIR = "tests/download_output"
# Ganti URL ini dengan video YouTube yang mau di-test
TEST_URL = "https://www.youtube.com/watch?v=vh5VbvP0dPM"


def test_ytdlp_installed():
    """Check if yt-dlp is installed."""
    print("\n[Step 1] Checking yt-dlp installation...")
    try:
        import yt_dlp
        print(f"  yt-dlp version: {yt_dlp.version.__version__}")
        return True
    except ImportError:
        print("  ERROR: yt-dlp not installed!")
        print("  Run: pip install yt-dlp")
        return False


def test_video_info(url: str):
    """Fetch video info without downloading."""
    print(f"\n[Step 2] Fetching video info: {url}")
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        print(f"  Title    : {info.get('title', 'N/A')}")
        print(f"  Duration : {info.get('duration', 0)}s ({info.get('duration', 0) / 60:.1f} min)")
        print(f"  Uploader : {info.get('uploader', 'N/A')}")
        print(f"  Resolution: {info.get('width', '?')}x{info.get('height', '?')}")
        print(f"  View count: {info.get('view_count', 'N/A')}")
        return info
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def test_download(url: str, output_dir: str):
    """Download video to output directory."""
    print(f"\n[Step 3] Downloading video to {output_dir}...")
    import yt_dlp

    os.makedirs(output_dir, exist_ok=True)

    output_template = os.path.join(output_dir, "%(title).50s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
        "progress_hooks": [_progress_hook],
    }

    try:
        t0 = time.time()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)

            # yt-dlp might merge to mp4
            if not filename.endswith(".mp4"):
                mp4_name = os.path.splitext(filename)[0] + ".mp4"
                if os.path.exists(mp4_name):
                    filename = mp4_name

        elapsed = time.time() - t0

        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / 1024 / 1024
            print(f"\n  [OK] Downloaded: {filename}")
            print(f"  Size: {file_size:.1f} MB")
            print(f"  Time: {elapsed:.1f}s")
            return filename
        else:
            print(f"\n  [FAIL] File not found after download: {filename}")
            return None

    except Exception as e:
        print(f"\n  [FAIL] Download error: {e}")
        return None


def _progress_hook(d):
    """Progress callback for yt-dlp."""
    if d["status"] == "downloading":
        pct = d.get("_percent_str", "?%")
        speed = d.get("_speed_str", "?")
        eta = d.get("_eta_str", "?")
        print(f"\r  Downloading: {pct} | Speed: {speed} | ETA: {eta}", end="", flush=True)
    elif d["status"] == "finished":
        print(f"\n  Download finished, merging...")


def main():
    url = TEST_URL

    # Allow custom URL from command line
    if len(sys.argv) > 1:
        url = sys.argv[1]

    print("=" * 60)
    print("  YouTube Download Test (yt-dlp)")
    print("=" * 60)

    # Step 1: Check yt-dlp
    if not test_ytdlp_installed():
        return

    # Step 2: Fetch info
    info = test_video_info(url)
    if not info:
        return

    # Step 3: Download
    filepath = test_download(url, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 60)
    if filepath:
        print(f"  SUCCESS! Video downloaded to: {filepath}")
    else:
        print("  FAILED! Check errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
