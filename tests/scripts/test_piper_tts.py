"""
Test Piper TTS - Hook voiceover generation test.

Install: pip install piper-tts

Voice models akan auto-download saat pertama kali dipakai.
Kalau mau manual download: https://huggingface.co/rhasspy/piper-voices

Usage:
    python test_piper_tts.py
    python test_piper_tts.py --lang en
    python test_piper_tts.py --lang zh
    python test_piper_tts.py --text "Custom text here" --lang id
    python test_piper_tts.py --speed 1.2
"""

import argparse
import os
import time
import wave
import sys

# Voice mapping: WhisperX language code → Piper voice model
VOICE_MAP = {
    "id": "id_ID-news_tts-medium",
    "en": "en_US-amy-medium",
    "zh": "zh_CN-huayan-medium",
}

# Sample hook texts per language
SAMPLE_HOOKS = {
    "id": "Ternyata AI bisa berbohong dan kita tidak pernah menyadarinya",
    "en": "You won't believe what AI can do now",
    "zh": "人工智能现在能做的事情你绝对想不到",
}


def download_voice_model(voice_name: str, data_dir: str) -> str:
    """
    Download Piper voice model if not already present.

    Returns path to the .onnx model file.
    """
    model_path = os.path.join(data_dir, f"{voice_name}.onnx")
    config_path = os.path.join(data_dir, f"{voice_name}.onnx.json")

    if os.path.exists(model_path) and os.path.exists(config_path):
        print(f"  Model already exists: {model_path}")
        return model_path

    print(f"  Downloading voice model: {voice_name}...")
    # Piper voices are hosted on HuggingFace
    # Format: https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/{lang}/{voice_name}/{quality}/{file}
    # Example: id/id_ID/news_tts/medium/id_ID-news_tts-medium.onnx
    parts = voice_name.split("-")  # e.g. id_ID-news_tts-medium
    lang_code = parts[0][:2]  # "id", "en", "zh"
    locale = parts[0]  # "id_ID", "en_US", "zh_CN"
    quality = parts[-1]  # "medium"
    voice = "-".join(parts[1:-1])  # "news_tts", "amy", "huayan"

    base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang_code}/{locale}/{voice}/{quality}"

    import urllib.request

    os.makedirs(data_dir, exist_ok=True)

    for filename in [f"{voice_name}.onnx", f"{voice_name}.onnx.json"]:
        url = f"{base_url}/{filename}"
        dest = os.path.join(data_dir, filename)
        print(f"  Downloading {url}...")
        try:
            urllib.request.urlretrieve(url, dest)
            size_mb = os.path.getsize(dest) / 1024 / 1024
            print(f"  Saved: {dest} ({size_mb:.1f}MB)")
        except Exception as e:
            print(f"  Failed to download {filename}: {e}")
            raise

    return model_path


def test_synthesis(lang: str, text: str, speed: float, output_dir: str, data_dir: str):
    """Test TTS synthesis for a given language."""
    from piper import PiperVoice

    voice_name = VOICE_MAP.get(lang)
    if not voice_name:
        print(f"No voice available for language: {lang}")
        print(f"Supported languages: {list(VOICE_MAP.keys())}")
        return

    print(f"\n{'='*60}")
    print(f"Language: {lang}")
    print(f"Voice: {voice_name}")
    print(f"Text: {text}")
    print(f"Speed: {speed}x")
    print(f"{'='*60}")

    # Step 1: Download model if needed
    model_path = download_voice_model(voice_name, data_dir)

    # Step 2: Load voice
    print(f"\n  Loading voice model...")
    t0 = time.time()
    voice = PiperVoice.load(model_path, use_cuda=False)
    load_time = time.time() - t0
    print(f"  Voice loaded in {load_time:.2f}s")

    # Step 3: Synthesize
    output_file = os.path.join(output_dir, f"hook_{lang}.wav")
    os.makedirs(output_dir, exist_ok=True)

    from piper.config import SynthesisConfig

    print(f"\n  Synthesizing...")
    t0 = time.time()
    with wave.open(output_file, "wb") as wav_file:
        # Piper 1.4+ synthesize returns a generator of AudioChunk
        # We need to use SynthesisConfig for parameters
        syn_config = SynthesisConfig(length_scale=1.0 / speed)
        
        first_chunk = True
        for chunk in voice.synthesize(text, syn_config=syn_config):
            if first_chunk:
                wav_file.setnchannels(chunk.sample_channels)
                wav_file.setsampwidth(chunk.sample_width)
                wav_file.setframerate(chunk.sample_rate)
                first_chunk = False
            wav_file.writeframes(chunk.audio_int16_bytes)
    synth_time = time.time() - t0

    # Step 4: Report results
    file_size = os.path.getsize(output_file) / 1024
    with wave.open(output_file, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        audio_duration = frames / rate

    print(f"\n  Results:")
    print(f"    Output: {output_file}")
    print(f"    File size: {file_size:.1f}KB")
    print(f"    Audio duration: {audio_duration:.2f}s")
    print(f"    Synthesis time: {synth_time:.3f}s")
    print(f"    Real-time factor: {synth_time/audio_duration:.3f}x (lower = faster)")
    print(f"    Sample rate: {rate}Hz")


def main():
    parser = argparse.ArgumentParser(description="Test Piper TTS for hook voiceover")
    parser.add_argument("--lang", default="id", choices=["id", "en", "zh", "all"],
                        help="Language to test (default: id)")
    parser.add_argument("--text", default=None, help="Custom text to synthesize")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed multiplier (default: 1.0)")
    parser.add_argument("--output-dir", default="../test_output/tts", help="Output directory for WAV files")
    parser.add_argument("--data-dir", default="../../model-tts/piper", help="Directory for Piper voice models")
    args = parser.parse_args()

    print("Piper TTS Test")
    print(f"Python: {sys.version}")

    try:
        import piper
        print(f"Piper: installed")
    except ImportError:
        print("Piper not installed! Run: pip install piper-tts")
        sys.exit(1)

    if args.lang == "all":
        # Test all languages
        for lang in VOICE_MAP:
            text = args.text or SAMPLE_HOOKS[lang]
            test_synthesis(lang, text, args.speed, args.output_dir, args.data_dir)
    else:
        text = args.text or SAMPLE_HOOKS[args.lang]
        test_synthesis(args.lang, text, args.speed, args.output_dir, args.data_dir)

    print(f"\n{'='*60}")
    print("Done! Check output files in:", args.output_dir)


if __name__ == "__main__":
    main()
