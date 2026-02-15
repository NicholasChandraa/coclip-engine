
import torchaudio
import torchaudio.backend
import sys
import os

print(f"Torchaudio version: {torchaudio.__version__}")
print(f"Backend available: {torchaudio.list_audio_backends()}")

try:
    print("Trying soundfile backend...")
    torchaudio.set_audio_backend("soundfile")
    print("Set backend: soundfile")
except Exception as e:
    print(f"Error setting soundfile backend: {e}")

# This might trigger torio FFmpeg load if default is used
try:
    print("Checking default backend behavior...")
    # Just accessing list might trigger init
    backends = torchaudio.list_audio_backends()
    print(f"Backends: {backends}")
except Exception as e:
    print(f"Error listing backends: {e}")
