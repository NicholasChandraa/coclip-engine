from torchaudio.functional import vad
from faster_whisper import WhisperModel

model_size = "deepdml/faster-whisper-large-v3-turbo-ct2"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe(
    "videoplayback (1).mp4",
    beam_size=5,
    vad_filter=True,
)

print(
    "Detected language '%s' with probability %f"
    % (info.language, info.language_probability)
)


def format_timestamp(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


total = ""
print("\n--- Segment Transcripts ---\n")
for segment in segments:
    start_str = format_timestamp(segment.start)
    end_str = format_timestamp(segment.end)
    print(f"[{start_str} -> {end_str}] {segment.text}")
    total += segment.text + " "

print("\n--- Full Transcript ---\n")
print(total)
