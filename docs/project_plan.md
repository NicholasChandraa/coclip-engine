# CoClip - AI-Powered Auto Video Clipper
## Project Plan & Technical Specification

---

## 📌 Project Overview

### Tujuan Aplikasi
Membangun aplikasi web otomatis untuk mengubah video panjang (YouTube/Upload) menjadi klip-klip pendek yang viral-worthy dengan subtitle otomatis.

## 🏗️ Tech Stack

### Frontend (User Interface)
- **Framework**: Next.js 14+ (App Router)
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Realtime**: Server-Sent Events (SSE)

### Backend (Orchestrator)
- **Language**: Golang 1.21+
- **Framework**: Gin
- **Database**: PostgreSQL
- **Queue**: Redis (Asynq)
- **Storage**: Local Filesystem (Dev) / MinIO (Prod)

### AI Engine (Video Processing)
- **Language**: Python 3.11+
- **Framework**: FastAPI + LangChain/LangGraph
- **STT Model**: **Faster-Whisper** (Use Model: `deepdml/faster-whisper-large-v3-turbo-ct2`) 🚀
  - Backend: CTranslate2 (Optimized Inference Engine)
  - Speed: **Turbo** variant is significantly faster with comparable accuracy to large-v3.
  - VRAM: Lebih hemat memory.
- **LLM**: Google Gemini 1.5 Flash (via LangChain)
- **Video Tools**: 
  - **FFmpeg** (via `ffmpeg-python`) 🛡️
    - *Why FFmpeg?*: "Old" = "Battle Tested". FFmpeg adalah backbone industri video (YouTube, Netflix, Disney+ use it).
    - Tidak ada tools lain yang punya fitur selengkap FFmpeg untuk programmatic editing (subtitle burning, exact frame cutting, re-encoding). 
    - Kita pakai wrapper Python supaya syntax-nya modern & clean, tapi engine-nya tetap FFmpeg.

---

## 🔄 Architecture Flow (LangGraph)

```
[START]
  ↓
[Download Video] (yt-dlp)
  ↓
[Extract Audio] (ffmpeg)
  ↓
[Transcribe] (Faster-Whisper Tool)
  ↓ ✨ Output: Transcript + Accurate Word Timestamps
  ↓
[Analyze] (Gemini + LangChain)
  ↓ ✨ Output: List of Viral Clips (Start-End times)
  ↓
[Cut & Burn] (ffmpeg-python)
  ↓
[Upload Results]
  ↓
[END]
```

## 🛠️ Implementation Steps

### Step 1: Setup Environment
- Frontend: `next.js`
- Backend: `golang`
- Engine: `uv` (modern python package manager)

### Step 2: Python Engine (The Core)
```python
# tools/transcriber.py
from faster_whisper import WhisperModel

# Load model once (Global/Singleton)
# using "deepdml/faster-whisper-large-v3-turbo-ct2"
model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2", device="cuda", compute_type="float16")

def transcribe(audio_path):
    segments, info = model.transcribe(
        audio_path, 
        beam_size=5, 
        word_timestamps=True
    )
    return list(segments) 
```

### Step 3: FFmpeg Integration
```python
import ffmpeg

def cut_and_caption(video_path, start, end, srt_path, output_path):
    (
        ffmpeg
        .input(video_path, ss=start, to=end)
        .filter('subtitles', srt_path)
        .output(output_path, c='copy')
        .run()
    )
```

---

## 💰 Resource Requirements
- **Disk**: Local Storage (Free)
- **GPU**: Recommended NVIDIA GPU (min 4GB VRAM) untuk Faster-Whisper "large-v3".
  - *Fallback*: CPU (jalan tapi lambat).
