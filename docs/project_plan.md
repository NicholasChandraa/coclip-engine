# Implementation Plan - Coclip (AI Auto Clipper)

## Goal Description

Build an automated video clipper application that takes YouTube links or video files, processes them to identify interesting clips, generates subtitles, and allows users to view/download them.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Next.js 14+ (App Router) | User Interface |
| **Backend** | Python (FastAPI) | API server, Auth, File Management, Job Queueing |
| **Pipeline** | LangGraph | DAG-based pipeline orchestration with state management |
| **Job Queue** | ARQ + Redis | Async background job processing untuk video panjang |
| **Transcription** | WhisperX (Faster-Whisper + Alignment + Diarization) | Speech-to-text dengan word-level timestamps + speaker detection |
| **Content Analysis** | Gemini (LangChain) | AI untuk detect viral clips |
| **Smart Crop** | S3FD Face Detection + Keyframe Tracking | Dynamic face-following crop with smooth transitions |
| **Video Processing** | FFmpeg (h264_nvenc GPU encode) | Video cutting, subtitle burning, smart portrait cropping |
| **Video Download** | yt-dlp | Download video dari YouTube |
| **Hook Generation** | Gemini (LangChain) — 2nd LLM call | Generate hook text + caption per clip |
| **TTS** | Piper TTS (multilingual, ONNX, offline) | Hook voiceover — `id_ID`, `en_US`, `zh_CN` (Female voices, zero VRAM) |
| **Database** | PostgreSQL | Persistent job & clip storage |

---

## Architecture Overview

```
┌─────────────────┐
│   Next.js       │  User uploads video / pastes YouTube URL
│   (Frontend)    │
└────────┬────────┘
         │ HTTP Request (direct)
         ↓
┌─────────────────┐
│  Python FastAPI │  - API server + Auth
│  (Backend)      │  - File upload OR YouTube URL
│                 │  - yt-dlp download (if URL)
│                 │  - Enqueue to ARQ worker
│                 │  - Return job_id instantly
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Redis Queue    │  Job queue + temporary progress storage
│  (ARQ)          │
└────────┬────────┘
         │
         ↓
┌──────────────────────────────────────────────────┐
│  LangGraph Pipeline (ARQ Worker)                  │
│                                                   │
│  transcription → analysis → editing → finalization│
│  (WhisperX)     (Gemini)   (FFmpeg)  (DB+cleanup) │
│                             +Gemini2              │
│                             +TTS)                 │
│                                                   │
│  Smart crop: S3FD face detect → keyframe tracking │
│  GPU Memory: WhisperX → unload → S3FD → unload    │
└────────┬─────────────────────────────────────────┘
         │
         ↓
┌─────────────────┐
│  PostgreSQL DB  │  Persistent job & clip metadata
└─────────────────┘
```

### Input Modes

| Mode | Flow |
|------|------|
| **File Upload** | User uploads video file → save to temp/ → enqueue pipeline |
| **YouTube URL** | User pastes URL → yt-dlp downloads to temp/ → enqueue pipeline |

---

## LangGraph Pipeline Flow

### **Phase 1: Transcription — "Ears" (0% → 25%)**
```
WhisperX Pipeline:
  0%  → Load audio from video
  5%  → Transcription (batched inference, language detection)
 15%  → Alignment (wav2vec2 phoneme-based word timestamps)
 20%  → Diarization (speaker labels via pyannote)
 25%  → Complete: TranscriptionResultDetailed saved to state
         → Unload WhisperX from GPU
```

### **Phase 2: Content Analysis — "Brain" (25% → 50%)**
```
Gemini LLM Analysis:
 25%  → Build transcript text from segments
 30%  → Send to Gemini via LangChain with viral detection prompt
 45%  → Parse response: clip candidates with timestamps,
         titles, viral scores, reasoning
 50%  → Complete: clip_candidates routed to editing
```

### **Phase 3: Video Editing — "Hands" (50% → 80%)**
```
Smart Crop + FFmpeg Processing:
 50%  → Load S3FD face detector
 52%  → For each clip candidate:
         1. Sample 15 frames across clip segment
         2. Detect faces per frame (S3FD or OpenCV fallback)
         3. Cluster faces → speaker positions
         4. Pick most prominent speaker (largest face + most visible)
         5. Calculate smart crop position (center on speaker)
 55%  → Unload face detector
 58%  → For each clip candidate (parallel, Semaphore=3):
         1. Generate ASS subtitle (word-level karaoke timing)
         2. FFmpeg: cut + smart crop to 9:16 + burn subtitles
 70%  → Hook generation (2nd Gemini call):
         1. Generate hook text per clip (attention-grabbing opening line)
         2. Generate social media caption per clip
 75%  → TTS voiceover for hooks:
         1. Piper TTS: convert hook text → audio (auto-select voice by WhisperX language)
            - id → id_ID-news_tts-medium (Female)
            - en → en_US-amy-medium (Female)
            - zh → zh_CN-huayan-medium (Female)
            - other → skip TTS, text overlay only
         2. FFmpeg: overlay hook audio + text at clip start (3-5s)
 78%  → All clips generated with subtitles + hooks
 80%  → Route to finalization
```

### **Phase 4: Finalization (80% → 100%)**
```
 80%  → Cache results to Redis (fast access)
 85%  → Save job + clips to PostgreSQL (persistent)
 90%  → Generate thumbnails (TODO)
 95%  → Cleanup temp files, mark job completed
100%  → Done! Clips ready for download
```

---

## Engine Structure

```
/coclip-engine
├── main.py                          # FastAPI app entry point
├── pyproject.toml                   # Dependencies
├── .env                             # Environment config
│
├── /app
│   ├── /api/routes
│   │   └── transcribe.py            # API endpoints (upload, status, result, download)
│   ├── /core
│   │   ├── config.py                # App settings (Whisper, Gemini, Redis, DB, SmartCrop)
│   │   └── database.py              # Async SQLAlchemy engine & session
│   ├── /models
│   │   └── __init__.py              # ORM models (Job, Clip)
│   ├── /schemas
│   │   ├── transcription.py         # Pydantic models (segments, words, results)
│   │   └── graph_schemas.py         # LangGraph state TypedDict
│   ├── /graphs
│   │   ├── video_processing_graph.py # LangGraph DAG definition
│   │   └── /nodes
│   │       ├── transcription_node.py # WhisperX transcription
│   │       ├── analysis_node.py      # Gemini viral clip detection
│   │       ├── editing_node.py       # FFmpeg cutting + smart crop + subtitle
│   │       └── finalization_node.py  # DB save + cleanup
│   ├── /tools
│   │   └── transcriber.py           # WhisperX model loader
│   ├── /utils
│   │   ├── logging.py               # Rich console + file logger
│   │   ├── progress_tracker.py      # Redis progress updates
│   │   ├── subtitle_generator.py    # ASS subtitle generation (word-level)
│   │   ├── video_formats.py         # Video format presets (TikTok, Reels, etc)
│   │   ├── speaker_detector.py      # Face tracking smart crop
│   │   └── downloader.py            # yt-dlp YouTube video downloader
│   └── /workers
│       └── transcription_worker.py  # ARQ worker (runs LangGraph pipeline)
│
├── /models
│   └── /loconet_repo                # S3FD face detector weights
│       └── /model/faceDetector/s3fd/sfd_face.pth
├── /clips                           # Generated clip output (per job_id)
├── /temp                            # Temporary uploaded videos
└── /logs                            # Application logs (coclip.log)
```

---

## Job Status States

```
queued → downloading (if URL) → transcribing → analyzing → editing → finalizing → completed
                                                                                     ↓
                                                                                  failed
```

---

## Running the Engine

### Development
```bash
# Terminal 1: FastAPI server
python main.py

# Terminal 2: ARQ Worker
arq app.workers.transcription_worker.WorkerSettings

# Redis must be running
redis-server
```

---

## Smart Crop Strategy

For podcast/interview videos with camera cuts (close-up ↔ wide shot):

1. **Face Detection**: S3FD neural face detector (GPU) with OpenCV Haar cascade fallback
2. **Per-frame Sampling**: Sample frames every SAMPLE_INTERVAL seconds (default 0.5s)
3. **Largest Face**: Pick the largest face per frame → calculate crop_x centered on face
4. **CROP_STRENGTH Damping**: Blend between center crop and face crop (`center + (face - center) * strength`)
5. **Exponential Smoothing**: Smooth crop_x across frames to reduce jitter
6. **Camera Cut Detection**: If crop_x jumps > 50px between frames, snap immediately (no smoothing)
7. **FFmpeg Expression**: Render as animated crop using nested `if(gte(t,...))` expressions with lerp transitions

Tunable constants (top of `speaker_detector.py`):
- `CROP_STRENGTH` (0.8) — 0=center only, 1=full face tracking
- `SAMPLE_INTERVAL` (0.5) — seconds between face detection samples
- `SMOOTHING` (0.3) — exponential smoothing alpha (lower=smoother)
- `TRANSITION_DURATION` (0.3) — FFmpeg lerp duration between positions

---

## Implementation Status

- [x] WhisperX transcription (Faster-Whisper + Alignment + Diarization)
- [x] FastAPI async endpoints (upload, status polling, result, clip download)
- [x] ARQ job queue integration (Redis-based background processing)
- [x] LangGraph pipeline orchestration (state management, node routing)
- [x] Gemini content analysis (viral clip detection via LangChain)
- [x] FFmpeg video cutting (async subprocess, per-clip progress)
- [x] Subtitle burning (ASS format, word-level karaoke timing from WhisperX)
- [x] Progress tracking per-phase (0% → 25% → 50% → 80% → 100%)
- [x] File logging (logs/coclip.log mirrors console output)
- [x] Streaming file upload (chunked 8KB)
- [x] Redis connection health check (keepalive, retry on timeout)
- [x] PostgreSQL database (async SQLAlchemy, Job + Clip models, persistent storage)
- [x] DB API endpoints (GET /jobs, GET /jobs/{job_id} for history)
- [x] Video format customization (TikTok 9:16, Reels, Shorts, Square, Landscape)
- [x] Auto-crop for aspect ratio conversion (landscape → portrait)
- [x] Enhanced subtitles (Arial Black 85px, 170px margin, Smart Layout 110% scale)
- [x] **Smart crop (S3FD face tracking + keyframe dynamic crop)**
- [x] **YouTube URL download (yt-dlp integration di API + worker)**
- [x] **Parallel clip cutting (asyncio.gather + Semaphore=3)**
- [x] **GPU encoding (h264_nvenc)**
- [x] **Per-step VRAM management (unload model setelah tiap step)**
- [x] **Speaker-aware smart crop (diarization-based face selection, dual-mapping + jitter for podcast)**
- [x] **Job abort endpoint (cancel download + prevent ARQ retry)**
- [ ] Hook generation (2nd Gemini call — hook text + caption per clip)
- [ ] TTS voiceover (Piper TTS multilingual — id/en/zh, auto-select by WhisperX language)
- [ ] Thumbnail generation
- [ ] Next.js frontend (upload UI, clip preview, download)
- [ ] Authentication (langsung di FastAPI, JWT/session)

---

## References & Resources

- **S3FD Face Detector**: Used for accurate face detection in video frames
- **Internal Module**: [`app/utils/speaker_detector.py`](../app/utils/speaker_detector.py) - Face tracking smart crop
- **Pretrained Weights**: S3FD Face Detector at `models/loconet_repo/model/faceDetector/s3fd/sfd_face.pth`


## Information
yt-dlp butuh JavaScript runtime untuk extract format YouTube yang lengkap. Tanpa itu, beberapa format mungkin gak tersedia (tapi download   tetap jalan).

Solusinya install Deno (yang di-recommend yt-dlp): 
# Windows (PowerShell)
irm https://deno.land/install.ps1 | iex

# Tambah install ini juga
yt-dlp --remote-components ejs:github "https://www.youtube.com/watch?v=vh5VbvP0dPM" --skip-download