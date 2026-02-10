# Implementation Plan - Coclip (AI Auto Clipper)

## 🎯 Goal Description

Build an automated video clipper application that takes YouTube links or video files, processes them to identify interesting clips, generates subtitles, and allows users to view/download them.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------| 
| **Frontend** | Next.js 14+ (App Router) | User Interface |
| **Backend (Orchestrator)** | Golang (Gin) | API Gateway, Auth, File Management, Job Queueing |
| **AI Engine** | Python (FastAPI) | Video Processing, Transcription, Content Analysis, Video Editing |
| **Pipeline** | LangGraph | DAG-based pipeline orchestration with state management |
| **Job Queue** | ARQ + Redis | Async background job processing untuk video panjang |
| **Transcription** | WhisperX (Faster-Whisper + Alignment + Diarization) | Speech-to-text dengan word-level timestamps + speaker detection |
| **Content Analysis** | Gemini (LangChain) | AI untuk detect viral clips |
| **Video Processing** | FFmpeg | Video cutting, subtitle burning, portrait cropping |
| **Video Download** | yt-dlp | Download video dari YouTube |
| **Database** | SQLite (dev) / PostgreSQL (prod) | Persistent job & clip storage |

---

## 🏗️ Architecture Overview

```
┌─────────────────┐
│   Next.js       │  User uploads video / YouTube URL
│   (Frontend)    │
└────────┬────────┘
         │ HTTP Request
         ↓
┌─────────────────┐
│   Golang (Gin)  │  - Auth & user management
│   (Orchestrator)│  - Proxy to Engine API
│                 │  - File management
└────────┬────────┘
         │ HTTP Call
         ↓
┌─────────────────┐
│  Python FastAPI │  - Receive job request
│  (AI Engine)    │  - Enqueue to ARQ worker
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
┌─────────────────────────────────────────┐
│  LangGraph Pipeline (ARQ Worker)        │
│                                         │
│  transcription → analysis → editing → finalization
│  (WhisperX)     (Gemini)   (FFmpeg)   (DB save)
└────────┬────────────────────────────────┘
         │
         ↓
┌─────────────────┐
│   SQLite DB     │  Persistent job & clip metadata
└─────────────────┘
```

---

## 📋 LangGraph Pipeline Flow

### **Phase 1: Transcription — "Ears" (0% → 25%)**
```
WhisperX Pipeline:
  0%  → Load audio from video
  5%  → Transcription (batched inference, language detection)
 15%  → Alignment (wav2vec2 phoneme-based word timestamps)
 20%  → Diarization (speaker labels via pyannote)
 25%  → Complete: TranscriptionResultDetailed saved to state
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
FFmpeg Processing (per clip):
 50%  → Create output directory clips/{job_id}/
        For each clip candidate:
        1. Generate ASS subtitle (word-level karaoke timing)
        2. Detect video orientation (landscape/portrait)
        3. FFmpeg: cut + crop to 9:16 + burn subtitles
 78%  → All clips generated with subtitles
 80%  → Route to finalization
```

### **Phase 4: Finalization (80% → 100%)**
```
 80%  → Save clip metadata to SQLite DB
 85%  → Generate thumbnails (TODO)
 90%  → Cleanup temp files
 95%  → Mark job completed in Redis + DB
100%  → Done! Clips ready for download
```

---

## 📁 Engine Structure

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
│   │   ├── config.py                # App settings (Whisper, Gemini, Redis, paths)
│   │   └── database.py              # SQLAlchemy engine & session (TODO)
│   ├── /models
│   │   └── models.py                # ORM models: Job, Clip (TODO)
│   ├── /schemas
│   │   ├── transcription.py         # Pydantic models (segments, words, results)
│   │   └── graph_schemas.py         # LangGraph state TypedDict
│   ├── /graphs
│   │   ├── video_processing_graph.py # LangGraph DAG definition
│   │   └── /nodes
│   │       ├── transcription_node.py # WhisperX transcription
│   │       ├── analysis_node.py      # Gemini viral clip detection
│   │       ├── editing_node.py       # FFmpeg cutting + subtitle burning
│   │       └── finalization_node.py  # Result saving + cleanup
│   ├── /tools
│   │   └── transcriber.py           # WhisperX model loader
│   ├── /utils
│   │   ├── logging.py               # Rich console + file logger
│   │   ├── progress_tracker.py      # Redis progress updates
│   │   └── subtitle_generator.py    # ASS subtitle generation (word-level)
│   └── /workers
│       └── transcription_worker.py  # ARQ worker (runs LangGraph pipeline)
│
├── /clips                           # Generated clip output (per job_id)
├── /temp                            # Temporary uploaded videos
└── /logs                            # Application logs (coclip.log)
```

---

## 🔄 Job Status States

```
queued → transcribing → analyzing → editing → finalizing → completed
                                                            ↓
                                                          failed
```

---

## 🚀 Running the Engine

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

## ✅ Implementation Status

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
- [ ] Portrait crop (9:16 for TikTok/Reels)
- [ ] SQLite database (persistent job & clip storage)
- [ ] Thumbnail generation
- [ ] Golang backend (API gateway, auth)
- [ ] Next.js frontend (upload UI, clip preview, download)
- [ ] Authentication
