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
| **Job Queue** | ARQ + Redis | Async background job processing untuk video panjang |
| **Transcription** | WhisperX (Faster-Whisper + Alignment + Diarization) | Speech-to-text dengan word-level timestamps + speaker detection |
| **Content Analysis** | Gemini (LLM) | AI untuk detect viral clips |
| **Video Processing** | FFmpeg | Video cutting & subtitle burning |
| **Video Download** | yt-dlp | Download video dari YouTube |

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
│   Golang (Gin)  │  - Save metadata to DB
│   (Orchestrator)│  - Create job entry
│                 │  - Trigger Python processing
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
│  Redis Queue    │  Job queue storage
│  (ARQ)          │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  ARQ Worker     │  Background processing:
│  (Python)       │  1. Download/Load video
│                 │  2. Transcribe (Whisper)
│                 │  3. Analyze (Gemini)
│                 │  4. Edit (FFmpeg)
│                 │  5. Update job status
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   Database      │  Job status & results
└─────────────────┘
```

---

## 📋 Detailed Flow

### **1. Upload Flow**

```
User (Next.js)
  ↓
  Provides YouTube URL or uploads video file
  ↓
Next.js sends to Golang API
  ↓
Golang:
  - Validates request
  - Saves file metadata to DB
  - Creates job entry (status: "pending")
  - Calls Python FastAPI
  ↓
Returns job_id to user (INSTANT response)
```

### **2. Processing Flow (Async Background)**

#### **Step 1: Job Enqueueing**
```
Python FastAPI receives request
  ↓
Saves uploaded file to temp storage
  ↓
Enqueues job to ARQ (Redis queue)
  ↓
Returns {job_id, status: "queued"} immediately
```

#### **Step 2: Background Processing (ARQ Worker)**

**Full Pipeline Progress Map:**
```
┌─────────────────────────────────────────────────────────┐
│  0%          25%          50%          80%         100%  │
│  ├───────────┼────────────┼────────────┼───────────┤    │
│  │  PHASE 1  │  PHASE 2   │  PHASE 3   │  PHASE 4  │    │
│  │Transcribe │  Analyze   │   Edit     │ Finalize  │    │
│  │(WhisperX) │  (Gemini)  │  (FFmpeg)  │           │    │
│  └───────────┴────────────┴────────────┴───────────┘    │
└─────────────────────────────────────────────────────────┘
```

**A. Download/Preparation (0%)**
```
ARQ Worker picks job from queue
  ↓
Update status: "downloading"
  ↓
If YouTube URL: download using yt-dlp
If uploaded file: use temp file
```

**B. Phase 1: Transcription — The "Ears" (0% → 25%)**
```
Status: "transcribing"

  0%  → ARQ Worker picks job, load audio
  5%  → WhisperX Step 1: Transcription (batched inference)
        Output: Raw transcript segments + detected language
 15%  → WhisperX Step 2: Alignment (wav2vec2 phoneme-based)
        Output: Precise word-level timestamps per segment
 20%  → WhisperX Step 3: Diarization [if ENABLE_DIARIZATION=true]
        Output: Speaker labels per segment + per word (pyannote)
 25%  → Transcription complete, save to Redis

Current implementation: ✅ DONE
```

**C. Phase 2: Content Analysis — The "Brain" (25% → 50%)**
```
Status: "analyzing"

 25%  → Feed full transcript to Gemini LLM
 30%  → Gemini processes transcript
        Prompt: "Based on this transcript, find viral-worthy clips"
 45%  → Gemini returns:
          - Clip timestamps (start/end)
          - Reasoning (why this clip is viral-worthy)
          - Suggested title/keywords
 50%  → Analysis complete, clip candidates identified

Current implementation: ❌ TODO
```

**D. Phase 3: Video Editing (50% → 80%)**
```
Status: "editing"

 50%  → Start video editing pipeline
        For each clip recommended by Gemini:
 55%  →   FFmpeg cuts video based on timestamps
 65%  →   Burns subtitles using WhisperX word-level timing
 75%  →   Encode & save clip to output directory
 80%  → All clips generated

Current implementation: ❌ TODO
```

**E. Phase 4: Finalization (80% → 100%)**
```
Status: "finalizing"

 80%  → Generate clip thumbnails
 85%  → Save clip metadata to DB
 90%  → Notify Golang (webhook/callback)
 95%  → Cleanup temp files (source video, intermediate files)
100%  → Job complete! Clips ready for review

Status: "completed"

Current implementation: ❌ TODO
```

### **3. Review Flow**

```
Golang receives completion notification
  ↓
Updates job status in DB
  ↓
Next.js polls job status (or receives webhook)
  ↓
Displays clips to user:
  - Preview thumbnails
  - Playback
  - Download options
```

---

## 🎬 Handling Long Videos (1-2 Hours)

### **Problem dengan Sync Processing:**
- HTTP timeout (>30 menit transcription)
- Memory issues (loading entire file)
- No progress feedback
- Blocks server resources

### **Solution: ARQ Job Queue**

**Technology:** ARQ (Async Redis Queue)
- **Why ARQ?** Async-native, simple setup, works perfectly with FastAPI
- **Why not Celery?** ARQ lebih simple, native async, cukup untuk use case ini

**Implementation:**
```python
# FastAPI Endpoint
@app.post("/transcribe-async")
async def transcribe_async(file: UploadFile):
    job_id = generate_id()
    save_file_streaming(file)  # Chunked upload
    await redis.enqueue_job('transcribe_task', job_id, file_path)
    return {"job_id": job_id, "status": "queued"}

# Status Polling
@app.get("/job/{job_id}")
async def get_job_status(job_id):
    status = await redis.get(f"job:{job_id}:status")
    progress = await redis.get(f"job:{job_id}:progress")
    return {"status": status, "progress": progress}
```

**Benefits:**
- ✅ Upload returns instantly (<1 second)
- ✅ Processing happens in background
- ✅ User can check progress via polling
- ✅ No HTTP timeout issues
- ✅ Can handle multiple jobs concurrently
- ✅ Scalable (add more workers)

---

## 📁 Component Structure

```
/coclip
├── /frontend
│   └── Next.js 14+ (App Router)
│       - File upload UI
│       - Job status polling
│       - Clip preview/download
│
├── /backend
│   └── Golang (Gin framework)
│       - API Gateway
│       - Auth & user management
│       - File metadata storage
│       - Job orchestration
│       - Database operations
│
└── /engine
    └── Python (FastAPI + ARQ)
        ├── /app
        │   ├── /api/routes
        │   │   └── transcribe.py (async endpoints)
        │   ├── /schemas
        │   │   └── transcription.py (Pydantic models)
        │   ├── /tools
        │   │   ├── transcriber.py (WhisperX)
        │   │   ├── analyzer.py (Gemini)
        │   │   └── editor.py (FFmpeg)
        │   └── /workers
        │       └── transcription_worker.py (ARQ tasks)
        ├── pyproject.toml (Poetry)
        └── requirements.txt
```

---

## 🔄 Job Status States

```
pending → queued → downloading → transcribing → analyzing → editing → finalizing → completed
                                                                                 ↓
                                                                               failed
```

**Status Definitions:**
- `pending`: Job created, waiting to be picked
- `queued`: In Redis queue, waiting for worker
- `downloading`: Downloading video (YouTube only)
- `transcribing`: Running WhisperX (transcribe + align + diarize) — 0-25%
- `analyzing`: Gemini analyzing transcript for viral clips — 25-50%
- `editing`: FFmpeg cutting clips + burning subtitles — 50-80%
- `finalizing`: Saving metadata, generating thumbnails, cleanup — 80-100%
- `completed`: All clips generated successfully
- `failed`: Error occurred (with error message)

---

## 🚀 Deployment Considerations

### **Development:**
- FastAPI: `uvicorn app.main:app --reload`
- ARQ Worker: `arq app.workers.transcription_worker.WorkerSettings`
- Redis: `redis-server` (local)

### **Production:**
- Multiple ARQ workers untuk concurrent processing
- Redis dengan persistence enabled
- Monitoring untuk job queue depth
- Auto-cleanup old jobs (TTL di Redis)

---

## ✅ Implementation Status

- [x] WhisperX transcription (Faster-Whisper + Alignment + Diarization)
- [x] FastAPI async endpoints (transcribe-async, status polling, full result)
- [x] ARQ job queue integration (Redis-based background processing)
- [x] Progress tracking per-step (0% → 10% → 40% → 70% → 90% → 100%)
- [x] Pydantic schemas separation (app/schemas/)
- [x] Diarization configurable via .env (ENABLE_DIARIZATION)
- [x] Streaming file upload (chunked 8KB)
- [x] Redis connection health check (keepalive, retry on timeout)
- [ ] Gemini content analysis
- [ ] FFmpeg video editing
- [ ] Subtitle burning
- [ ] Golang backend
- [ ] Next.js frontend
- [ ] Database schema
- [ ] Authentication
