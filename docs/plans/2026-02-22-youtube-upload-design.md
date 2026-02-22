# Design: YouTube Upload (Manual Review Mode)

**Date:** 2026-02-22
**Status:** Approved
**Scope:** YouTube upload only, Manual Review mode (user selects clips + edits metadata before upload)

---

## Architecture

**Approach C — Auth-service owns OAuth + token storage, Engine handles upload logic**

```
Frontend (Next.js)
  ├── OAuth flow  →  Auth-service (Go)   → Google OAuth2
  └── Upload      →  Engine (Python)     → YouTube Data API v3
                          │
                          └── internal call → Auth-service /internal/social/token
```

- Auth-service: Single source of truth for all credentials (JWT + OAuth tokens)
- Engine: Upload execution only — fetches token from auth-service via service token
- Frontend: Two API targets for social features (auth-service for OAuth, engine for upload)

---

## OAuth Flow

```
1. Frontend  →  GET /api/v1/social/auth/youtube/start  (auth-service, Bearer JWT)
2.           ←  { url: "https://accounts.google.com/o/oauth2/auth?state=<random>" }
3. Frontend  →  window.location.href = url
4. Google    →  GET /api/v1/social/auth/youtube/callback?code=..&state=..  (auth-service)
5. Auth-svc     validate state (Redis) → exchange code → encrypt tokens → save to DB
6.           →  redirect to https://coclip.site/settings?connected=youtube
```

---

## Upload Flow

```
1. Frontend  →  POST /api/v1/social/upload  (engine, Bearer JWT)
               body: { clip_id, platform, title, description, tags, privacy }
2. Engine       decode JWT → get user_id
3. Engine    →  GET /internal/social/token/{user_id}/youtube  (auth-service, service token)
4. Auth-svc     check expiry → refresh if needed → return access_token
5. Engine       start asyncio background task: resumable upload to YouTube Data API v3
6. Engine    ←  { upload_id, status: "uploading" }
7. Frontend     poll GET /api/v1/social/upload/{upload_id} every 3s
8. Engine       update ClipUpload status → "completed" + platform_video_id
9. Frontend     show YouTube link
```

---

## Database Schema

### Auth-service — `social_accounts` table (GORM)

```go
type SocialAccount struct {
    ID               uuid.UUID  `gorm:"primaryKey"`
    UserID           uuid.UUID  `gorm:"index;not null"`
    Platform         string     // "youtube" | "instagram" | "tiktok"
    AccessToken      string     // AES-GCM encrypted
    RefreshToken     string     // AES-GCM encrypted
    TokenExpiry      time.Time
    Scope            string
    PlatformUserID   string     // YouTube channel ID
    PlatformUsername string     // YouTube channel handle
    CreatedAt        time.Time
    UpdatedAt        time.Time
}
```

### Engine — `clip_uploads` table (SQLAlchemy)

```python
class ClipUpload(Base):
    __tablename__ = "clip_uploads"
    id                = Column(UUID, primary_key=True, default=uuid4)
    clip_id           = Column(String, ForeignKey("clips.clip_id"))
    user_id           = Column(UUID, nullable=False)
    platform          = Column(String, nullable=False)   # "youtube"
    status            = Column(String, default="uploading")  # uploading|completed|failed
    platform_video_id = Column(String, nullable=True)
    platform_url      = Column(String, nullable=True)
    error             = Column(Text, nullable=True)
    title             = Column(String, nullable=False)
    description       = Column(Text, nullable=True)
    tags              = Column(JSON, nullable=True)
    privacy           = Column(String, default="private")  # public|unlisted|private
    created_at        = Column(DateTime, default=datetime.utcnow)
    completed_at      = Column(DateTime, nullable=True)
```

---

## API Endpoints

### Auth-service (Go)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/api/v1/social/auth/youtube/start` | JWT | Generate Google OAuth URL + save state to Redis |
| `GET` | `/api/v1/social/auth/youtube/callback` | Public | Exchange code → save encrypted token → redirect |
| `GET` | `/api/v1/social/accounts` | JWT | List connected accounts |
| `DELETE` | `/api/v1/social/accounts/youtube` | JWT | Disconnect YouTube |
| `GET` | `/internal/social/token/:user_id/youtube` | Service Token | Get valid access token (auto-refresh) |

### Engine (Python)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/api/v1/social/upload` | JWT | Trigger YouTube upload, return upload_id |
| `GET` | `/api/v1/social/upload/:upload_id` | JWT | Poll upload status |
| `GET` | `/api/v1/social/uploads` | JWT | Upload history for current user |

---

## Frontend Changes

### New: `/settings` page
- Fetch connected accounts from auth-service
- Connect button → redirect to OAuth start URL
- Show channel name + Disconnect button when connected
- Handle `?connected=youtube` query param → toast notification

### Modified: ClipDetailModal — add Upload section
- Pre-fill: title from `clip.title`, description from `clip.suggested_caption`, tags from `clip.tags`
- Privacy selector (default: private)
- If YouTube not connected → show "Connect YouTube" link to Settings
- Upload button → POST to engine → poll status every 3s
- If already uploaded → show YouTube link

### New: `src/lib/social-api.ts`
- `startYouTubeOAuth()` — GET auth-service start endpoint
- `getSocialAccounts()` — GET auth-service accounts
- `disconnectAccount(platform)` — DELETE auth-service account
- `uploadClip(params)` — POST engine upload
- `getUploadStatus(uploadId)` — GET engine upload status

---

## Environment Variables

### Auth-service `.env`
```
YOUTUBE_CLIENT_ID=
YOUTUBE_CLIENT_SECRET=
YOUTUBE_REDIRECT_URI=https://auth.coclip.site/api/v1/social/auth/youtube/callback
SOCIAL_TOKEN_ENCRYPTION_KEY=   # 32-byte AES key
FRONTEND_URL=https://coclip.site
```

### Engine `.env`
```
AUTH_SERVICE_URL=http://localhost:8005
AUTH_SERVICE_TOKEN=            # shared service token
```

---

## Implementation Order

1. Auth-service: `social_accounts` model + GORM migration
2. Auth-service: OAuth handler (start + callback) + token encryption util
3. Auth-service: accounts list/delete endpoints
4. Auth-service: internal token endpoint (with auto-refresh)
5. Engine: `clip_uploads` model + migration
6. Engine: YouTube upload utility (resumable upload)
7. Engine: `/social/upload` + `/social/upload/:id` endpoints
8. Frontend: `social-api.ts` helper
9. Frontend: Settings page
10. Frontend: ClipDetailModal upload section
