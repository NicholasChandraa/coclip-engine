# YouTube Upload (Manual Review) — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let users connect a YouTube account via OAuth and upload individual clips from ClipDetailModal with editable metadata.

**Architecture:** Auth-service (Go) owns OAuth flow + token storage (AES-encrypted in `social_accounts` table). Engine (Python) calls auth-service internal endpoint to get a valid access token, then uploads to YouTube Data API v3 via resumable upload. Frontend uses existing `authFetch`/`engineFetch` wrappers.

**Tech Stack:** Go (Gin + GORM), Python (FastAPI + httpx), Next.js 14, YouTube Data API v3, AES-256-GCM

---

## Prerequisites

```bash
# Auth-service — add oauth2 package
cd auth-service
go get golang.org/x/oauth2

# Engine — add httpx
cd engine
uv add httpx

# Generate a 32-byte hex key for token encryption (run once, save to .env)
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Phase 1: Auth-service (Go)

### Task 1: Add YouTube OAuth Config

**Files:**
- Modify: `auth-service/internal/config/config.go`

**Step 1: Add config structs and fields**

In `config.go`, after the existing `ServiceConfig` struct, add:

```go
// YouTubeConfig holds Google OAuth2 credentials for YouTube
type YouTubeConfig struct {
    ClientID     string
    ClientSecret string
    RedirectURI  string
}

// SocialConfig holds all social media OAuth config
type SocialConfig struct {
    YouTube         YouTubeConfig
    EncryptionKey   string // 64-char hex = 32 bytes for AES-256
    FrontendURL     string
}
```

Add `Social SocialConfig` field to the main `Config` struct (after `Service ServiceConfig`).

In the `Load()` function, add to the return value:

```go
Social: SocialConfig{
    YouTube: YouTubeConfig{
        ClientID:     getEnv("YOUTUBE_CLIENT_ID", ""),
        ClientSecret: getEnv("YOUTUBE_CLIENT_SECRET", ""),
        RedirectURI:  getEnv("YOUTUBE_REDIRECT_URI", "http://localhost:8005/api/v1/social/auth/youtube/callback"),
    },
    EncryptionKey: getEnv("SOCIAL_TOKEN_ENCRYPTION_KEY", ""),
    FrontendURL:   getEnv("FRONTEND_URL", "http://localhost:3000"),
},
```

**Step 2: Add env vars to auth-service/.env**

```env
YOUTUBE_CLIENT_ID=your_google_client_id
YOUTUBE_CLIENT_SECRET=your_google_client_secret
YOUTUBE_REDIRECT_URI=http://localhost:8005/api/v1/social/auth/youtube/callback
SOCIAL_TOKEN_ENCRYPTION_KEY=<64-char hex from prerequisites>
FRONTEND_URL=http://localhost:3000
```

**Step 3: Build to verify no errors**

```bash
cd auth-service && go build ./...
```
Expected: no errors

**Step 4: Commit**

```bash
git add auth-service/internal/config/config.go auth-service/.env
git commit -m "feat(auth): add YouTube OAuth and social token encryption config"
```

---

### Task 2: AES-256-GCM Crypto Utility

**Files:**
- Create: `auth-service/pkg/crypto/aes.go`
- Create: `auth-service/pkg/crypto/aes_test.go`

**Step 1: Write failing test**

```go
// auth-service/pkg/crypto/aes_test.go
package crypto_test

import (
    "testing"
    "auth-service/pkg/crypto"
)

func TestEncryptDecryptRoundtrip(t *testing.T) {
    // 32 bytes = 64 hex chars
    key := "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20"
    plaintext := "ya29.A0AXeO80T_example_access_token"

    encrypted, err := crypto.EncryptAES256GCM(key, plaintext)
    if err != nil {
        t.Fatalf("encrypt failed: %v", err)
    }
    if encrypted == plaintext {
        t.Fatal("encrypted text should not equal plaintext")
    }

    decrypted, err := crypto.DecryptAES256GCM(key, encrypted)
    if err != nil {
        t.Fatalf("decrypt failed: %v", err)
    }
    if decrypted != plaintext {
        t.Fatalf("expected %q, got %q", plaintext, decrypted)
    }
}

func TestEncryptDifferentNonceEachTime(t *testing.T) {
    key := "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20"
    plaintext := "same_token"
    enc1, _ := crypto.EncryptAES256GCM(key, plaintext)
    enc2, _ := crypto.EncryptAES256GCM(key, plaintext)
    if enc1 == enc2 {
        t.Fatal("two encryptions of same text should differ (different nonces)")
    }
}

func TestInvalidKey(t *testing.T) {
    _, err := crypto.EncryptAES256GCM("tooshort", "plaintext")
    if err == nil {
        t.Fatal("expected error for invalid key")
    }
}
```

**Step 2: Run test to verify it fails**

```bash
cd auth-service && go test ./pkg/crypto/... -v
```
Expected: FAIL — `crypto` package not found

**Step 3: Implement**

```go
// auth-service/pkg/crypto/aes.go
package crypto

import (
    "crypto/aes"
    "crypto/cipher"
    "crypto/rand"
    "encoding/base64"
    "encoding/hex"
    "errors"
    "io"
)

// EncryptAES256GCM encrypts plaintext with AES-256-GCM.
// hexKey must be 64 hex chars (32 bytes). Returns base64(nonce+ciphertext).
func EncryptAES256GCM(hexKey, plaintext string) (string, error) {
    key, err := hex.DecodeString(hexKey)
    if err != nil || len(key) != 32 {
        return "", errors.New("invalid key: must be 64-char hex (32 bytes)")
    }
    block, err := aes.NewCipher(key)
    if err != nil {
        return "", err
    }
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return "", err
    }
    nonce := make([]byte, gcm.NonceSize())
    if _, err = io.ReadFull(rand.Reader, nonce); err != nil {
        return "", err
    }
    ciphertext := gcm.Seal(nonce, nonce, []byte(plaintext), nil)
    return base64.StdEncoding.EncodeToString(ciphertext), nil
}

// DecryptAES256GCM decrypts a base64(nonce+ciphertext) produced by EncryptAES256GCM.
func DecryptAES256GCM(hexKey, encoded string) (string, error) {
    key, err := hex.DecodeString(hexKey)
    if err != nil || len(key) != 32 {
        return "", errors.New("invalid key: must be 64-char hex (32 bytes)")
    }
    data, err := base64.StdEncoding.DecodeString(encoded)
    if err != nil {
        return "", err
    }
    block, err := aes.NewCipher(key)
    if err != nil {
        return "", err
    }
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return "", err
    }
    if len(data) < gcm.NonceSize() {
        return "", errors.New("ciphertext too short")
    }
    nonce, ciphertext := data[:gcm.NonceSize()], data[gcm.NonceSize():]
    plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
    if err != nil {
        return "", err
    }
    return string(plaintext), nil
}
```

**Step 4: Run test to verify it passes**

```bash
cd auth-service && go test ./pkg/crypto/... -v
```
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add auth-service/pkg/crypto/
git commit -m "feat(auth): add AES-256-GCM token encryption utility"
```

---

### Task 3: SocialAccount Domain Model

**Files:**
- Create: `auth-service/internal/domain/social.go`

**Step 1: Create the file**

```go
// auth-service/internal/domain/social.go
package domain

import (
    "time"
    "github.com/google/uuid"
)

// SocialAccount stores encrypted OAuth tokens for connected social platforms.
// One row per user per platform (unique index on user_id + platform).
type SocialAccount struct {
    ID               uuid.UUID `gorm:"type:uuid;primaryKey;default:gen_random_uuid()" json:"id"`
    UserID           uuid.UUID `gorm:"type:uuid;not null;uniqueIndex:idx_social_user_platform" json:"user_id"`
    Platform         string    `gorm:"type:varchar(50);not null;uniqueIndex:idx_social_user_platform" json:"platform"`
    AccessToken      string    `gorm:"type:text;not null" json:"-"` // AES-256-GCM encrypted
    RefreshToken     string    `gorm:"type:text;not null" json:"-"` // AES-256-GCM encrypted
    TokenExpiry      time.Time `gorm:"not null" json:"token_expiry"`
    Scope            string    `gorm:"type:text" json:"scope"`
    PlatformUserID   string    `gorm:"type:varchar(255)" json:"platform_user_id"`   // YouTube channel ID
    PlatformUsername string    `gorm:"type:varchar(255)" json:"platform_username"` // YouTube channel title
    CreatedAt        time.Time `gorm:"autoCreateTime" json:"created_at"`
    UpdatedAt        time.Time `gorm:"autoUpdateTime" json:"updated_at"`
}

func (SocialAccount) TableName() string { return "social_accounts" }

// --- DTOs ---

type OAuthStartResponse struct {
    URL string `json:"url"`
}

type SocialAccountResponse struct {
    ID               string    `json:"id"`
    Platform         string    `json:"platform"`
    PlatformUserID   string    `json:"platform_user_id"`
    PlatformUsername string    `json:"platform_username"`
    ConnectedAt      time.Time `json:"connected_at"`
}

type InternalTokenResponse struct {
    AccessToken string    `json:"access_token"`
    TokenExpiry time.Time `json:"token_expiry"`
}
```

**Step 2: Build check**

```bash
cd auth-service && go build ./...
```
Expected: no errors

**Step 3: Commit**

```bash
git add auth-service/internal/domain/social.go
git commit -m "feat(auth): add SocialAccount domain model and DTOs"
```

---

### Task 4: SocialAccount Repository

**Files:**
- Create: `auth-service/internal/repository/social_repository.go`

**Step 1: Create the file**

```go
// auth-service/internal/repository/social_repository.go
package repository

import (
    "context"
    "time"

    "auth-service/internal/domain"
    "github.com/google/uuid"
    "gorm.io/gorm"
    "gorm.io/gorm/clause"
)

type SocialAccountRepository interface {
    Upsert(ctx context.Context, account *domain.SocialAccount) error
    FindByUserAndPlatform(ctx context.Context, userID uuid.UUID, platform string) (*domain.SocialAccount, error)
    FindAllByUser(ctx context.Context, userID uuid.UUID) ([]domain.SocialAccount, error)
    DeleteByUserAndPlatform(ctx context.Context, userID uuid.UUID, platform string) error
    UpdateTokens(ctx context.Context, id uuid.UUID, encAccess, encRefresh string, expiry time.Time) error
}

type socialAccountRepository struct {
    db *gorm.DB
}

func NewSocialAccountRepository(db *gorm.DB) SocialAccountRepository {
    return &socialAccountRepository{db: db}
}

func (r *socialAccountRepository) Upsert(ctx context.Context, account *domain.SocialAccount) error {
    return r.db.WithContext(ctx).
        Clauses(clause.OnConflict{
            Columns: []clause.Column{{Name: "user_id"}, {Name: "platform"}},
            DoUpdates: clause.AssignmentColumns([]string{
                "access_token", "refresh_token", "token_expiry",
                "scope", "platform_user_id", "platform_username", "updated_at",
            }),
        }).
        Create(account).Error
}

func (r *socialAccountRepository) FindByUserAndPlatform(ctx context.Context, userID uuid.UUID, platform string) (*domain.SocialAccount, error) {
    var account domain.SocialAccount
    err := r.db.WithContext(ctx).
        Where("user_id = ? AND platform = ?", userID, platform).
        First(&account).Error
    if err != nil {
        return nil, err
    }
    return &account, nil
}

func (r *socialAccountRepository) FindAllByUser(ctx context.Context, userID uuid.UUID) ([]domain.SocialAccount, error) {
    var accounts []domain.SocialAccount
    err := r.db.WithContext(ctx).
        Where("user_id = ?", userID).
        Find(&accounts).Error
    return accounts, err
}

func (r *socialAccountRepository) DeleteByUserAndPlatform(ctx context.Context, userID uuid.UUID, platform string) error {
    return r.db.WithContext(ctx).
        Where("user_id = ? AND platform = ?", userID, platform).
        Delete(&domain.SocialAccount{}).Error
}

func (r *socialAccountRepository) UpdateTokens(ctx context.Context, id uuid.UUID, encAccess, encRefresh string, expiry time.Time) error {
    return r.db.WithContext(ctx).
        Model(&domain.SocialAccount{}).
        Where("id = ?", id).
        Updates(map[string]any{
            "access_token":  encAccess,
            "refresh_token": encRefresh,
            "token_expiry":  expiry,
        }).Error
}
```

**Step 2: Build check**

```bash
cd auth-service && go build ./...
```

**Step 3: Commit**

```bash
git add auth-service/internal/repository/social_repository.go
git commit -m "feat(auth): add SocialAccount repository"
```

---

### Task 5: Social Usecase

**Files:**
- Create: `auth-service/internal/usecase/social_usecase.go`

**Step 1: Create the file**

```go
// auth-service/internal/usecase/social_usecase.go
package usecase

import (
    "context"
    "encoding/json"
    "errors"
    "net/http"
    "net/url"
    "time"

    "auth-service/internal/config"
    "auth-service/internal/domain"
    "auth-service/internal/repository"
    "auth-service/pkg/crypto"
    "github.com/google/uuid"
    "github.com/redis/go-redis/v9"
)

type SocialUseCase interface {
    GetYouTubeOAuthURL(ctx context.Context, userID uuid.UUID) (string, error)
    HandleYouTubeCallback(ctx context.Context, code, state string) error
    GetConnectedAccounts(ctx context.Context, userID uuid.UUID) ([]domain.SocialAccountResponse, error)
    DisconnectAccount(ctx context.Context, userID uuid.UUID, platform string) error
    GetValidToken(ctx context.Context, userID uuid.UUID, platform string) (string, time.Time, error)
}

type socialUseCase struct {
    socialRepo repository.SocialAccountRepository
    redis      *redis.Client
    config     *config.Config
}

func NewSocialUseCase(socialRepo repository.SocialAccountRepository, redis *redis.Client, cfg *config.Config) SocialUseCase {
    return &socialUseCase{socialRepo: socialRepo, redis: redis, config: cfg}
}

// GetYouTubeOAuthURL generates a Google OAuth2 URL and saves state→userID in Redis.
func (u *socialUseCase) GetYouTubeOAuthURL(ctx context.Context, userID uuid.UUID) (string, error) {
    state := uuid.New().String()
    if err := u.redis.Set(ctx, "social_oauth_state:"+state, userID.String(), 10*time.Minute).Err(); err != nil {
        return "", err
    }
    params := url.Values{
        "client_id":     {u.config.Social.YouTube.ClientID},
        "redirect_uri":  {u.config.Social.YouTube.RedirectURI},
        "response_type": {"code"},
        "scope":         {"https://www.googleapis.com/auth/youtube"},
        "access_type":   {"offline"},
        "prompt":        {"consent"}, // force refresh_token on every consent
        "state":         {state},
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + params.Encode(), nil
}

// HandleYouTubeCallback exchanges code for tokens, fetches channel info, saves encrypted to DB.
func (u *socialUseCase) HandleYouTubeCallback(ctx context.Context, code, state string) error {
    // Validate state
    userIDStr, err := u.redis.GetDel(ctx, "social_oauth_state:"+state).Result()
    if err != nil {
        return errors.New("invalid or expired OAuth state")
    }
    userID, err := uuid.Parse(userIDStr)
    if err != nil {
        return errors.New("invalid user_id in state")
    }

    // Exchange code for tokens
    tokens, err := u.exchangeCode(ctx, code)
    if err != nil {
        return err
    }

    // Get YouTube channel info
    channelID, channelTitle, err := u.getChannelInfo(ctx, tokens.AccessToken)
    if err != nil {
        return err
    }

    // Encrypt tokens
    encAccess, err := crypto.EncryptAES256GCM(u.config.Social.EncryptionKey, tokens.AccessToken)
    if err != nil {
        return err
    }
    encRefresh, err := crypto.EncryptAES256GCM(u.config.Social.EncryptionKey, tokens.RefreshToken)
    if err != nil {
        return err
    }

    // Upsert social account
    account := &domain.SocialAccount{
        ID:               uuid.New(),
        UserID:           userID,
        Platform:         "youtube",
        AccessToken:      encAccess,
        RefreshToken:     encRefresh,
        TokenExpiry:      time.Now().Add(time.Duration(tokens.ExpiresIn) * time.Second),
        Scope:            tokens.Scope,
        PlatformUserID:   channelID,
        PlatformUsername: channelTitle,
    }
    return u.socialRepo.Upsert(ctx, account)
}

// GetConnectedAccounts returns all connected social accounts for a user (no tokens).
func (u *socialUseCase) GetConnectedAccounts(ctx context.Context, userID uuid.UUID) ([]domain.SocialAccountResponse, error) {
    accounts, err := u.socialRepo.FindAllByUser(ctx, userID)
    if err != nil {
        return nil, err
    }
    result := make([]domain.SocialAccountResponse, len(accounts))
    for i, a := range accounts {
        result[i] = domain.SocialAccountResponse{
            ID:               a.ID.String(),
            Platform:         a.Platform,
            PlatformUserID:   a.PlatformUserID,
            PlatformUsername: a.PlatformUsername,
            ConnectedAt:      a.CreatedAt,
        }
    }
    return result, nil
}

// DisconnectAccount removes a connected social account.
func (u *socialUseCase) DisconnectAccount(ctx context.Context, userID uuid.UUID, platform string) error {
    return u.socialRepo.DeleteByUserAndPlatform(ctx, userID, platform)
}

// GetValidToken returns a valid (non-expired) access token, refreshing if needed.
func (u *socialUseCase) GetValidToken(ctx context.Context, userID uuid.UUID, platform string) (string, time.Time, error) {
    account, err := u.socialRepo.FindByUserAndPlatform(ctx, userID, platform)
    if err != nil {
        return "", time.Time{}, errors.New("account not connected")
    }

    // Decrypt access token
    accessToken, err := crypto.DecryptAES256GCM(u.config.Social.EncryptionKey, account.AccessToken)
    if err != nil {
        return "", time.Time{}, err
    }

    // Return if still valid (5min buffer)
    if time.Now().Add(5 * time.Minute).Before(account.TokenExpiry) {
        return accessToken, account.TokenExpiry, nil
    }

    // Expired — refresh
    refreshToken, err := crypto.DecryptAES256GCM(u.config.Social.EncryptionKey, account.RefreshToken)
    if err != nil {
        return "", time.Time{}, err
    }
    newTokens, err := u.refreshToken(ctx, refreshToken)
    if err != nil {
        return "", time.Time{}, err
    }

    newExpiry := time.Now().Add(time.Duration(newTokens.ExpiresIn) * time.Second)
    encAccess, _ := crypto.EncryptAES256GCM(u.config.Social.EncryptionKey, newTokens.AccessToken)
    encRefresh := account.RefreshToken // Google only rotates refresh token if prompt=consent
    if newTokens.RefreshToken != "" {
        encRefresh, _ = crypto.EncryptAES256GCM(u.config.Social.EncryptionKey, newTokens.RefreshToken)
    }
    _ = u.socialRepo.UpdateTokens(ctx, account.ID, encAccess, encRefresh, newExpiry)

    return newTokens.AccessToken, newExpiry, nil
}

// --- Internal helpers ---

type googleTokenResponse struct {
    AccessToken  string `json:"access_token"`
    RefreshToken string `json:"refresh_token"`
    ExpiresIn    int    `json:"expires_in"`
    TokenType    string `json:"token_type"`
    Scope        string `json:"scope"`
    Error        string `json:"error"`
}

func (u *socialUseCase) exchangeCode(ctx context.Context, code string) (*googleTokenResponse, error) {
    data := url.Values{
        "code":          {code},
        "client_id":     {u.config.Social.YouTube.ClientID},
        "client_secret": {u.config.Social.YouTube.ClientSecret},
        "redirect_uri":  {u.config.Social.YouTube.RedirectURI},
        "grant_type":    {"authorization_code"},
    }
    resp, err := http.PostForm("https://oauth2.googleapis.com/token", data)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    var tokens googleTokenResponse
    if err := json.NewDecoder(resp.Body).Decode(&tokens); err != nil {
        return nil, err
    }
    if tokens.Error != "" {
        return nil, errors.New("Google token exchange error: " + tokens.Error)
    }
    if tokens.AccessToken == "" {
        return nil, errors.New("no access_token in response")
    }
    return &tokens, nil
}

func (u *socialUseCase) refreshToken(ctx context.Context, refreshToken string) (*googleTokenResponse, error) {
    data := url.Values{
        "refresh_token": {refreshToken},
        "client_id":     {u.config.Social.YouTube.ClientID},
        "client_secret": {u.config.Social.YouTube.ClientSecret},
        "grant_type":    {"refresh_token"},
    }
    resp, err := http.PostForm("https://oauth2.googleapis.com/token", data)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    var tokens googleTokenResponse
    if err := json.NewDecoder(resp.Body).Decode(&tokens); err != nil {
        return nil, err
    }
    if tokens.Error != "" {
        return nil, errors.New("Google token refresh error: " + tokens.Error)
    }
    return &tokens, nil
}

func (u *socialUseCase) getChannelInfo(ctx context.Context, accessToken string) (id, title string, err error) {
    req, err := http.NewRequestWithContext(ctx, "GET",
        "https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true", nil)
    if err != nil {
        return "", "", err
    }
    req.Header.Set("Authorization", "Bearer "+accessToken)
    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return "", "", err
    }
    defer resp.Body.Close()
    var result struct {
        Items []struct {
            ID      string `json:"id"`
            Snippet struct {
                Title string `json:"title"`
            } `json:"snippet"`
        } `json:"items"`
    }
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return "", "", err
    }
    if len(result.Items) == 0 {
        return "", "", errors.New("no YouTube channel found for this account")
    }
    return result.Items[0].ID, result.Items[0].Snippet.Title, nil
}
```

**Step 2: Build check**

```bash
cd auth-service && go build ./...
```

**Step 3: Commit**

```bash
git add auth-service/internal/usecase/social_usecase.go
git commit -m "feat(auth): add Social usecase with YouTube OAuth and token refresh"
```

---

### Task 6: Social Handler + Routes

**Files:**
- Create: `auth-service/internal/handler/social_handler.go`
- Modify: `auth-service/internal/handler/router.go`
- Modify: `auth-service/cmd/server/main.go` (wire dependencies)

**Step 1: Create social_handler.go**

```go
// auth-service/internal/handler/social_handler.go
package handler

import (
    "net/http"

    "auth-service/internal/usecase"
    "auth-service/internal/config"
    "github.com/gin-gonic/gin"
    "github.com/google/uuid"
)

type SocialHandler struct {
    socialUC usecase.SocialUseCase
    config   *config.Config
}

func NewSocialHandler(socialUC usecase.SocialUseCase, cfg *config.Config) *SocialHandler {
    return &SocialHandler{socialUC: socialUC, config: cfg}
}

// GET /api/v1/social/auth/youtube/start — returns Google OAuth URL
func (h *SocialHandler) StartYouTubeOAuth(c *gin.Context) {
    userID := c.MustGet("user_id").(uuid.UUID)
    authURL, err := h.socialUC.GetYouTubeOAuthURL(c.Request.Context(), userID)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to generate OAuth URL"})
        return
    }
    c.JSON(http.StatusOK, gin.H{"url": authURL})
}

// GET /api/v1/social/auth/youtube/callback — OAuth callback from Google
func (h *SocialHandler) YouTubeCallback(c *gin.Context) {
    code := c.Query("code")
    state := c.Query("state")
    if code == "" || state == "" {
        c.Redirect(http.StatusFound, h.config.Social.FrontendURL+"/settings?error=missing_params")
        return
    }
    if err := h.socialUC.HandleYouTubeCallback(c.Request.Context(), code, state); err != nil {
        c.Redirect(http.StatusFound, h.config.Social.FrontendURL+"/settings?error=youtube_connect_failed")
        return
    }
    c.Redirect(http.StatusFound, h.config.Social.FrontendURL+"/settings?connected=youtube")
}

// GET /api/v1/social/accounts — list connected accounts
func (h *SocialHandler) GetAccounts(c *gin.Context) {
    userID := c.MustGet("user_id").(uuid.UUID)
    accounts, err := h.socialUC.GetConnectedAccounts(c.Request.Context(), userID)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to fetch accounts"})
        return
    }
    c.JSON(http.StatusOK, accounts)
}

// DELETE /api/v1/social/accounts/:platform — disconnect an account
func (h *SocialHandler) DisconnectAccount(c *gin.Context) {
    userID := c.MustGet("user_id").(uuid.UUID)
    platform := c.Param("platform")
    if err := h.socialUC.DisconnectAccount(c.Request.Context(), userID, platform); err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to disconnect"})
        return
    }
    c.JSON(http.StatusOK, gin.H{"message": "disconnected"})
}

// GET /internal/social/token/:user_id/:platform — engine calls this to get valid token
func (h *SocialHandler) GetInternalToken(c *gin.Context) {
    userIDStr := c.Param("user_id")
    platform := c.Param("platform")
    userID, err := uuid.Parse(userIDStr)
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "invalid user_id"})
        return
    }
    accessToken, expiry, err := h.socialUC.GetValidToken(c.Request.Context(), userID, platform)
    if err != nil {
        c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
        return
    }
    c.JSON(http.StatusOK, gin.H{"access_token": accessToken, "token_expiry": expiry})
}
```

**Step 2: Register routes in router.go**

In `router.go`, find where `setupInternalRoutes` is called and add `setupSocialRoutes` next to it. The `Router` struct will need a `SocialHandler` field added to it.

In the `Router` struct (find it in `router.go`), add:
```go
socialHandler *SocialHandler
```

In the `NewRouter` constructor, add `socialHandler *SocialHandler` parameter and assign it.

Add the setup method at the bottom of `router.go`:
```go
func (r *Router) setupSocialRoutes(v1 *gin.RouterGroup) {
    // JWT-protected social routes
    social := v1.Group("/social")
    social.Use(r.authMiddleware.RequireAuth())
    {
        social.GET("/auth/youtube/start", r.socialHandler.StartYouTubeOAuth)
        social.GET("/accounts", r.socialHandler.GetAccounts)
        social.DELETE("/accounts/:platform", r.socialHandler.DisconnectAccount)
    }

    // OAuth callback — public (Google redirects here, no JWT)
    v1.GET("/social/auth/youtube/callback", r.socialHandler.YouTubeCallback)
}
```

Add to `Setup()`:
```go
r.setupSocialRoutes(v1)
```

**Step 3: Wire in main.go**

In `cmd/server/main.go`, after creating the existing repositories and usecases, add:

```go
// Social
socialRepo := repository.NewSocialAccountRepository(db)
socialUC := usecase.NewSocialUseCase(socialRepo, redisClient, cfg)
socialHandler := handler.NewSocialHandler(socialUC, cfg)
```

Pass `socialHandler` to `NewRouter(...)`.

**Step 4: Build check**

```bash
cd auth-service && go build ./...
```

**Step 5: Commit**

```bash
git add auth-service/internal/handler/social_handler.go auth-service/internal/handler/router.go auth-service/cmd/
git commit -m "feat(auth): add Social handler and routes for YouTube OAuth"
```

---

### Task 7: AutoMigrate + Add golang.org/x/oauth2

**Files:**
- Modify: `auth-service/internal/database/migrations.go`

**Step 1: Add SocialAccount to AutoMigrate**

In `migrations.go`, inside the `db.AutoMigrate(...)` call, add `&domain.SocialAccount{}`:

```go
if err := db.AutoMigrate(
    &domain.User{},
    &domain.Role{},
    &domain.Permission{},
    &domain.RefreshToken{},
    &domain.UserActivity{},
    &domain.SocialAccount{},  // ← add this line
); err != nil {
    return err
}
```

**Step 2: Build and run auth-service to verify table creation**

```bash
cd auth-service
go build ./...
go run ./cmd/server
```
Expected: server starts, logs show `social_accounts` table created (or already exists).

**Step 3: Commit**

```bash
git add auth-service/internal/database/migrations.go
git commit -m "feat(auth): add social_accounts table migration"
```

---

## Phase 2: Engine (Python)

### Task 8: ClipUpload Model

**Files:**
- Modify: `engine/app/models/__init__.py`

**Step 1: Add ClipUpload model**

At the bottom of `__init__.py`, add:

```python
from uuid import uuid4

class ClipUpload(Base):
    """Upload record for a clip to a social media platform."""

    __tablename__ = "clip_uploads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    clip_id: Mapped[str] = mapped_column(String(255), ForeignKey("clips.clip_id", ondelete="CASCADE"))
    user_id: Mapped[str] = mapped_column(String(36), nullable=False)
    platform: Mapped[str] = mapped_column(String(50), nullable=False)  # "youtube"
    status: Mapped[str] = mapped_column(String(50), default="uploading")  # uploading|completed|failed
    platform_video_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    platform_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    privacy: Mapped[str] = mapped_column(String(20), default="private")  # public|unlisted|private
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
```

**Step 2: Verify init_db creates the table**

`init_db()` in `core/database.py` calls `Base.metadata.create_all(bind=engine)` — the new table will be created automatically on next startup.

```bash
cd engine && uv run python -c "from app.models import ClipUpload; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add engine/app/models/__init__.py
git commit -m "feat(engine): add ClipUpload model for social media upload tracking"
```

---

### Task 9: Engine Config + httpx Dependency

**Files:**
- Modify: `engine/app/core/config.py`

**Step 1: Add social config fields**

In `Settings` class, add after the existing fields:

```python
# Social / Auth-service communication
AUTH_SERVICE_URL: str = os.getenv("AUTH_SERVICE_URL", "http://localhost:8005")
AUTH_SERVICE_TOKEN: str = os.getenv("AUTH_SERVICE_TOKEN", "")
```

**Step 2: Add env vars to engine/.env**

```env
AUTH_SERVICE_URL=http://localhost:8005
AUTH_SERVICE_TOKEN=your_shared_service_token   # must match auth-service SERVICE_TOKEN
```

**Step 3: Add httpx**

```bash
cd engine && uv add httpx
```

**Step 4: Build check**

```bash
cd engine && uv run python -c "from app.core.config import settings; print(settings.AUTH_SERVICE_URL)"
```
Expected: `http://localhost:8005`

**Step 5: Commit**

```bash
git add engine/app/core/config.py engine/.env
git commit -m "feat(engine): add auth-service URL/token config and httpx dependency"
```

---

### Task 10: YouTube Upload Utility

**Files:**
- Create: `engine/app/utils/youtube_uploader.py`

**Step 1: Create the file**

```python
# engine/app/utils/youtube_uploader.py
"""YouTube Data API v3 — resumable upload utility."""

import asyncio
from pathlib import Path
import httpx

YOUTUBE_UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
CHUNK_SIZE = 5 * 1024 * 1024  # 5 MB chunks


async def upload_to_youtube(
    access_token: str,
    clip_path: str,
    title: str,
    description: str,
    tags: list[str],
    privacy: str,  # "public" | "unlisted" | "private"
) -> dict:
    """Upload a video file to YouTube via resumable upload.

    Returns {"video_id": str, "url": str}.
    Raises httpx.HTTPStatusError or Exception on failure.
    """
    file_path = Path(clip_path)
    file_size = file_path.stat().st_size

    metadata = {
        "snippet": {
            "title": title,
            "description": description or "",
            "tags": tags or [],
            "categoryId": "22",  # People & Blogs
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Initiate resumable upload session
        init_resp = await client.post(
            f"{YOUTUBE_UPLOAD_URL}?uploadType=resumable&part=snippet,status",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "X-Upload-Content-Type": "video/mp4",
                "X-Upload-Content-Length": str(file_size),
            },
            json=metadata,
        )
        init_resp.raise_for_status()
        upload_url = init_resp.headers["Location"]

    # Step 2: Upload file in chunks (new client for long-running upload)
    video_id = await _upload_chunks(upload_url, file_path, file_size)
    return {
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
    }


async def _upload_chunks(upload_url: str, file_path: Path, file_size: int) -> str:
    """Upload file in 5MB chunks using YouTube resumable upload protocol."""
    offset = 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        with open(file_path, "rb") as f:
            while offset < file_size:
                chunk = f.read(CHUNK_SIZE)
                chunk_len = len(chunk)
                end = offset + chunk_len - 1

                resp = await client.put(
                    upload_url,
                    content=chunk,
                    headers={
                        "Content-Range": f"bytes {offset}-{end}/{file_size}",
                        "Content-Type": "video/mp4",
                    },
                )

                if resp.status_code in (200, 201):
                    data = resp.json()
                    return data["id"]
                elif resp.status_code == 308:
                    # Resume Incomplete — continue
                    range_header = resp.headers.get("Range", "")
                    if range_header:
                        offset = int(range_header.split("-")[1]) + 1
                    else:
                        offset += chunk_len
                else:
                    raise Exception(
                        f"YouTube upload failed: {resp.status_code} {resp.text}"
                    )

    raise Exception("Upload ended without receiving video ID")
```

**Step 2: Verify import**

```bash
cd engine && uv run python -c "from app.utils.youtube_uploader import upload_to_youtube; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add engine/app/utils/youtube_uploader.py
git commit -m "feat(engine): add YouTube resumable upload utility"
```

---

### Task 11: Social Routes + Register in main.py

**Files:**
- Create: `engine/app/api/routes/social.py`
- Modify: `engine/main.py`

**Step 1: Create social.py**

```python
# engine/app/api/routes/social.py
"""Social media upload endpoints."""

import asyncio
import uuid
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from app.core.config import settings
from app.core.database import async_session
from app.middleware.auth import CurrentUser, get_current_user
from app.models import Clip, ClipUpload
from app.utils.youtube_uploader import upload_to_youtube
from app.utils.logging import logger

router = APIRouter()


class UploadRequest(BaseModel):
    clip_id: str
    platform: str  # "youtube"
    title: str
    description: str | None = None
    tags: list[str] | None = None
    privacy: str = "private"  # public | unlisted | private


# POST /social/upload — trigger upload, return upload_id immediately
@router.post("/social/upload")
async def start_upload(
    req: UploadRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    async with async_session() as session:
        # Verify clip exists
        result = await session.execute(
            select(Clip).where(Clip.clip_id == req.clip_id)
        )
        clip = result.scalar_one_or_none()
        if not clip:
            raise HTTPException(status_code=404, detail="Clip not found")

        # Get valid access token from auth-service
        access_token = await _get_token(str(current_user.id), req.platform)

        # Create upload record
        upload_id = str(uuid.uuid4())
        upload = ClipUpload(
            id=upload_id,
            clip_id=req.clip_id,
            user_id=str(current_user.id),
            platform=req.platform,
            status="uploading",
            title=req.title,
            description=req.description,
            tags=req.tags,
            privacy=req.privacy,
        )
        session.add(upload)
        await session.commit()
        clip_path = clip.file_path

    # Fire background task (don't await)
    asyncio.create_task(
        _run_upload(upload_id, access_token, clip_path, req)
    )

    return {"upload_id": upload_id, "status": "uploading"}


# GET /social/upload/{upload_id} — poll status
@router.get("/social/upload/{upload_id}")
async def get_upload_status(
    upload_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    async with async_session() as session:
        upload = await session.get(ClipUpload, upload_id)
        if not upload or upload.user_id != str(current_user.id):
            raise HTTPException(status_code=404, detail="Upload not found")
        return {
            "upload_id": upload.id,
            "status": upload.status,
            "platform_video_id": upload.platform_video_id,
            "platform_url": upload.platform_url,
            "error": upload.error,
        }


# GET /social/uploads — upload history for current user
@router.get("/social/uploads")
async def list_uploads(
    current_user: CurrentUser = Depends(get_current_user),
):
    async with async_session() as session:
        result = await session.execute(
            select(ClipUpload)
            .where(ClipUpload.user_id == str(current_user.id))
            .order_by(ClipUpload.created_at.desc())
            .limit(50)
        )
        uploads = result.scalars().all()
        return [
            {
                "upload_id": u.id,
                "clip_id": u.clip_id,
                "platform": u.platform,
                "status": u.status,
                "platform_url": u.platform_url,
                "title": u.title,
                "created_at": u.created_at.isoformat(),
            }
            for u in uploads
        ]


async def _get_token(user_id: str, platform: str) -> str:
    """Fetch a valid access token from auth-service internal endpoint."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{settings.AUTH_SERVICE_URL}/api/v1/internal/social/token/{user_id}/{platform}",
            headers={"X-Service-Token": settings.AUTH_SERVICE_TOKEN},
        )
        if resp.status_code == 404:
            raise HTTPException(
                status_code=400,
                detail=f"{platform} account not connected. Go to Settings to connect.",
            )
        resp.raise_for_status()
        return resp.json()["access_token"]


async def _run_upload(
    upload_id: str,
    access_token: str,
    clip_path: str,
    req: UploadRequest,
) -> None:
    """Background task: upload video and update ClipUpload status."""
    async with async_session() as session:
        try:
            logger.info(f"Starting YouTube upload for upload_id={upload_id}")
            result = await upload_to_youtube(
                access_token=access_token,
                clip_path=clip_path,
                title=req.title,
                description=req.description or "",
                tags=req.tags or [],
                privacy=req.privacy,
            )
            upload = await session.get(ClipUpload, upload_id)
            if upload:
                upload.status = "completed"
                upload.platform_video_id = result["video_id"]
                upload.platform_url = result["url"]
                upload.completed_at = datetime.now(timezone.utc)
                await session.commit()
            logger.info(f"Upload completed: {result['url']}")
        except Exception as e:
            logger.error(f"Upload failed for {upload_id}: {e}")
            upload = await session.get(ClipUpload, upload_id)
            if upload:
                upload.status = "failed"
                upload.error = str(e)
                await session.commit()
```

**Step 2: Register router in main.py**

In `main.py`, add:
```python
from app.api.routes import transcribe, jobs, social  # add social
```

After the existing `app.include_router(jobs.router, ...)`, add:
```python
app.include_router(
    social.router, prefix=settings.API_V1_STR, tags=["Social Upload"]
)
```

**Step 3: Start engine to verify routes registered**

```bash
cd engine && uv run python main.py
```
Open `http://localhost:8000/docs` — verify `/social/upload`, `/social/upload/{upload_id}`, `/social/uploads` appear.

**Step 4: Commit**

```bash
git add engine/app/api/routes/social.py engine/main.py
git commit -m "feat(engine): add social upload routes (YouTube)"
```

---

## Phase 3: Frontend (Next.js)

### Task 12: social-api.ts

**Files:**
- Create: `coclip-frontend/src/lib/social-api.ts`

**Step 1: Create the file**

```typescript
// coclip-frontend/src/lib/social-api.ts
import { AUTH_BASE, ENGINE_BASE } from "@/lib/api";

// --- Types ---

export interface SocialAccount {
  id: string;
  platform: string;
  platform_user_id: string;
  platform_username: string;
  connected_at: string;
}

export interface UploadStatus {
  upload_id: string;
  status: "uploading" | "completed" | "failed";
  platform_video_id?: string;
  platform_url?: string;
  error?: string;
}

export interface UploadRequest {
  clip_id: string;
  platform: string;
  title: string;
  description?: string;
  tags?: string[];
  privacy: "public" | "unlisted" | "private";
}

type FetchFn = (path: string, options?: RequestInit) => Promise<Response>;

// --- Auth-service calls ---

/** Returns the list of connected social accounts. */
export async function getSocialAccounts(authFetch: FetchFn): Promise<SocialAccount[]> {
  const res = await authFetch("/social/accounts");
  if (!res.ok) throw new Error("Failed to fetch connected accounts");
  return res.json();
}

/** Returns the Google OAuth redirect URL — redirect the browser to it. */
export async function getYouTubeOAuthUrl(authFetch: FetchFn): Promise<string> {
  const res = await authFetch("/social/auth/youtube/start");
  if (!res.ok) throw new Error("Failed to start YouTube OAuth");
  const data = await res.json();
  return data.url;
}

/** Disconnects a social account. */
export async function disconnectAccount(authFetch: FetchFn, platform: string): Promise<void> {
  const res = await authFetch(`/social/accounts/${platform}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Failed to disconnect ${platform}`);
}

// --- Engine calls ---

/** Starts a clip upload. Returns the upload_id for polling. */
export async function startUpload(
  engineFetch: FetchFn,
  req: UploadRequest,
): Promise<{ upload_id: string }> {
  const res = await engineFetch("/social/upload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Upload failed");
  }
  return res.json();
}

/** Polls upload status by upload_id. */
export async function getUploadStatus(
  engineFetch: FetchFn,
  uploadId: string,
): Promise<UploadStatus> {
  const res = await engineFetch(`/social/upload/${uploadId}`);
  if (!res.ok) throw new Error("Failed to get upload status");
  return res.json();
}
```

**Step 2: Verify import**

```bash
cd coclip-frontend && npx tsc --noEmit
```
Expected: no errors related to social-api.ts

**Step 3: Commit**

```bash
git add coclip-frontend/src/lib/social-api.ts
git commit -m "feat(frontend): add social-api.ts for YouTube OAuth and upload calls"
```

---

### Task 13: Settings Page

**Files:**
- Create: `coclip-frontend/src/app/(app)/settings/page.tsx`

**Step 1: Create the file**

```tsx
// coclip-frontend/src/app/(app)/settings/page.tsx
"use client";

import { useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useAuth } from "@/contexts/auth-context";
import {
  getSocialAccounts,
  getYouTubeOAuthUrl,
  disconnectAccount,
  type SocialAccount,
} from "@/lib/social-api";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Loader2, Youtube } from "lucide-react";
import { toast } from "sonner";

export default function SettingsPage() {
  const { authFetch } = useAuth();
  const searchParams = useSearchParams();

  const [accounts, setAccounts] = useState<SocialAccount[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isConnecting, setIsConnecting] = useState(false);
  const [disconnecting, setDisconnecting] = useState<string | null>(null);

  const fetchAccounts = useCallback(async () => {
    try {
      const data = await getSocialAccounts(authFetch);
      setAccounts(data);
    } catch {
      toast.error("Failed to load connected accounts");
    } finally {
      setIsLoading(false);
    }
  }, [authFetch]);

  useEffect(() => {
    fetchAccounts();
  }, [fetchAccounts]);

  // Handle redirect back from OAuth
  useEffect(() => {
    const connected = searchParams.get("connected");
    const error = searchParams.get("error");
    if (connected) {
      toast.success(`${connected} connected successfully!`);
      fetchAccounts();
    }
    if (error) {
      toast.error(`Failed to connect: ${error.replace(/_/g, " ")}`);
    }
  }, [searchParams, fetchAccounts]);

  const handleConnect = async (platform: string) => {
    setIsConnecting(true);
    try {
      if (platform === "youtube") {
        const url = await getYouTubeOAuthUrl(authFetch);
        window.location.href = url;
      }
    } catch {
      toast.error(`Failed to start ${platform} connection`);
      setIsConnecting(false);
    }
  };

  const handleDisconnect = async (platform: string) => {
    setDisconnecting(platform);
    try {
      await disconnectAccount(authFetch, platform);
      setAccounts((prev) => prev.filter((a) => a.platform !== platform));
      toast.success(`${platform} disconnected`);
    } catch {
      toast.error(`Failed to disconnect ${platform}`);
    } finally {
      setDisconnecting(null);
    }
  };

  const youtubeAccount = accounts.find((a) => a.platform === "youtube");

  return (
    <div className="max-w-2xl mx-auto px-4 py-8 space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Settings</h1>
        <p className="text-muted-foreground text-sm mt-1">Manage your connected accounts</p>
      </div>

      <div className="space-y-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Connected Accounts
        </h2>

        {isLoading ? (
          <div className="flex items-center gap-2 text-muted-foreground text-sm py-4">
            <Loader2 className="w-4 h-4 animate-spin" />
            Loading…
          </div>
        ) : (
          <div className="border border-border rounded-lg divide-y divide-border">
            {/* YouTube row */}
            <div className="flex items-center justify-between px-4 py-3.5">
              <div className="flex items-center gap-3">
                <Youtube className="w-5 h-5 text-red-500" />
                <div>
                  <p className="text-sm font-medium">YouTube</p>
                  {youtubeAccount ? (
                    <p className="text-xs text-muted-foreground">
                      {youtubeAccount.platform_username}
                    </p>
                  ) : (
                    <p className="text-xs text-muted-foreground">Not connected</p>
                  )}
                </div>
              </div>

              {youtubeAccount ? (
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="text-xs bg-emerald-500/10 text-emerald-400 border-emerald-500/20">
                    Connected
                  </Badge>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-xs h-7 text-muted-foreground hover:text-destructive"
                    disabled={disconnecting === "youtube"}
                    onClick={() => handleDisconnect("youtube")}
                  >
                    {disconnecting === "youtube" ? (
                      <Loader2 className="w-3 h-3 animate-spin" />
                    ) : (
                      "Disconnect"
                    )}
                  </Button>
                </div>
              ) : (
                <Button
                  size="sm"
                  className="h-7 text-xs"
                  disabled={isConnecting}
                  onClick={() => handleConnect("youtube")}
                >
                  {isConnecting ? (
                    <><Loader2 className="w-3 h-3 mr-1.5 animate-spin" />Connecting…</>
                  ) : (
                    "Connect"
                  )}
                </Button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
```

**Step 2: Add Settings link to navbar**

Find the navbar component (likely in `src/app/(app)/layout.tsx` or a Navbar component). Add a Settings link pointing to `/settings`.

**Step 3: Build check**

```bash
cd coclip-frontend && npm run build 2>&1 | tail -20
```
Expected: no errors

**Step 4: Commit**

```bash
git add coclip-frontend/src/app/(app)/settings/
git commit -m "feat(frontend): add Settings page with YouTube account connect/disconnect"
```

---

### Task 14: ClipDetailModal — Upload Section

**Files:**
- Modify: `coclip-frontend/src/app/(app)/jobs/[id]/ClipDetailModal.tsx`

**Step 1: Add imports and state**

At the top of `ClipDetailModal.tsx`, add to imports:
```tsx
import { useEffect, useRef, useState } from "react";  // already there
import { useAuth } from "@/contexts/auth-context";     // add this
import {
  getSocialAccounts,
  startUpload,
  getUploadStatus,
  type UploadStatus,
} from "@/lib/social-api";                            // add this
import { ExternalLink, Upload } from "lucide-react";   // add these icons
import { Input } from "@/components/ui/input";          // add this
import { Textarea } from "@/components/ui/textarea";    // add this (or use existing)
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";                       // add if not already imported
```

Note: Check if `Input`, `Textarea`, `Select` are in `src/components/ui/` — if not, run:
```bash
npx shadcn@latest add textarea select
```

**Step 2: Add upload state to ClipDetailModal**

Inside the `ClipDetailModal` function, after the existing `useState` calls, add:

```tsx
const { authFetch, engineFetch } = useAuth();
const [youtubeConnected, setYoutubeConnected] = useState<boolean | null>(null);
const [uploadTitle, setUploadTitle] = useState(clip.title ?? "");
const [uploadDesc, setUploadDesc] = useState(clip.suggested_caption ?? "");
const [uploadPrivacy, setUploadPrivacy] = useState<"public" | "unlisted" | "private">("private");
const [uploadStatus, setUploadStatus] = useState<UploadStatus | null>(null);
const [isUploading, setIsUploading] = useState(false);
const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

// Reset upload state when clip changes
useEffect(() => {
  setUploadTitle(clip.title ?? "");
  setUploadDesc(clip.suggested_caption ?? "");
  setUploadStatus(null);
  setIsUploading(false);
  if (pollRef.current) clearInterval(pollRef.current);
}, [clip.clip_id]);

// Check if YouTube is connected
useEffect(() => {
  getSocialAccounts(authFetch)
    .then((accounts) => setYoutubeConnected(accounts.some((a) => a.platform === "youtube")))
    .catch(() => setYoutubeConnected(false));
}, [authFetch]);

// Cleanup poll on unmount
useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

const handleUpload = async () => {
  setIsUploading(true);
  try {
    const { upload_id } = await startUpload(engineFetch, {
      clip_id: clip.clip_id,
      platform: "youtube",
      title: uploadTitle,
      description: uploadDesc,
      tags: clip.tags ?? [],
      privacy: uploadPrivacy,
    });
    setUploadStatus({ upload_id, status: "uploading" });

    // Poll every 4 seconds
    pollRef.current = setInterval(async () => {
      try {
        const status = await getUploadStatus(engineFetch, upload_id);
        setUploadStatus(status);
        if (status.status !== "uploading") {
          clearInterval(pollRef.current!);
          setIsUploading(false);
          if (status.status === "completed") toast.success("Uploaded to YouTube!");
          else toast.error(`Upload failed: ${status.error}`);
        }
      } catch {
        clearInterval(pollRef.current!);
        setIsUploading(false);
      }
    }, 4000);
  } catch (e: unknown) {
    toast.error(e instanceof Error ? e.message : "Upload failed");
    setIsUploading(false);
  }
};
```

**Step 3: Add upload UI in the scrollable content section**

In the JSX, inside `{/* Scrollable content */}`, after the Tags section (before Reasoning), add:

```tsx
{/* YouTube Upload */}
<div className="space-y-3 border-t border-border pt-4">
  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
    Upload to YouTube
  </p>

  {youtubeConnected === null ? (
    <p className="text-xs text-muted-foreground">Checking connection…</p>
  ) : youtubeConnected === false ? (
    <p className="text-xs text-muted-foreground">
      <a href="/settings" className="underline hover:text-foreground">Connect YouTube</a> in Settings to upload.
    </p>
  ) : uploadStatus?.status === "completed" ? (
    <div className="flex items-center gap-2 text-sm text-emerald-400">
      <Check className="w-4 h-4" />
      Uploaded!{" "}
      <a
        href={uploadStatus.platform_url}
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center gap-0.5 underline hover:text-emerald-300"
      >
        Watch <ExternalLink className="w-3 h-3" />
      </a>
    </div>
  ) : (
    <div className="space-y-2">
      <Input
        value={uploadTitle}
        onChange={(e) => setUploadTitle(e.target.value)}
        placeholder="Video title"
        className="text-sm h-8"
        disabled={isUploading}
      />
      <Textarea
        value={uploadDesc}
        onChange={(e) => setUploadDesc(e.target.value)}
        placeholder="Description"
        className="text-sm min-h-[60px] resize-none"
        disabled={isUploading}
      />
      <Select
        value={uploadPrivacy}
        onValueChange={(v) => setUploadPrivacy(v as "public" | "unlisted" | "private")}
        disabled={isUploading}
      >
        <SelectTrigger className="h-8 text-sm">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="private">Private</SelectItem>
          <SelectItem value="unlisted">Unlisted</SelectItem>
          <SelectItem value="public">Public</SelectItem>
        </SelectContent>
      </Select>
      {uploadStatus?.status === "failed" && (
        <p className="text-xs text-destructive">{uploadStatus.error}</p>
      )}
      <Button
        size="sm"
        className="w-full h-8 text-xs"
        onClick={handleUpload}
        disabled={isUploading || !uploadTitle.trim()}
      >
        {isUploading ? (
          <><Loader2 className="w-3 h-3 mr-1.5 animate-spin" />Uploading…</>
        ) : (
          <><Upload className="w-3 h-3 mr-1.5" />Upload to YouTube</>
        )}
      </Button>
    </div>
  )}
</div>
```

**Step 4: Build check**

```bash
cd coclip-frontend && npm run build 2>&1 | tail -20
```
Expected: no errors

**Step 5: Commit**

```bash
git add coclip-frontend/src/app/(app)/jobs/[id]/ClipDetailModal.tsx
git commit -m "feat(frontend): add YouTube upload section to ClipDetailModal"
```

---

## End-to-End Test Checklist

After all tasks are complete, verify the full flow:

1. [x] Start auth-service — verify `social_accounts` table created in auth DB
2. [x] Start engine — verify `clip_uploads` table created in engine DB
3. [x] Start frontend — go to `/settings`
4. [x] Click **Connect YouTube** → redirected to Google OAuth
5. [x] Authorize → redirected to `/settings?connected=youtube` → see channel name
6. [x] Open a completed job with clips → open ClipDetailModal
7. [x] Fill in title, description, privacy=unlisted → click Upload
8. [x] Watch status change to uploading → completed
9. [x] Click YouTube link → verify video uploaded correctly
10. [x] Go back to Settings → click Disconnect → YouTube row shows "Not connected"

---

## Google Cloud Console Setup

Before testing, set up YouTube Data API v3:

1. Go to https://console.cloud.google.com
2. Create a project (or use existing)
3. Enable **YouTube Data API v3** (APIs & Services → Enable APIs)
4. Create **OAuth 2.0 Client ID** (APIs & Services → Credentials → Create Credentials → OAuth Client ID)
   - Application type: **Web application**
   - Authorized redirect URIs: `http://localhost:8005/api/v1/social/auth/youtube/callback`
5. Copy Client ID and Client Secret to auth-service `.env`
6. Configure **OAuth consent screen** — add your email as test user (while in testing mode)
