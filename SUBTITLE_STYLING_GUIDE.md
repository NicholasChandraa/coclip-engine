# Modern Subtitle Styling Guide

Quick reference untuk customize subtitle appearance.

## 🎨 Current vs Modern Styles

### Current Style (Classic Karaoke)
```
❌ Yellow highlight color (&H0000FFFF) — old-school karaoke
❌ Karaoke effect ({\\kf}) — distracting
```

### Modern Options

#### 1. **TikTok Style** (Recommended)
```
✅ Pure white text (&H00FFFFFF)
✅ Thick black outline (Outline=3)
✅ NO karaoke/highlight effect
✅ Bold font (Bold=-1)
✅ Bigger size (36px)
```

#### 2. **Instagram Reels Style**
```
✅ All caps text
✅ White text with strong shadow (Shadow=3)
✅ Bold heavy font (Arial Black)
✅ Clean and simple
```

#### 3. **Minimal Clean**
```
✅ White text only
✅ Subtle outline (Outline=2)
✅ No effects
✅ Easy to read
```

---

## 🔧 Quick Test Tool

Test subtitle styles tanpa process full video:

```bash
# Try modern style
python test_subtitle_preview.py --style modern

# Try TikTok style  
python test_subtitle_preview.py --style tiktok

# Try minimal style
python test_subtitle_preview.py --style minimal

# Test on specific video
python test_subtitle_preview.py --style modern --video path/to/video.mp4
```

Tool ini akan:
1. Generate test ASS file dengan sample text
2. Render 10 detik pertama clip kamu dengan subtitle tersebut
3. Output: `subtitle_preview.mp4` yang bisa langsung dibuka

---

## 🎯 Recommended Modern Style

Edit `app/utils/subtitle_generator.py`, function `_generate_ass_header()`:

**Replace this line:**
```python
Style: Default,Arial,{font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2.5,0,2,10,10,{margin_bottom},1
```

**With (TikTok style):**
```python
Style: Default,Arial,36,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,{margin_bottom},1
```

### What Changed:
- `Fontsize`: 28 → **36** (bigger)
- `SecondaryColour`: `&H0000FFFF` (yellow) → `&H00FFFFFF` (white) — NO MORE KARAOKE HIGHLIGHT
- `BackColour`: `&H80000000` (semi-transparent black) → `&H00000000` (solid black)
- `Outline`: 2.5 → **3** (thicker outline)

---

## 🎨 ASS Color Reference

ASS colors use format `&HAABBGGRR` (hex, alpha-blue-green-red):

| Color | Code | Use |
|-------|------|-----|
| **White** | `&H00FFFFFF` | Primary text color |
| **Black** | `&H00000000` | Outline, background |
| **Yellow** | `&H0000FFFF` | Old karaoke style (avoid) |
| **Red** | `&H000000FF` | Accent text |
| **Transparent** | `&HFF000000` | Invisible background |

**Alpha channel:**
- `00` = opaque
- `80` = 50% transparent
- `FF` = fully transparent

---

## 📝 ASS Style Parameters Explained

Full style string format:
```
Style: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
```

### Key Parameters:
- **Fontname**: `Arial`, `Arial Black`, `Helvetica`, `Impact`
- **Fontsize**: `28-40` (TikTok uses ~36)
- **Bold**: `-1` (yes), `0` (no)
- **BorderStyle**: `1` (outline+shadow), `3` (box background)
- **Outline**: `2-4` (outline thickness in pixels)
- **Shadow**: `0-3` (drop shadow depth)
- **Alignment**: `2` (bottom center), `5` (middle center), `8` (top center)
- **MarginV**: Bottom margin in pixels

---

## ⚡ Quick Apply: Modern Style

Want TikTok-style subtitles right now?

1. Run preview tool to see options:
```bash
python test_subtitle_preview.py --style tiktok
python test_subtitle_preview.py --style modern
```

2. Pick your favorite → I'll update `subtitle_generator.py` untuk kamu!

3. Restart server, process video baru → modern subtitles ✨
