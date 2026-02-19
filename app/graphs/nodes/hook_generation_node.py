"""
Hook generation node for LangGraph video processing pipeline.

Implements Phase 3A: Gemini hook text generation + TTS voiceover (50% -> 58%)
Generates a short hook intro text and voiceover audio for each clip.
"""

import os
import json
import asyncio
from typing import Literal
from langgraph.types import Command
from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.graph_schemas import VideoProcessingState
from app.utils.progress_tracker import create_progress_tracker
from app.utils.tts_engine import TTSEngine
from app.utils.logging import logger
from app.core.config import settings
from redis import asyncio as aioredis


async def hook_generation_node(
    state: VideoProcessingState, redis: aioredis.Redis
) -> Command[Literal["editing"]]:
    """
    Phase 3A: Generate hook text + TTS voiceover per clip.

    Pipeline steps (50% -> 58%):
    1. Gemini batch call for hook text (50% -> 54%)
    2. TTS synthesis for each hook (54% -> 58%)

    Args:
        state: Current LangGraph state
        redis: Async Redis connection for progress tracking

    Returns:
        Command with hooks data routing to editing
    """
    job_id = state["job_id"]
    tracker = create_progress_tracker(redis, job_id)

    # Skip if hooks disabled
    if not settings.ENABLE_HOOKS:
        logger.info(f"[Job {job_id}] Hooks disabled, skipping to editing")
        return Command(
            update={
                "progress": 58,
                "status": "editing",
                "current_phase": "Phase 3B: Video Editing",
            },
            goto="editing",
        )

    clip_candidates = state.get("clip_candidates", [])
    transcription = state.get("transcription_result")

    if not clip_candidates:
        logger.warning(f"[Job {job_id}] No clip candidates for hook generation")
        return Command(
            update={
                "progress": 58,
                "status": "editing",
                "current_phase": "Phase 3B: Video Editing",
            },
            goto="editing",
        )

    language = transcription.language if transcription else "en"

    try:
        logger.info(
            f"[Job {job_id}] Starting Phase 3A: Hook Generation "
            f"({len(clip_candidates)} clips, lang={language})"
        )
        await tracker.update_progress(50, "generating_hooks", "Phase 3A: Hook Generation")

        # --- Phase A: Gemini batch call (50% -> 54%) ---
        segments = transcription.segments if transcription else []
        hooks_data = await _generate_hooks_batch(clip_candidates, language, job_id, segments)
        await tracker.update_progress(54, phase="Hook texts generated")

        if not hooks_data:
            logger.warning(f"[Job {job_id}] Gemini hook generation returned empty, skipping hooks")
            return Command(
                update={
                    "progress": 58,
                    "status": "editing",
                    "current_phase": "Phase 3B: Video Editing",
                },
                goto="editing",
            )

        # --- Phase B: TTS synthesis (54% -> 58%) ---
        # Run TTS in thread pool to avoid blocking event loop (CPU-bound ONNX inference)
        logger.info(f"[Job {job_id}] Synthesizing TTS for {len(hooks_data)} hooks...")
        job_clips_dir = os.path.join(settings.CLIPS_DIR, job_id)
        os.makedirs(job_clips_dir, exist_ok=True)

        tts = TTSEngine()
        loop = asyncio.get_event_loop()
        for i, hook in enumerate(hooks_data):
            audio_path = os.path.join(job_clips_dir, f"hook_{hook['clip_index']}.wav")
            result_path = await loop.run_in_executor(
                None, tts.synthesize, hook["hook_text"], language, audio_path
            )
            hook["audio_path"] = result_path
            hook["language"] = language

            progress = 54 + int(((i + 1) / len(hooks_data)) * 4)
            await tracker.update_progress(progress, phase=f"TTS {i + 1}/{len(hooks_data)}")

        # Unload TTS models to free GPU/memory before editing phase
        tts.unload()

        logger.info(f"[Job {job_id}] Phase 3A COMPLETE! Generated {len(hooks_data)} hooks")
        await tracker.update_progress(58, "editing", "Phase 3A complete")

        return Command(
            update={
                "hooks": hooks_data,
                "progress": 58,
                "status": "editing",
                "current_phase": "Phase 3B: Video Editing",
            },
            goto="editing",
        )

    except Exception as e:
        error_msg = f"Hook generation failed: {str(e)}"
        logger.error(f"[Job {job_id}] {error_msg}", exc_info=True)
        # Non-fatal: skip hooks, proceed to editing
        logger.warning(f"[Job {job_id}] Skipping hooks due to error, proceeding to editing")
        return Command(
            update={
                "progress": 58,
                "status": "editing",
                "current_phase": "Phase 3B: Video Editing",
                "errors": [error_msg],
            },
            goto="editing",
        )


def _extract_clip_transcript(segments, clip_start: float, clip_end: float) -> str:
    """Extract transcript text for a clip's time range."""
    lines = []
    for seg in segments:
        if seg.end < clip_start or seg.start > clip_end:
            continue
        speaker = seg.speaker if hasattr(seg, "speaker") and seg.speaker else "Speaker"
        mins = int(seg.start // 60)
        secs = int(seg.start % 60)
        lines.append(f"  [{mins:02d}:{secs:02d}] {speaker}: \"{seg.text}\"")
    return "\n".join(lines) if lines else "  (no transcript)"


async def _generate_hooks_batch(
    clip_candidates: list, language: str, job_id: str, segments: list = None
) -> list[dict]:
    """
    Generate hook text for all clips in a single Gemini call.

    Args:
        clip_candidates: List of clip candidate dicts
        language: Detected language code
        job_id: Job ID for logging
        segments: Transcription segments for extracting clip transcripts

    Returns:
        List of hook dicts: [{"clip_index": 0, "hook_text": "..."}, ...]
    """
    from app.graphs.nodes.analysis_node import get_gemini_llm

    llm = get_gemini_llm()

    # Build clip summaries with transcript snippets
    clip_summaries = []
    for i, clip in enumerate(clip_candidates):
        title = clip.get("title", f"Clip {i + 1}")
        reasoning = clip.get("reasoning", "")
        summary = f"Clip {i}: \"{title}\"\nReasoning: {reasoning}"

        if segments:
            transcript = _extract_clip_transcript(segments, clip["start"], clip["end"])
            summary += f"\nTranscript:\n{transcript}"

        clip_summaries.append(summary)

    clips_text = "\n\n".join(clip_summaries)

    lang_names = {"id": "Indonesian", "en": "English", "zh": "Chinese"}
    lang_name = lang_names.get(language, "English")

    system_prompt = f"""You are a viral content strategist for TikTok, Reels, and Shorts. Your job is to write killer hook lines that stop people from scrolling.

The hook text will be spoken as a voiceover intro (3-5 seconds) BEFORE the clip plays. The viewer must feel compelled to keep watching.

You are given each clip's title, reasoning, and the actual transcript. Read the transcript carefully to understand what's being discussed, then craft a hook that teases the most interesting part WITHOUT spoiling it.

Hook techniques to use:
- **Open loop**: Tease something surprising that happens ("Dia bilang sesuatu yang bikin semua orang kaget...")
- **Bold claim**: Make a strong statement ("Ini alasan kenapa 90% orang salah soal AI")
- **Direct question**: Challenge the viewer ("Kamu yakin kamu paham cara kerja ChatGPT?")
- **Controversy/shock**: Reference a shocking moment ("Coba denger apa yang dia bilang di detik ke-30")
- **Relatable pain**: Connect to viewer's experience ("Pasti kamu pernah ngalamin ini juga")

Rules:
- 1-2 short sentences, MAX 15 words. Shorter is better.
- Language: Write in {lang_name}.
- Be SPECIFIC — reference actual topics/details from the transcript.
- DO NOT be generic. "Kamu harus nonton ini!" is BAD. "Ternyata ChatGPT bisa nolak perintah kita" is GOOD.
- Match the energy: funny clips get playful hooks, serious clips get dramatic hooks.

Output format (JSON array):
```json
[
  {{"clip_index": 0, "hook_text": "..."}},
  {{"clip_index": 1, "hook_text": "..."}}
]
```"""

    user_prompt = f"""Generate hooks for these {len(clip_candidates)} clips:

{clips_text}

Return ONLY valid JSON array."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    import time as _time

    _start = _time.time()
    logger.info(f"[USER PROMPT]: {messages}")
    response = await llm.ainvoke(messages)
    logger.info(f"[AI RESPONSE]: {response.content}")
    _elapsed = _time.time() - _start

    logger.info(f"[Job {job_id}] Gemini hooks response in {_elapsed:.1f}s")

    # Parse response
    return _parse_hooks_response(response.content, len(clip_candidates), job_id)


def _parse_hooks_response(response_text: str, num_clips: int, job_id: str) -> list[dict]:
    """Parse Gemini hook generation response."""
    try:
        if isinstance(response_text, list):
            text_parts = []
            for item in response_text:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            json_text = "".join(text_parts)
        else:
            json_text = str(response_text).strip()

        # Remove markdown code blocks
        if json_text.startswith("```json"):
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif json_text.startswith("```"):
            json_text = json_text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(json_text)

        # Handle both array and object-with-array formats
        if isinstance(parsed, dict):
            parsed = parsed.get("hooks", parsed.get("clips", []))

        hooks = []
        for item in parsed:
            clip_index = item.get("clip_index", -1)
            hook_text = item.get("hook_text", "")

            if 0 <= clip_index < num_clips and hook_text:
                hooks.append({
                    "clip_index": clip_index,
                    "hook_text": hook_text,
                })

        logger.info(f"[Job {job_id}] Parsed {len(hooks)}/{num_clips} hooks from Gemini")
        return hooks

    except json.JSONDecodeError as e:
        logger.error(f"[Job {job_id}] Failed to parse hooks JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"[Job {job_id}] Unexpected error parsing hooks: {e}")
        return []
