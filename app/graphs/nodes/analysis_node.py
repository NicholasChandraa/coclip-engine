"""
Content analysis node for LangGraph video processing pipeline.

Implements Phase 2: Gemini content analysis (25% → 50%)
Uses LangChain Google GenAI for future-proof LLM integration.
"""

import json
from typing import Literal
from langgraph.types import Command
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from app.schemas.graph_schemas import VideoProcessingState
from app.utils.progress_tracker import create_progress_tracker
from app.utils.logging import logger
from app.core.config import settings
from redis import asyncio as aioredis


# Initialize Gemini LLM (singleton pattern for reuse)
_gemini_llm = None


def get_gemini_llm():
    """Get or create Gemini LLM instance."""
    global _gemini_llm
    if _gemini_llm is None:
        _gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",  # Latest Gemini model
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.4,  # Balanced creativity
            convert_system_message_to_human=True,  # Gemini compatibility
        )
    return _gemini_llm


async def analysis_node(
    state: VideoProcessingState, redis: aioredis.Redis
) -> Command[Literal["editing", "finalization"]]:
    """
    Phase 2: Content analysis using Gemini LLM.

    Pipeline steps (25% → 50%):
    1. Prepare transcript for Gemini (25% → 28%)
    2. Feed transcript to Gemini for viral clip detection (28% → 42%)
    3. Parse Gemini response & format clip candidates (42% → 50%)

    Args:
        state: Current LangGraph state
        redis: Async Redis connection for progress tracking

    Returns:
        Command with updated state and routing decision
    """
    job_id = state["job_id"]
    transcription = state.get("transcription_result")

    # Initialize progress tracker
    tracker = create_progress_tracker(redis, job_id)

    try:
        logger.info(f"🧠 [Job {job_id}] Starting Phase 2: Gemini Content Analysis")
        await tracker.update_progress(25, "analyzing", "Phase 2: Content Analysis")

        # Validate transcription exists
        if not transcription or not transcription.segments:
            logger.warning(f"⚠️ [Job {job_id}] No transcription available for analysis")
            return Command(
                update={
                    "analysis_result": {"clip_candidates": []},
                    "clip_candidates": [],
                    "progress": 80,
                    "status": "finalizing",
                    "current_phase": "Phase 4: Finalization",
                    "errors": ["No transcription available for analysis"],
                },
                goto="finalization",
            )

        # Step 1: Prepare transcript (25% → 28%)
        logger.info(f"📝 [Job {job_id}] Preparing transcript for Gemini...")
        transcript_text = _format_transcript_for_analysis(transcription)
        await tracker.update_progress(28, phase="Transcript prepared")

        # Step 2: Call Gemini for viral clip detection (28% → 42%)
        logger.info(f"🤖 [Job {job_id}] Sending transcript to Gemini for analysis...")
        llm = get_gemini_llm()

        # Create prompt for viral clip detection
        system_prompt = _create_analysis_system_prompt()
        user_prompt = _create_analysis_user_prompt(
            transcript_text, transcription.duration
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        logger.info(f"USER PROMPT DIKIRIM KE AI: {user_prompt}")

        # Invoke Gemini (async)
        response = await llm.ainvoke(messages)

        logger.info(f"[RESPONSE AI]: {response.content}")
        await tracker.update_progress(42, phase="Gemini analysis complete")

        logger.info(f"✅ [Job {job_id}] Received Gemini response")

        # Step 3: Parse response (42% → 50%)
        logger.info(f"📊 [Job {job_id}] Parsing Gemini response...")
        clip_candidates = _parse_gemini_response(response.content, job_id)
        await tracker.update_progress(50, "editing", "Phase 2 complete")

        logger.info(
            f"✅ [Job {job_id}] Phase 2 COMPLETE! "
            f"Found {len(clip_candidates)} clip candidates"
        )

        # Check if we have clips to edit
        if clip_candidates and len(clip_candidates) > 0:
            # Route to editing
            return Command(
                update={
                    "analysis_result": {
                        "clip_candidates": clip_candidates,
                        "total_candidates": len(clip_candidates),
                    },
                    "clip_candidates": clip_candidates,
                    "progress": 50,
                    "status": "editing",
                    "current_phase": "Phase 3: Video Editing",
                },
                goto="editing",
            )
        else:
            # No clips found, skip to finalization
            # This is a valid outcome, not an error!
            logger.warning(
                f"⚠️ [Job {job_id}] No viral-worthy clips identified by Gemini"
            )
            return Command(
                update={
                    "analysis_result": {
                        "clip_candidates": [],
                        "message": "No viral-worthy clips found in this video",
                    },
                    "clip_candidates": [],
                    "progress": 80,
                    "status": "finalizing",
                    "current_phase": "Phase 4: Finalization",
                },
                goto="finalization",
            )

    except Exception as e:
        error_msg = f"Content analysis failed: {str(e)}"
        logger.error(f"❌ [Job {job_id}] {error_msg}", exc_info=True)
        await tracker.set_error(error_msg)

        return Command(
            update={
                "progress": 25,
                "status": "failed",
                "current_phase": "Failed",
                "errors": [error_msg],
            },
            goto="finalization",
        )


def _format_transcript_for_analysis(transcription) -> str:
    """
    Format transcription segments for Gemini analysis.

    Args:
        transcription: TranscriptionResultDetailed object

    Returns:
        Formatted transcript string with timestamps
    """
    formatted_segments = []

    for i, segment in enumerate(transcription.segments):
        # Format: [00:15 - 00:42] Speaker A: "Text content here"
        start_time = _seconds_to_timestamp(segment.start)
        end_time = _seconds_to_timestamp(segment.end)
        speaker = (
            segment.speaker
            if hasattr(segment, "speaker") and segment.speaker
            else "Speaker"
        )

        formatted_segments.append(
            f'[{start_time} - {end_time}] {speaker}: "{segment.text}"'
        )

    return "\n".join(formatted_segments)


def _seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def _create_analysis_system_prompt() -> str:
    """Create system prompt for Gemini viral clip detection."""
    return """You are an expert content analyst specializing in identifying the best video clips for social media platforms (TikTok, YouTube Shorts, Instagram Reels).

Your task is to analyze video transcripts and identify segments that are ENGAGING, COMPLETE, and make sense as STANDALONE content.

**CRITICAL RULES:**
- Every clip MUST tell a complete story/topic. A viewer who has NEVER seen the full video must fully understand what is being discussed.
- ALWAYS include the setup/introduction of a topic BEFORE the punchline or key moment. Don't start a clip in the middle of a thought.
- If a topic takes 2 minutes to fully explain, the clip should be 2 minutes. DO NOT cut short just to keep it under 60 seconds.
- It's much better to have a longer clip with full context than a short clip that confuses viewers.

**Clip Quality Criteria:**
1. **Context Completeness** (0-10): Can a new viewer fully understand this clip without watching the full video? Does it include the setup, explanation, and conclusion of the topic?
2. **Hook Factor** (0-10): Does the opening grab attention?
3. **Emotional Impact** (0-10): Funny, shocking, inspirational, relatable?
4. **Shareability** (0-10): Would viewers share this with friends?
5. **Standalone Value** (0-10): Does this clip deliver value on its own?

**Output Format (JSON):**
```json
{
  "clips": [
    {
      "start": 15.5,
      "end": 105.2,
      "title": "Catchy Title (max 60 chars)",
      "reasoning": "Why this clip works as standalone content",
      "context_completeness": 9,
      "hook_factor": 8,
      "emotional_impact": 8,
      "shareability": 9,
      "standalone_value": 9,
      "viral_score": 8.6,
      "suggested_caption": "Engaging caption for social media"
    }
  ]
}
```

**Instructions:**
- Identify as many good clips as possible (aim for 1 clip per 3-5 minutes of video)
- Each clip should be 30-180 seconds (longer is fine if the topic needs it)
- NEVER cut a topic in the middle. Always include the full discussion from start to finish.
- Start each clip a few seconds BEFORE the topic begins (to include natural transitions)
- End each clip a few seconds AFTER the topic concludes (to avoid abrupt cuts)
- Prioritize clips with viral_score >= 7.0
- Avoid overlapping clips
- If no good content found, return empty clips array"""


def _create_analysis_user_prompt(transcript: str, duration: float) -> str:
    """Create user prompt with transcript for analysis."""
    minutes = duration / 60
    min_clips = max(3, int(minutes / 5))
    max_clips = max(5, int(minutes / 3))

    return f"""Analyze this video transcript and identify the best clips for social media:

**Video Duration:** {duration:.1f} seconds ({minutes:.1f} minutes)
**Target:** Find {min_clips}-{max_clips} clips (more is better, as long as quality is good)
**Transcript:**
{transcript}

IMPORTANT:
- Each clip MUST contain a COMPLETE topic/story from beginning to end.
- A viewer who has never seen the full video must understand the full context.
- Clips can be 30 seconds to 3 minutes — use whatever length the topic needs.
- DO NOT cut topics short. Include the full discussion.
- For a {minutes:.0f}-minute video, there should be MANY good clips. Don't be too selective.

Return ONLY valid JSON."""


def _parse_gemini_response(response_text: str, job_id: str) -> list[dict]:
    """
    Parse Gemini JSON response into clip candidates.

    Args:
        response_text: Raw Gemini response
        job_id: Job ID for logging

    Returns:
        List of clip candidate dicts
    """
    try:
        # Handle LangChain response format - might be list of content blocks
        if isinstance(response_text, list):
            # Extract text from list of content blocks
            text_parts = []
            for item in response_text:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            json_text = "".join(text_parts)
        else:
            # Already a string
            json_text = str(response_text).strip()

        # Remove markdown code blocks if present
        if json_text.startswith("```json"):
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif json_text.startswith("```"):
            json_text = json_text.split("```")[1].split("```")[0].strip()

        # Parse JSON
        parsed = json.loads(json_text)
        clips = parsed.get("clips", [])

        # Validate and filter clips
        valid_clips = []
        for clip in clips:
            if _validate_clip_candidate(clip):
                valid_clips.append(clip)
            else:
                logger.warning(f"⚠️ [Job {job_id}] Invalid clip candidate: {clip}")

        return valid_clips

    except json.JSONDecodeError as e:
        logger.error(f"❌ [Job {job_id}] Failed to parse Gemini JSON: {e}")
        logger.debug(f"Raw response: {response_text}")
        return []
    except Exception as e:
        logger.error(f"❌ [Job {job_id}] Unexpected error parsing response: {e}")
        logger.debug(f"Response type: {type(response_text)}, content: {response_text}")
        return []


def _validate_clip_candidate(clip: dict) -> bool:
    """Validate clip candidate has required fields."""
    required_fields = ["start", "end", "title", "viral_score"]

    # Check all required fields exist
    if not all(field in clip for field in required_fields):
        return False

    # Validate types and ranges
    if not isinstance(clip["start"], (int, float)) or clip["start"] < 0:
        return False
    if not isinstance(clip["end"], (int, float)) or clip["end"] <= clip["start"]:
        return False
    if not isinstance(clip["viral_score"], (int, float)) or clip["viral_score"] < 0:
        return False

    return True
