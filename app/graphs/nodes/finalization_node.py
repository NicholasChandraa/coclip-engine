"""
Finalization node for LangGraph video processing pipeline.

Implements Phase 4: Finalization (80% → 100%)
Handles DB persistence, Redis caching, cleanup, and completion.
"""

import os
import json
from datetime import datetime, timezone
from langgraph.types import Command
from app.schemas.graph_schemas import VideoProcessingState
from app.utils.progress_tracker import create_progress_tracker
from app.utils.logging import logger
from app.core.database import async_session
from app.models import Job, Clip
from redis import asyncio as aioredis


async def finalization_node(state: VideoProcessingState, redis: aioredis.Redis) -> dict:
    """
    Phase 4: Finalization and cleanup.

    Pipeline steps (80% → 100%):
    1. Save final result to Redis cache (80% → 85%)
    2. Save job + clips to PostgreSQL (85% → 90%)
    3. Generate thumbnails (90% → 95%) [TODO]
    4. Cleanup temp files (95% → 100%)

    Args:
        state: Current LangGraph state
        redis: Async Redis connection for progress tracking

    Returns:
        Updated state dict (final node, no Command needed)
    """
    job_id = state["job_id"]
    video_path = state.get("video_path")
    transcription = state.get("transcription_result")
    clips = state.get("clips", [])
    errors = state.get("errors", [])
    current_status = state.get("status", "finalizing")

    # Initialize progress tracker
    tracker = create_progress_tracker(redis, job_id)

    try:
        # Handle aborted job — skip saving, langsung cleanup
        if current_status == "aborted":
            logger.info(f"🛑 [Job {job_id}] Job aborted — cleaning up temp files")
            if video_path and os.path.exists(video_path):
                try:
                    if "temp" in video_path.lower() or "tmp" in video_path.lower():
                        os.remove(video_path)
                        logger.info(f"🗑️ [Job {job_id}] Cleaned up temp file: {video_path}")
                except Exception as e:
                    logger.warning(f"⚠️ [Job {job_id}] Cleanup failed: {e}")
            await redis.set(f"job:{job_id}:status", "aborted")
            return {
                "progress": 0,
                "status": "aborted",
                "current_phase": "Aborted",
            }

        logger.info(f"🏁 [Job {job_id}] Starting Phase 4: Finalization")
        await tracker.update_progress(80, "finalizing", "Phase 4: Finalization")

        # Step 1: Save final result to Redis cache (80% → 85%)
        if current_status != "failed" and transcription:
            logger.info(
                f"💾 [Job {job_id}] Caching to Redis: "
                f"lang={transcription.language}, duration={transcription.duration:.1f}s, "
                f"segments={transcription.total_segments}, clips={len(clips)}"
            )
            await redis.set(
                f"job:{job_id}:transcription",
                transcription.model_dump_json(),
                ex=3600,
            )

            final_result = {
                "job_id": job_id,
                "language": transcription.language,
                "duration": transcription.duration,
                "total_segments": transcription.total_segments,
                "clips_count": len(clips),
                "clips": clips,
                "status": "completed" if not errors else "completed_with_warnings",
            }

            await redis.set(f"job:{job_id}:result", json.dumps(final_result), ex=3600)
        else:
            logger.warning(f"⚠️ [Job {job_id}] Caching failed result to Redis, errors={len(errors)}")
            final_result = {"job_id": job_id, "status": "failed", "errors": errors}
            await redis.set(f"job:{job_id}:result", json.dumps(final_result), ex=3600)

        await tracker.update_progress(85, phase="Results saved to Redis")

        # Step 2: Save to PostgreSQL (85% → 90%)
        try:
            async with async_session() as session:
                async with session.begin():
                    # Determine final status
                    job_status = (
                        "completed"
                        if (current_status != "failed" and not errors)
                        else "failed"
                    )

                    # Ambil display name: custom/YouTube title > basename video
                    title_raw = await redis.get(f"job:{job_id}:title")
                    if title_raw:
                        video_name = title_raw.decode()
                    elif video_path:
                        video_name = os.path.basename(video_path)
                    else:
                        video_name = "unknown"

                    # Create/update Job record
                    job = Job(
                        id=job_id,
                        user_id=state.get("user_id"),
                        video_name=video_name,
                        source=state.get("source", "upload"),
                        source_url=state.get("source_url"),
                        language=transcription.language if transcription else None,
                        duration=transcription.duration if transcription else None,
                        total_segments=(
                            transcription.total_segments if transcription else None
                        ),
                        status=job_status,
                        error="; ".join(errors) if errors else None,
                        completed_at=datetime.now(timezone.utc),
                    )
                    session.add(job)

                    logger.info(f"💾 [Job {job_id}] DB: saving job (status={job_status})")

                    # Create Clip records
                    for clip_data in clips:
                        clip = Clip(
                            clip_id=clip_data.get("clip_id", ""),
                            job_id=job_id,
                            clip_number=clip_data.get("clip_number", 0),
                            start=clip_data.get("start", 0),
                            end=clip_data.get("end", 0),
                            duration=clip_data.get("duration", 0),
                            title=clip_data.get("title", ""),
                            reasoning=clip_data.get("reasoning"),
                            viral_score=clip_data.get("viral_score"),
                            suggested_caption=clip_data.get("suggested_caption"),
                            hook_text=clip_data.get("hook_text"),
                            transcript_text=clip_data.get("transcript_text"),
                            tags=clip_data.get("tags"),
                            file_path=clip_data.get("file_path", ""),
                            file_size=clip_data.get("file_size"),
                            has_subtitles=clip_data.get("has_subtitles", False),
                            status=clip_data.get("status", "ready"),
                        )
                        session.add(clip)

                logger.info(
                    f"💾 [Job {job_id}] Saved to PostgreSQL: "
                    f"1 job + {len(clips)} clips"
                )
        except Exception as db_error:
            logger.error(f"⚠️ [Job {job_id}] DB save failed (non-fatal): {db_error}")
            # Don't fail the whole job due to DB error — Redis still has the data

        await tracker.update_progress(90, phase="Saved to database")

        # Step 3: Generate thumbnails (90% → 95%) [TODO]
        logger.info(
            f"⚠️ [Job {job_id}] Thumbnail generation not implemented (placeholder)"
        )
        await tracker.update_progress(95, phase="Thumbnail generation skipped")

        # Step 4: Cleanup temp files (95% → 100%)
        if video_path and os.path.exists(video_path):
            try:
                if "temp" in video_path.lower() or "tmp" in video_path.lower():
                    os.remove(video_path)
                    logger.info(f"🗑️ [Job {job_id}] Cleaned up temp file: {video_path}")
                else:
                    logger.info(f"⏭️ [Job {job_id}] Skipping cleanup (not a temp file)")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ [Job {job_id}] Cleanup failed: {cleanup_error}")

        # Final status update
        if current_status == "failed" or errors:
            await tracker.update_progress(100, "failed", "Job completed with errors")
            final_status = "failed"
            logger.error(f"❌ [Job {job_id}] Job completed with errors: {errors}")
        else:
            await tracker.set_completed()
            final_status = "completed"
            logger.info(f"✅ [Job {job_id}] Job completed successfully!")

        return {
            "final_result": final_result,
            "progress": 100,
            "status": final_status,
            "current_phase": "Completed",
        }

    except Exception as e:
        error_msg = f"Finalization failed: {str(e)}"
        logger.error(f"❌ [Job {job_id}] {error_msg}", exc_info=True)
        await tracker.set_error(error_msg)

        return {
            "progress": 100,
            "status": "failed",
            "current_phase": "Failed",
            "errors": errors + [error_msg],
        }
