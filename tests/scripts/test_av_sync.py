import asyncio
import os
import shutil
import logging
from app.graphs.nodes.editing_node import (
    _cut_clip_ffmpeg,
    _create_hook_intro,
    _concat_intro_and_clip,
    _detect_audio_sample_rate,
    _detect_video_resolution,
    _detect_clip_fps
)
from app.utils.tts_engine import TTSEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_test():
    # Setup test env
    test_dir = "tests/test_sync_output"
    os.makedirs(test_dir, exist_ok=True)
    
    # We need a small source video for testing.
    # Looking for one in tests/video...
    source_vid = "tests/video/sample_video_1.mp4" # Assuming this exists or we will need to provide one
    if not os.path.exists(source_vid):
        # Fallback to the first mp4 in videos/ if any
        vids = [f for f in os.listdir("videos") if f.endswith(".mp4")] if os.path.exists("videos") else []
        if not vids:
            logger.error("No source video found for testing.")
            return
        source_vid = os.path.join("videos", vids[0])
        
    logger.info(f"Using source video: {source_vid}")
        
    sample_rate = await _detect_audio_sample_rate(source_vid)
    fps =  await _detect_clip_fps(source_vid)
    
    # 1. Cut a short clip (e.g. 10-15s)
    clip_path = os.path.join(test_dir, "test_clip.mp4")
    logger.info("Cutting clip...")
    result = await _cut_clip_ffmpeg(
        input_path=source_vid,
        output_path=clip_path,
        start=10.0,
        end=20.0,
        job_id="test",
        clip_index=1,
        subtitle_path=None,
        crop_filter=None,
        sample_rate=sample_rate
    )
    
    if not result.get("success"):
        logger.error("Clip cut failed")
        return
        
    # 2. Generate a Hook Audio via TTS
    tts = TTSEngine()
    audio_path = os.path.join(test_dir, "hook_audio.wav")
    logger.info("Generating TTS...")
    tts_res = await asyncio.to_thread(
        tts.synthesize, "Dikatakan ternyata cek tiga miliar ini palsu.", "id", audio_path
    )
    
    # 3. Create Hook Intro
    logger.info("Creating Hook Intro...")
    hook = {"hook_text": "CEK 3 MILIAR INI PALSU!", "audio_path": audio_path}
    intro_path = await _create_hook_intro(
        hook=hook,
        main_clip_path=clip_path,
        output_dir=test_dir,
        clip_index=1,
        job_id="test",
        target_width=1080,
        target_height=1920,
        clip_fps=fps,
        sample_rate=sample_rate
    )
    
    if not intro_path:
        logger.error("Intro creation failed")
        return
        
    # 4. Concat
    logger.info("Concatenating...")
    final_path = os.path.join(test_dir, "final_sync_test.mp4")
    final_res = await _concat_intro_and_clip(
        intro_path=intro_path,
        main_clip_path=clip_path,
        output_path=final_path,
        job_id="test"
    )
    
    if final_res:
        logger.info(f"Success! Check output at: {final_path}")
    else:
        logger.error("Concat failed")

if __name__ == "__main__":
    asyncio.run(run_test())
