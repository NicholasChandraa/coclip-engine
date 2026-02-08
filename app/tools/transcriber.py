import sys
import os
import gc

# --- PyTorch 2.6+ Workaround ---
# PyTorch 2.6 mengubah default weights_only=True yang memblokir load_model pyannote
# Patch torch.load agar default weights_only=False
import torch
import torch.serialization

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # FORCE weights_only=False
    kwargs['weights_only'] = False

    # Add safe globals for omegaconf (dipakai pyannote)
    try:
        from omegaconf import DictConfig, ListConfig
        torch.serialization.add_safe_globals([DictConfig, ListConfig])
    except:
        pass

    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
# -------------------------------

# Fix untuk ctranslate2 ROCm DLL path error di Windows
if sys.platform == "win32":
    # Suppress ROCm DLL path errors on Windows
    # ctranslate2 nyari ROCm SDK yang ga ada kalau pakai CUDA/CPU
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Patch os.add_dll_directory untuk ignore FileNotFoundError
    _original_add_dll_directory = os.add_dll_directory
    def _patched_add_dll_directory(path):
        try:
            return _original_add_dll_directory(path)
        except (FileNotFoundError, OSError):
            # Ignore ROCm path not found, lanjut pakai CUDA/CPU
            pass
    os.add_dll_directory = _patched_add_dll_directory

# --- Patch huggingface_hub deprecated 'use_auth_token' ---
# HARUS sebelum import whisperx/pyannote agar patch kena
# Newer huggingface_hub hapus 'use_auth_token', ganti 'token'
# Tapi whisperx & pyannote masih pakai 'use_auth_token'
import huggingface_hub
_original_hf_hub_download = huggingface_hub.hf_hub_download
def _patched_hf_hub_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    return _original_hf_hub_download(*args, **kwargs)
huggingface_hub.hf_hub_download = _patched_hf_hub_download
# ----------------------------------------------------------

import whisperx
from app.core.config import settings
from app.utils.logging import logger
import time
from typing import Optional, List, Any
from threading import Lock
from app.schemas.transcription import TranscriptionResultDetailed, TranscriptionSegment





class WhisperXTranscriber:
    """
    Singleton class untuk WhisperX transcription.

    WhisperX advantages over faster-whisper:
    1. More accurate word-level timestamps (phoneme-based alignment)
    2. Speaker diarization (detect who is speaking when)
    3. Batch processing support

    Perfect untuk auto-clipper karena:
    - Presisi timestamp = clip boundaries lebih bagus
    - Speaker detection = bisa highlight viral moments dari speaker tertentu

    Thread-safe dengan Lock pattern.
    """
    _instance: Optional['WhisperXTranscriber'] = None
    _model: Optional[Any] = None
    _align_model: Optional[Any] = None
    _align_metadata: Optional[dict] = None
    _diarize_model: Optional[Any] = None
    _last_language: Optional[str] = None
    _lock: Lock = Lock()

    # Config
    _device: str = settings.WHISPER_DEVICE
    _compute_type: str = settings.WHISPER_COMPUTE_TYPE
    _batch_size: int = 16  # Adjust based on GPU memory

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(WhisperXTranscriber, cls).__new__(cls)
        return cls._instance

    def load_model(self) -> None:
        """
        Load WhisperX model untuk transcription.

        Model di-load sekali dan di-reuse untuk semua requests.
        Alignment model dan diarization model di-load on-demand.
        """
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info(f"📥 Loading WhisperX Model: {settings.WHISPER_MODEL} on {self._device}...")
                    start_time = time.time()

                    try:
                        self._model = whisperx.load_model(
                            settings.WHISPER_MODEL,
                            self._device,
                            compute_type=self._compute_type
                        )
                        duration = time.time() - start_time
                        logger.info(f"✅ WhisperX model loaded in {duration:.2f}s")
                    except Exception as e:
                        import traceback
                        logger.error(f"❌ Failed to load WhisperX model: {e}")
                        logger.error(traceback.format_exc())
                        raise e

    def _load_align_model(self, language_code: str) -> None:
        """
        Load alignment model untuk language tertentu.

        Alignment model dipakai untuk:
        - Word-level timestamps yang lebih presisi
        - Phoneme-based alignment (lebih akurat dari Whisper native)

        Model di-cache per language untuk efficiency.
        """
        # Kalau language sama dengan sebelumnya, skip reload
        if self._align_model is not None and self._last_language == language_code:
            return

        with self._lock:
            logger.info(f"📥 Loading alignment model for language: {language_code}")
            start_time = time.time()

            try:
                # Cleanup previous alignment model
                if self._align_model is not None:
                    del self._align_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Handle Indonesian-specific model
                align_model_name = None
                if language_code == "id":
                    align_model_name = "indonesian-nlp/wav2vec2-large-xlsr-indonesian"
                    logger.info(f"🇮🇩 Using Indonesian alignment model: {align_model_name}")

                # Load alignment model
                if align_model_name:
                    self._align_model, self._align_metadata = whisperx.load_align_model(
                        language_code=language_code,
                        device=self._device,
                        model_name=align_model_name
                    )
                else:
                    self._align_model, self._align_metadata = whisperx.load_align_model(
                        language_code=language_code,
                        device=self._device
                    )

                self._last_language = language_code
                duration = time.time() - start_time
                logger.info(f"✅ Alignment model loaded in {duration:.2f}s")

            except Exception as e:
                logger.error(f"❌ Failed to load alignment model: {e}")
                raise e

    def _load_diarize_model(self) -> None:
        """
        Load speaker diarization model via WhisperX built-in.

        Requires HF_TOKEN di .env untuk download pyannote model.
        """
        if self._diarize_model is None:
            with self._lock:
                if self._diarize_model is None:
                    hf_token = os.getenv("HF_TOKEN")

                    if not hf_token:
                        logger.warning("⚠️ HF_TOKEN not found. Diarization disabled.")
                        return

                    logger.info("📥 Loading diarization model...")
                    start_time = time.time()

                    try:
                        # WhisperX built-in diarization
                        # huggingface_hub patch sudah di-apply di top-level
                        from whisperx.diarize import DiarizationPipeline
                        self._diarize_model = DiarizationPipeline(
                            use_auth_token=hf_token,
                            device=torch.device(self._device)
                        )

                        duration = time.time() - start_time
                        logger.info(f"✅ Diarization model loaded in {duration:.2f}s")
                    except Exception as e:
                        logger.error(f"❌ Failed to load diarization model: {e}")
                        logger.info("📌 Continuing without diarization...")

    def load_audio(self, audio_path: str):
        """Load audio file into numpy array."""
        logger.info(f"🎙️ Loading audio: {audio_path}")
        return whisperx.load_audio(audio_path)

    def step_transcribe(self, audio) -> dict:
        """
        Step 1: Transcription - Convert audio to text.

        Args:
            audio: Numpy array dari load_audio()

        Returns:
            Raw transcription result (segments + language)
        """
        if self._model is None:
            self.load_model()

        assert self._model is not None, "Model should be loaded"

        logger.info("📝 Step 1/3: Transcribing audio...")
        start = time.time()

        result = self._model.transcribe(audio, batch_size=self._batch_size)

        duration = time.time() - start
        logger.info(f"✅ Transcription done in {duration:.2f}s. Language: {result['language']}")

        return result

    def step_align(self, segments: list, audio, language: str) -> dict:
        """
        Step 2: Alignment - Get precise word-level timestamps.

        Args:
            segments: Raw segments dari step_transcribe()
            audio: Numpy array dari load_audio()
            language: Detected language code

        Returns:
            Aligned result dengan word-level timestamps
        """
        logger.info("🎯 Step 2/3: Aligning for word-level timestamps...")
        start = time.time()

        self._load_align_model(language)

        result = whisperx.align(
            segments,
            self._align_model,
            self._align_metadata,
            audio,
            self._device,
            return_char_alignments=False
        )

        duration = time.time() - start
        logger.info(f"✅ Alignment done in {duration:.2f}s")

        return result

    def step_diarize(self, audio, result: dict) -> dict:
        """
        Step 3: Diarization - Detect speakers.

        Args:
            audio: Numpy array dari load_audio()
            result: Aligned result dari step_align()

        Returns:
            Result dengan speaker labels assigned
        """
        logger.info("👥 Step 3/3: Detecting speakers...")
        start = time.time()

        self._load_diarize_model()

        if self._diarize_model is None:
            logger.info("⏭️ Diarization skipped (no HF_TOKEN)")
            return result

        try:
            diarize_segments = self._diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

            speakers = set(seg.get("speaker", "Unknown") for seg in result["segments"])
            duration = time.time() - start
            logger.info(f"✅ Diarization done in {duration:.2f}s. Speakers: {speakers}")
        except Exception as e:
            logger.warning(f"⚠️ Diarization failed: {e}")

        return result

    def format_result(self, result: dict, language: str) -> TranscriptionResultDetailed:
        """
        Format raw WhisperX result ke TranscriptionResultDetailed.

        Args:
            result: Raw result dari step_align() atau step_diarize()
            language: Detected language code

        Returns:
            Formatted TranscriptionResultDetailed (Pydantic Model)
        """
        segments = [
            TranscriptionSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                words=seg.get("words"),
                speaker=seg.get("speaker")
            )
            for seg in result["segments"]
        ]

        duration = segments[-1].end if segments else 0.0

        return TranscriptionResultDetailed(
            language=language,
            duration=duration,
            total_segments=len(segments),
            segments=segments
        )


# Global singleton instance
# Import dan pakai instance ini di seluruh aplikasi
transcriber = WhisperXTranscriber()
