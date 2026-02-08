import sys
import os

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

from faster_whisper import WhisperModel
from app.core.config import settings
from app.utils.logging import logger
import time
from typing import Optional, TypedDict, List
from threading import Lock


class TranscriptionSegment(TypedDict):
    """
    Struktur data untuk satu segment hasil transcription.
    Setiap segment berisi timing dan text dari audio.
    """
    start: float
    end: float
    text: str


class TranscriptionResult(TypedDict):
    """
    Struktur data lengkap hasil transcription.
    Berisi metadata (language, duration) dan list of segments.
    """
    language: str
    duration: float
    segments: List[TranscriptionSegment]


class WhisperTranscriber:
    """
    Singleton class untuk Whisper model transcription.

    Pattern singleton digunakan karena:
    1. Whisper model besar (ratusan MB - beberapa GB di memory/VRAM)
    2. Loading model lambat (2-10 detik tergantung device)
    3. Hanya perlu 1 instance untuk handle multiple requests

    Thread-safe menggunakan Lock untuk prevent race condition saat
    multiple threads mencoba load model secara bersamaan.
    """
    _instance: Optional['WhisperTranscriber'] = None
    _model: Optional[WhisperModel] = None
    _lock: Lock = Lock()  # Thread safety untuk singleton pattern

    def __new__(cls):
        """
        Override __new__ untuk implement singleton pattern.
        Memastikan hanya ada 1 instance WhisperTranscriber di seluruh aplikasi.
        """
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern untuk thread safety
                if cls._instance is None:
                    cls._instance = super(WhisperTranscriber, cls).__new__(cls)
        return cls._instance

    def load_model(self) -> None:
        """
        Load Whisper model ke memory/GPU.

        Dipanggil saat:
        1. Startup aplikasi (preload untuk response cepat)
        2. First transcription request (lazy loading)

        Model di-load dengan parameter:
        - device: CPU/CUDA/MPS tergantung hardware
        - compute_type: Precision (float32/float16/int8) untuk balance accuracy vs speed
        """
        if self._model is None:
            with self._lock:
                # Double-check untuk prevent multiple model loads
                if self._model is None:
                    logger.info(f"📥 Loading Whisper Model: {settings.WHISPER_MODEL} on {settings.WHISPER_DEVICE}...")
                    start_time = time.time()

                    try:
                        # Initialize Whisper model dengan config dari settings
                        self._model = WhisperModel(
                            settings.WHISPER_MODEL,  # Model size: tiny/base/small/medium/large-v3
                            device=settings.WHISPER_DEVICE,  # cpu/cuda/mps
                            compute_type=settings.WHISPER_COMPUTE_TYPE  # float32/float16/int8
                        )
                        duration = time.time() - start_time
                        logger.info(f"✅ Model loaded successfully in {duration:.2f}s")
                    except Exception as e:
                        logger.error(f"❌ Failed to load model: {e}")
                        raise e

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe audio file menjadi text dengan timestamps.

        Args:
            audio_path: Path ke file audio/video yang akan di-transcribe

        Returns:
            TranscriptionResult dengan language, duration, dan segments

        Proses:
        1. Load model kalau belum ready (lazy loading)
        2. Panggil faster-whisper transcribe dengan:
           - beam_size=5: Beam search untuk akurasi lebih baik (trade-off: lebih lambat)
           - word_timestamps=True: Generate timestamp per-word (penting untuk auto-clipping!)
        3. Convert generator segments ke list (blocking operation - perlu refactor)

        PERHATIAN: Ini adalah BLOCKING operation yang CPU/GPU intensive.
        Untuk production, wrap dengan asyncio.to_thread() atau background task.
        """
        # Auto-load model kalau belum siap
        if self._model is None:
            self.load_model()

        # Type guard: Setelah load_model(), _model pasti tidak None
        # Kalau load_model() gagal, akan raise exception
        assert self._model is not None, "Model should be loaded at this point"

        logger.info(f"🎙️ Transcribing: {audio_path}")
        start_time = time.time()

        # Panggil faster-whisper transcribe
        # Return value adalah tuple (segments_generator, info_object)
        segments, info = self._model.transcribe(
            audio_path,
            beam_size=5,  # Beam search width: lebih tinggi = lebih akurat tapi lambat
            word_timestamps=True  # Generate timestamp per word (bukan per segment)
                                  # CRITICAL untuk fitur auto-clipping berdasarkan kata kunci!
        )

        # Convert generator to list menggunakan list comprehension
        # Untuk auto-clipper: full transcript HARUS di-collect karena LLM perlu analyze
        # keseluruhan context untuk decide clipping points
        # List comprehension ~20-30% lebih cepat dari for-loop + append
        result_segments: List[TranscriptionSegment] = [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip()
            }
            for seg in segments
        ]

        duration = time.time() - start_time
        logger.info(
            f"✅ Transcription done in {duration:.2f}s. "
            f"Detected language: {info.language}, "
            f"Segments: {len(result_segments)}"
        )

        return {
            "language": info.language,  # Auto-detected language (en/id/etc)
            "duration": info.duration,  # Total audio duration in seconds
            "segments": result_segments  # List of transcribed segments dengan timestamps
        }


# Global singleton instance
# Import dan pakai instance ini di seluruh aplikasi
transcriber = WhisperTranscriber()
