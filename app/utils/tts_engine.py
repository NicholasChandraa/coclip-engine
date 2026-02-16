"""
Multi-language TTS engine: Piper (en/zh) + F5-TTS (id).

Wraps Piper TTS and F5-TTS into a single interface
with automatic language-based engine selection.
"""

import os
import re
import wave
import urllib.request
from app.utils.logging import logger
from app.core.config import settings


class TTSEngine:
    """Multi-language TTS: Piper (en/zh) + Coqui (id).

    Models are loaded once and cached for reuse across multiple synthesis calls.
    """

    PIPER_VOICES = {
        "en": "en_US-amy-medium",
        "zh": "zh_CN-huayan-medium",
    }

    def __init__(self):
        self._f5_model = None
        self._f5_vocoder = None
        self._piper_voices = {}  # cache: language -> PiperVoice
        self._f5_ready = False

    def unload(self):
        """Free TTS models from memory/GPU after all synthesis is done."""
        if self._f5_model is not None or self._f5_vocoder is not None:
            try:
                import torch
                # Explicitly delete models
                self._f5_model = None
                self._f5_vocoder = None
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Ensure GPU is cleared
                logger.info("F5-TTS model and vocoder unloaded from GPU")
            except Exception as e:
                logger.error(f"Error unloading F5-TTS: {e}")
                self._f5_model = None
                self._f5_vocoder = None

        if self._piper_voices:
            # Piper is CPU-only ONNX, but still free memory
            self._piper_voices.clear()
            logger.info("Piper TTS voices unloaded")

    def synthesize(self, text: str, language: str, output_path: str) -> str:
        """
        Generate WAV audio file from text.

        Args:
            text: Text to synthesize
            language: Language code (id, en, zh)
            output_path: Path to write the WAV file

        Returns:
            output_path if successful, empty string otherwise
        """
        logger.info(f"TTS START [{language}]")
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            result = ""
            if language == "id":
                result = self._synthesize_id(text, output_path)
            elif language in self.PIPER_VOICES:
                result = self._synthesize_piper(text, language, output_path)
            else:
                logger.warning(f"TTS: unsupported language '{language}', skipping")
                result = ""

            logger.info(f"TTS END [{language}]")
            return result
        except Exception as e:
            logger.error(f"TTS synthesis failed for '{language}': {e}")
            return ""


    def _synthesize_piper(self, text: str, language: str, output_path: str) -> str:
        """Piper TTS for en/zh."""
        from piper import PiperVoice
        from piper.config import SynthesisConfig

        # Cache voice model per language (load once, reuse for all clips)
        if language not in self._piper_voices:
            voice_name = self.PIPER_VOICES[language]
            data_dir = os.path.join(settings.TTS_DATA_DIR, "piper")
            model_path = self._download_piper_voice(voice_name, data_dir)
            self._piper_voices[language] = PiperVoice.load(model_path, use_cuda=False)
            logger.info(f"Piper voice loaded for '{language}': {voice_name}")

        voice = self._piper_voices[language]

        syn_config = SynthesisConfig(length_scale=1.0)
        with wave.open(output_path, "wb") as wav_file:
            first_chunk = True
            for chunk in voice.synthesize(text, syn_config=syn_config):
                if first_chunk:
                    wav_file.setnchannels(chunk.sample_channels)
                    wav_file.setsampwidth(chunk.sample_width)
                    wav_file.setframerate(chunk.sample_rate)
                    first_chunk = False
                wav_file.writeframes(chunk.audio_int16_bytes)

        logger.info(f"Piper TTS [{language}]: {output_path}")
        return output_path

    def _download_piper_voice(self, voice_name: str, data_dir: str) -> str:
        """Download Piper voice model if not already present. Returns model path."""
        model_path = os.path.join(data_dir, f"{voice_name}.onnx")
        config_path = os.path.join(data_dir, f"{voice_name}.onnx.json")

        if os.path.exists(model_path) and os.path.exists(config_path):
            return model_path

        logger.info(f"Downloading Piper voice: {voice_name}...")
        parts = voice_name.split("-")
        lang_code = parts[0][:2]
        locale = parts[0]
        quality = parts[-1]
        voice = "-".join(parts[1:-1])

        base_url = (
            f"https://huggingface.co/rhasspy/piper-voices/resolve/main"
            f"/{lang_code}/{locale}/{voice}/{quality}"
        )

        os.makedirs(data_dir, exist_ok=True)
        for filename in [f"{voice_name}.onnx", f"{voice_name}.onnx.json"]:
            url = f"{base_url}/{filename}"
            dest = os.path.join(data_dir, filename)
            logger.info(f"Downloading {url}...")
            urllib.request.urlretrieve(url, dest)

        return model_path

    def _synthesize_id(self, text: str, output_path: str) -> str:
        """Synthesize Indonesian text using F5-TTS with Reporter voice."""
        try:
            import torch
            import soundfile as sf
            from f5_tts.model import DiT
            from f5_tts.infer.utils_infer import (
                load_model,
                infer_process,
                load_vocoder,
                preprocess_ref_audio_text,
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_dir = os.path.join(settings.TTS_DATA_DIR, "f5-tts-indo")

            # 1. Load Model & Vocoder if not ready
            if self._f5_model is None:
                logger.info("Loading F5-TTS model for Indonesian...")
                model_cfg = dict(
                    dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
                )
                ckpt_path = os.path.join(model_dir, "f5_tts_indo_v2.pt")
                vocab_path = os.path.join(model_dir, "vocab.txt")

                self._f5_model = load_model(
                    model_cls=DiT,
                    model_cfg=model_cfg,
                    ckpt_path=ckpt_path,
                    vocab_file=vocab_path,
                    device=device,
                )
                self._f5_vocoder = load_vocoder(is_local=False, device=device)

            # 2. Reference Settings (Reporter Voice)
            ref_audio_raw = os.path.join(model_dir, "ref_reporter.mp3")
            ref_text_manual = "dikatakan ternyata cek 3 miliar yang diberikan untuk mahar pernikahan ini adalah palsu."

            # 3. Inference
            logger.info(f"F5-TTS [id] synthesizing: {text[:50]}...")
            ref_audio, ref_text = preprocess_ref_audio_text(
                ref_audio_raw, ref_text_manual
            )

            audio, sample_rate, _ = infer_process(
                ref_audio,
                ref_text,
                text,
                self._f5_model,
                self._f5_vocoder,
                nfe_step=48,  # High quality
                cfg_strength=2.0,
                device=device,
            )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio, sample_rate)

            # 4. Mandatory GPU Cleanup removed from here
            # Caller is responsible for calling unload() after the batch is done.
            return output_path


        except Exception as e:
            logger.error(f"F5-TTS [id] synthesis failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            raise


def _normalize_indonesian(text: str) -> str:
    """Normalize Indonesian text for the Coqui vits-tts-id model.

    The model handles standard Indonesian orthography well on its own.
    Only apply targeted word-level fixes for known problem words.
    """
    text = text.lower()

    word_fixes = {
        # Loan words / slang
        "oke": "okey",
        "okay": "okey",
        "video": "fidio",
        "hp": "ha pe",
        # Words with tricky syllable boundaries
        "eliminasi": "e li mi na si",
    }
    for word, fix in word_fixes.items():
        text = re.sub(r"\b" + re.escape(word) + r"\b", fix, text)

    return text
