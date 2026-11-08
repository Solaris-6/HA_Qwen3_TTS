"""Support for the Qwen3-TTS text to speech service."""

import io
import logging
import os
from typing import Any

import numpy as np
import torch

from homeassistant.components.tts import Provider, TtsAudioType
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

from .const import (
    CONF_LANGUAGE,
    CONF_MODEL_PATH,
    CONF_VOICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL_PATH,
    DEFAULT_VOICE,
    DOMAIN,
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)

_model: Any = None
_processor: Any = None
_model_loaded: bool = False


def _load_model(model_path: str) -> tuple[Any, Any]:
    """Carica il modello Qwen3-TTS e il processor."""
    global _model, _processor, _model_loaded

    if _model_loaded:
        return _model, _processor

    try:
        from transformers import Qwen3TTSForConditionalGeneration, AutoProcessor

        _LOGGER.info("Caricamento modello Qwen3-TTS da %s", model_path)

        _processor = AutoProcessor.from_pretrained(model_path)
        _model = Qwen3TTSForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        _model_loaded = True
        _LOGGER.info("Modello Qwen3-TTS caricato con successo")

    except ImportError as err:
        _LOGGER.error("Impossibile importare transformers: %s", err)
        raise
    except Exception as err:
        _LOGGER.error("Errore durante il caricamento del modello Qwen3-TTS: %s", err)
        raise

    return _model, _processor


def _synthesize_audio(
    model: Any,
    processor: Any,
    text: str,
    language: str,
    voice: str,
) -> bytes:
    """Sintetizza audio da testo usando Qwen3-TTS."""
    try:
        inputs = processor(
            text=text,
            language=language,
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8192,
                do_sample=True,
                temperature=0.9,
                top_p=1.0,
                top_k=50,
            )

        audio = processor.decode(outputs[0], skip_special_tokens=False)

        audio_array = np.array(audio, dtype=np.float32)
        audio_array = (audio_array * 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_array.tobytes())

        return buffer.getvalue()

    except Exception as err:
        _LOGGER.error("Errore durante la sintesi audio Qwen3-TTS: %s", err)
        raise


class Qwen3TTSProvider(Provider):
    """Qwen3-TTS speech api provider."""

    def __init__(
        self,
        hass: HomeAssistant,
        model_path: str,
        voice: str,
        language: str,
    ) -> None:
        """Initialize Qwen3-TTS provider for TTS."""
        self.hass = hass
        self.model_path = model_path
        self._attr_name = "Qwen3-TTS"
        self._attr_unique_id = "qwen3_tts"
        self._default_voice = voice
        self._default_language = language

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return SUPPORTED_LANGUAGES

    @property
    def default_language(self) -> str:
        """Return the default language."""
        return self._default_language

    @property
    def default_options(self) -> dict[str, str]:
        """Return dict include default options."""
        return {
            CONF_VOICE: self._default_voice,
            CONF_LANGUAGE: self._default_language,
        }

    @property
    def supported_options(self) -> list[str]:
        """Return a list of supported options."""
        return [CONF_VOICE, CONF_LANGUAGE]

    def get_tts_audio(
        self,
        message: str,
        language: str,
        options: dict[str, Any],
    ) -> TtsAudioType:
        """Request TTS file from Qwen3-TTS."""
        voice = options.get(CONF_VOICE, self._default_voice)
        lang = options.get(CONF_LANGUAGE, self._default_language)

        if lang not in SUPPORTED_LANGUAGES:
            _LOGGER.error("Lingua %s non supportata da Qwen3-TTS", lang)
            return None, None

        _LOGGER.debug("Richiesta TTS per testo: %s", message)

        try:
            model, processor = _load_model(self.model_path)
            audio_data = _synthesize_audio(
                model=model,
                processor=processor,
                text=message,
                language=lang,
                voice=voice,
            )

            _LOGGER.debug("Risposta TTS ricevuta per: %s", message)
            return ("wav", audio_data)

        except Exception:
            _LOGGER.exception("Errore durante la generazione TTS Qwen3-TTS")
            return None, None

    def async_process(self, message: str) -> TtsAudioType:
        """Process a text message and return TTS audio."""
        return self.get_tts_audio(message, "it", {})

    async def async_process_async(self, message: str) -> TtsAudioType:
        """Process a text message asynchronously and return TTS audio."""
        return await asyncio.to_thread(
            self.get_tts_audio, message, "it", {}
        )