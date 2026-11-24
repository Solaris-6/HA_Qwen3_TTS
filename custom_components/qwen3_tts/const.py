"""Constants for the Qwen3-TTS text to speech service."""

from __future__ import annotations

import os
from typing import Final

DOMAIN: Final = "qwen3_tts"

CONF_MODEL_PATH: Final = "model_path"
CONF_REF_AUDIO: Final = "ref_audio"
CONF_REF_TEXT: Final = "ref_text"
CONF_VOICE: Final = "voice"
CONF_LANGUAGE: Final = "language"

DEFAULT_MODEL_PATH: Final = "/config/models/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_VOICE: Final = "default"
DEFAULT_LANGUAGE: Final = "it"

# Reference voice shipped with the integration. Used when the user does not
# configure their own reference audio.
DEFAULT_REF_AUDIO: Final = os.path.join(
    os.path.dirname(__file__), "audio", "claudia.wav"
)

SUPPORTED_LANGUAGES: Final[list[str]] = [
    "it",
    "en",
    "zh",
    "de",
    "pt",
    "es",
    "ja",
    "ko",
    "fr",
    "ru",
]

# Qwen3-TTS expects the full English language name (see the model card), not
# the ISO 639-1 code used by Home Assistant.
LANGUAGE_NAMES: Final[dict[str, str]] = {
    "it": "Italian",
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "pt": "Portuguese",
    "es": "Spanish",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "ru": "Russian",
}

SUPPORTED_OPTIONS: Final[list[str]] = [CONF_VOICE]

# Native sample rate of the Qwen3-TTS models.
SAMPLE_RATE: Final = 24000

# Config flow step IDs
STEP_USER: Final = "user"
