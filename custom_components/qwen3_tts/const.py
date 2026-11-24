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

# Where the model runs. Home Assistant itself cannot install PyTorch on the
# official images (they are built on Alpine/musl and PyTorch only publishes
# glibc wheels), so the model usually runs on a server in the LAN.
CONF_MODE: Final = "mode"
CONF_SERVER_URL: Final = "server_url"
MODE_SERVER: Final = "server"
MODE_LOCAL: Final = "local"

DEFAULT_MODEL_PATH: Final = "/config/models/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_SERVER_URL: Final = ""
DEFAULT_VOICE: Final = "default"
DEFAULT_LANGUAGE: Final = "it"

# The model can be slow on CPU-only hardware (a Raspberry Pi needs tens of
# seconds per sentence), so the HTTP call gets a generous timeout.
SERVER_TIMEOUT: Final = 600.0
HEALTH_TIMEOUT: Final = 10.0

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
