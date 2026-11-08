"""Qwen3-TTS integration for Home Assistant."""

from __future__ import annotations

import logging

_LOGGER = logging.getLogger(__name__)

async def async_setup(hass, config):
    """Set up the Qwen3-TTS platform."""
    return True

async def async_setup_entry(hass, entry, async_add_entities):
    """Set up Qwen3-TTS from a config entry."""
    model_path = entry.data.get("model_path")
    voice = entry.data.get("voice", "default")
    language = entry.data.get("language", "it")

    from .tts import Qwen3TTSProvider

    async_add_entities([
        Qwen3TTSProvider(hass, model_path, voice, language)
    ])

async def async_unload_entry(hass, entry):
    """Unload a config entry."""
    return True