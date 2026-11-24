"""Qwen3-TTS integration for Home Assistant."""

from __future__ import annotations

from homeassistant.helpers import config_validation as cv

from .const import DOMAIN

PLATFORMS = ["tts"]

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


async def async_setup(_hass, _config):
    """Set up the Qwen3-TTS platform."""
    return True


async def async_setup_entry(hass, config_entry, _async_add_entities=None):
    """Set up Qwen3-TTS from a config entry."""
    await hass.config_entries.async_forward_entry_setups(config_entry, PLATFORMS)
    return None


async def async_unload_entry(hass, config_entry):
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(config_entry, PLATFORMS)
