"""Qwen3-TTS integration for Home Assistant."""

from __future__ import annotations

from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv

from . import deps
from .const import DOMAIN

PLATFORMS = ["tts"]

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


async def async_setup(_hass, _config):
    """Set up the Qwen3-TTS platform."""
    return True


async def async_setup_entry(hass, config_entry, _async_add_entities=None):
    """Set up Qwen3-TTS from a config entry."""
    # Le dipendenze (qwen-tts + PyTorch e compagni) non stanno più nel
    # manifest: Home Assistant le installerebbe mentre carica il config flow e
    # un errore in quella fase fa esplodere l'interfaccia con un «500».
    # Qui l'installazione viene invece gestita da deps.ensure_installed, che
    # logga l'output completo e lascia la scheda in stato «Setup retry».
    if not deps.is_installed():
        error = await hass.async_add_executor_job(deps.ensure_installed)
        if error is not None:
            raise ConfigEntryNotReady(
                "Dipendenze di Qwen3-TTS (qwen-tts==0.1.1) non installate: "
                f"{error}. Controlla il log per l'output completo dell'installer."
            )

    await hass.config_entries.async_forward_entry_setups(config_entry, PLATFORMS)
    return None


async def async_unload_entry(hass, config_entry):
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(config_entry, PLATFORMS)
