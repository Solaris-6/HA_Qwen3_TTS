"""Config flow for Qwen3-TTS integration."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.config_entries import ConfigFlow

from .const import (
    CONF_LANGUAGE,
    CONF_MODEL_PATH,
    CONF_REF_AUDIO,
    CONF_REF_TEXT,
    CONF_VOICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL_PATH,
    DEFAULT_REF_AUDIO,
    DEFAULT_VOICE,
    DOMAIN,
    SUPPORTED_LANGUAGES,
)

if TYPE_CHECKING:
    # Imported only for static type checking: the import must never run at
    # runtime so the config flow keeps loading on every Home Assistant
    # version that still exports FlowResult.
    from homeassistant.data_entry_flow import FlowResult

_LOGGER = logging.getLogger(__name__)

_MODEL_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    os.path.join("speech_tokenizer", "config.json"),
)


def _folder_has_weights(folder: str) -> bool:
    """Return True when the folder contains a model weight file."""
    try:
        names = os.listdir(folder)
    except OSError:
        return False
    return any(
        name.startswith(("model", "pytorch_model")) and name.endswith((".safetensors", ".bin"))
        for name in names
    )


def _model_error(model_path: str) -> str | None:
    """Return the translation key describing why the model folder is unusable."""
    if not model_path or not os.path.isdir(model_path):
        return "model_path_not_found"

    for name in _MODEL_REQUIRED_FILES:
        if not os.path.isfile(os.path.join(model_path, name)):
            return "model_incomplete"

    if not _folder_has_weights(model_path):
        return "model_incomplete"
    if not _folder_has_weights(os.path.join(model_path, "speech_tokenizer")):
        return "model_incomplete"

    return None


def _ref_audio_error(ref_audio: str) -> str | None:
    """Return the translation key describing why the reference audio is unusable."""
    if not ref_audio or os.path.isfile(ref_audio):
        return None
    return "ref_audio_not_found"


def _schema(defaults: dict[str, Any]) -> vol.Schema:
    """Build the shared configuration schema."""
    return vol.Schema(
        {
            vol.Optional(
                CONF_MODEL_PATH,
                default=defaults.get(CONF_MODEL_PATH, DEFAULT_MODEL_PATH),
            ): str,
            vol.Optional(
                CONF_REF_AUDIO,
                default=defaults.get(CONF_REF_AUDIO, DEFAULT_REF_AUDIO),
            ): str,
            vol.Optional(
                CONF_REF_TEXT,
                default=defaults.get(CONF_REF_TEXT, ""),
            ): str,
            vol.Optional(
                CONF_LANGUAGE,
                default=defaults.get(CONF_LANGUAGE, DEFAULT_LANGUAGE),
            ): vol.In(SUPPORTED_LANGUAGES),
        }
    )


class Qwen3TTSConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Qwen3-TTS."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            model_path = user_input.get(CONF_MODEL_PATH, DEFAULT_MODEL_PATH)
            ref_audio = user_input.get(CONF_REF_AUDIO, DEFAULT_REF_AUDIO)

            error = await self.hass.async_add_executor_job(_model_error, model_path)
            if error:
                errors["base"] = error
            elif (
                error := await self.hass.async_add_executor_job(
                    _ref_audio_error, ref_audio
                )
            ) is not None:
                errors[CONF_REF_AUDIO] = error
            else:
                # The entity has a fixed unique id, only one entry is allowed.
                await self.async_set_unique_id(DOMAIN)
                self._abort_if_unique_id_configured()
                user_input.setdefault(CONF_REF_TEXT, "")
                user_input.setdefault(CONF_VOICE, DEFAULT_VOICE)
                return self.async_create_entry(
                    title="Qwen3-TTS",
                    data=user_input,
                )

        return self.async_show_form(
            step_id="user",
            data_schema=_schema(user_input or {}),
            errors=errors,
        )

    @staticmethod
    def async_get_options_flow(config_entry: config_entries.ConfigEntry):
        """Get the options flow for this handler."""
        return Qwen3TTSOptionsFlow()


class Qwen3TTSOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for Qwen3-TTS."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        errors: dict[str, str] = {}

        if user_input is not None:
            model_path = user_input.get(CONF_MODEL_PATH, DEFAULT_MODEL_PATH)
            ref_audio = user_input.get(CONF_REF_AUDIO, "")

            error = await self.hass.async_add_executor_job(_model_error, model_path)
            if error:
                errors["base"] = error
            elif (
                error := await self.hass.async_add_executor_job(
                    _ref_audio_error, ref_audio
                )
            ) is not None:
                errors[CONF_REF_AUDIO] = error
            else:
                return self.async_create_entry(title="", data=user_input)

        defaults: dict[str, Any] = {}
        for key in (CONF_MODEL_PATH, CONF_REF_AUDIO, CONF_REF_TEXT, CONF_LANGUAGE):
            value = self.config_entry.options.get(key)
            if value is None:
                value = self.config_entry.data.get(key)
            if value is not None:
                defaults[key] = value
        defaults.setdefault(CONF_MODEL_PATH, DEFAULT_MODEL_PATH)
        defaults.setdefault(CONF_REF_AUDIO, DEFAULT_REF_AUDIO)
        defaults.setdefault(CONF_REF_TEXT, "")
        defaults.setdefault(CONF_LANGUAGE, DEFAULT_LANGUAGE)

        return self.async_show_form(
            step_id="init",
            data_schema=_schema(defaults),
            errors=errors,
        )
