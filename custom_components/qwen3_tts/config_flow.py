"""Config flow for Qwen3-TTS integration."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import aiohttp
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.config_entries import ConfigFlow
from homeassistant.helpers import aiohttp_client

from .const import (
    CONF_LANGUAGE,
    CONF_MODE,
    CONF_MODEL_PATH,
    CONF_REF_AUDIO,
    CONF_REF_TEXT,
    CONF_SERVER_URL,
    CONF_VOICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL_PATH,
    DEFAULT_REF_AUDIO,
    DEFAULT_SERVER_URL,
    DEFAULT_VOICE,
    DOMAIN,
    HEALTH_TIMEOUT,
    MODE_LOCAL,
    MODE_SERVER,
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

_SERVER_NAME = "qwen3-tts-server"


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


def _server_url_error(url: str) -> str | None:
    """Return the translation key describing why the server URL is unusable."""
    if not url or url.isspace():
        return "invalid_url"
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return "invalid_url"
    if any(char.isspace() for char in url):
        return "invalid_url"
    return None


async def _server_status(hass: Any, url: str) -> str | None:
    """Return None when `url` answers as a Qwen3-TTS server, else an error key."""
    clean = url.strip().rstrip("/")
    session = aiohttp_client.async_get_clientsession(hass)
    try:
        async with session.get(
            f"{clean}/health", timeout=aiohttp.ClientTimeout(total=HEALTH_TIMEOUT)
        ) as response:
            if response.status != 200:
                return "cannot_connect"
            payload = await response.json(content_type=None)
    except (aiohttp.ClientError, TimeoutError, ValueError, TypeError):
        return "cannot_connect"

    if not isinstance(payload, dict) or payload.get("name") != _SERVER_NAME:
        return "cannot_connect"
    if payload.get("status") == "error":
        return "server_error"
    return None


def _ref_schema(defaults: dict[str, Any]) -> dict[Any, Any]:
    """Return the fields shared by both modes."""
    return {
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


def _local_schema(defaults: dict[str, Any]) -> vol.Schema:
    """Build the schema of the mode that runs the model inside HA."""
    return vol.Schema(
        {
            vol.Optional(
                CONF_MODEL_PATH,
                default=defaults.get(CONF_MODEL_PATH, DEFAULT_MODEL_PATH),
            ): str,
            **_ref_schema(defaults),
        }
    )


def _server_url_schema(defaults: dict[str, Any]) -> vol.Schema:
    """Build the schema asking for the server address."""
    return vol.Schema(
        {
            vol.Optional(
                CONF_SERVER_URL,
                default=defaults.get(CONF_SERVER_URL, DEFAULT_SERVER_URL),
            ): str,
        }
    )


def _server_schema(defaults: dict[str, Any]) -> vol.Schema:
    """Build the schema of the mode that uses a server in the network."""
    return vol.Schema(_ref_schema(defaults))


class Qwen3TTSConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Qwen3-TTS."""

    VERSION = 1

    _server_url: str = ""

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Ask where the model runs: a server in the network or this HA."""
        errors: dict[str, str] = {}

        if user_input is not None:
            url = (user_input.get(CONF_SERVER_URL) or "").strip()

            if not url:
                # No server: run the model inside Home Assistant.
                return await self.async_step_local()

            if (error := _server_url_error(url)) is not None:
                errors["base"] = error
            elif (error := await _server_status(self.hass, url)) is not None:
                errors["base"] = error
            else:
                self._server_url = url.rstrip("/")
                return await self.async_step_server()

        return self.async_show_form(
            step_id="user",
            data_schema=_server_url_schema(user_input or {}),
            errors=errors,
        )

    async def async_step_server(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure the voice used with the remote server."""
        errors: dict[str, str] = {}

        if user_input is not None:
            ref_audio = user_input.get(CONF_REF_AUDIO, DEFAULT_REF_AUDIO)
            if (
                error := await self.hass.async_add_executor_job(
                    _ref_audio_error, ref_audio
                )
            ) is not None:
                errors[CONF_REF_AUDIO] = error
            else:
                await self.async_set_unique_id(DOMAIN)
                self._abort_if_unique_id_configured()
                user_input[CONF_MODE] = MODE_SERVER
                user_input[CONF_SERVER_URL] = self._server_url
                user_input.setdefault(CONF_REF_TEXT, "")
                user_input.setdefault(CONF_VOICE, DEFAULT_VOICE)
                return self.async_create_entry(title="Qwen3-TTS", data=user_input)

        return self.async_show_form(
            step_id="server",
            data_schema=_server_schema(user_input or {}),
            errors=errors,
        )

    async def async_step_local(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure the model running inside Home Assistant."""
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
                user_input[CONF_MODE] = MODE_LOCAL
                user_input.setdefault(CONF_REF_TEXT, "")
                user_input.setdefault(CONF_VOICE, DEFAULT_VOICE)
                return self.async_create_entry(
                    title="Qwen3-TTS",
                    data=user_input,
                )

        return self.async_show_form(
            step_id="local",
            data_schema=_local_schema(user_input or {}),
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
        server_mode = self.config_entry.data.get(CONF_MODE) == MODE_SERVER

        if user_input is not None:
            if server_mode:
                url = (user_input.get(CONF_SERVER_URL) or "").strip()
                if (error := _server_url_error(url)) is not None:
                    errors["base"] = error
                elif (error := await _server_status(self.hass, url)) is not None:
                    errors["base"] = error
                else:
                    user_input[CONF_SERVER_URL] = url.rstrip("/")
                    error = await self.hass.async_add_executor_job(
                        _ref_audio_error, user_input.get(CONF_REF_AUDIO, "")
                    )
                    if error:
                        errors[CONF_REF_AUDIO] = error
                    else:
                        return self.async_create_entry(title="", data=user_input)
            else:
                model_path = user_input.get(CONF_MODEL_PATH, DEFAULT_MODEL_PATH)
                ref_audio = user_input.get(CONF_REF_AUDIO, "")

                error = await self.hass.async_add_executor_job(
                    _model_error, model_path
                )
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
        keys = [CONF_REF_AUDIO, CONF_REF_TEXT, CONF_LANGUAGE]
        if server_mode:
            keys.insert(0, CONF_SERVER_URL)
        else:
            keys.insert(0, CONF_MODEL_PATH)

        for key in keys:
            value = self.config_entry.options.get(key)
            if value is None:
                value = self.config_entry.data.get(key)
            if value is not None:
                defaults[key] = value

        defaults.setdefault(CONF_MODEL_PATH, DEFAULT_MODEL_PATH)
        defaults.setdefault(CONF_SERVER_URL, DEFAULT_SERVER_URL)
        defaults.setdefault(CONF_REF_AUDIO, DEFAULT_REF_AUDIO)
        defaults.setdefault(CONF_REF_TEXT, "")
        defaults.setdefault(CONF_LANGUAGE, DEFAULT_LANGUAGE)

        if server_mode:
            # The URL is asked in the config flow's first step, but it has to
            # be editable here too (the server may change address).
            schema = vol.Schema(
                {
                    vol.Optional(
                        CONF_SERVER_URL, default=defaults[CONF_SERVER_URL]
                    ): str,
                    **_ref_schema(defaults),
                }
            )
        else:
            schema = _local_schema(defaults)

        return self.async_show_form(
            step_id="init",
            data_schema=schema,
            errors=errors,
        )
