"""Support for the Qwen3-TTS text to speech service."""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
import wave
from typing import Any

import aiohttp

from homeassistant.components.tts import TextToSpeechEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import aiohttp_client
from homeassistant.helpers.entity_platform import AddEntitiesCallback

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
    DEFAULT_VOICE,
    HEALTH_TIMEOUT,
    LANGUAGE_NAMES,
    MODE_LOCAL,
    MODE_SERVER,
    SERVER_TIMEOUT,
    SUPPORTED_LANGUAGES,
    SUPPORTED_OPTIONS,
)

_LOGGER = logging.getLogger(__name__)

_SERVER_NAME = "qwen3-tts-server"

_model: Any = None
_loaded_model_path: str | None = None
_load_lock = threading.Lock()


def _load_model(model_path: str) -> Any:
    """Load the Qwen3-TTS model, blocking until it is ready.

    This runs in the executor, it must never be called from the event loop.
    """
    global _model, _loaded_model_path

    if _model is not None and _loaded_model_path == model_path:
        return _model

    with _load_lock:
        if _model is not None and _loaded_model_path == model_path:
            return _model

        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as err:
            _LOGGER.error(
                "Missing dependency for Qwen3-TTS (%s). "
                "Home Assistant installs the dependencies when the config "
                "entry is set up: check the log for messages from "
                "'qwen3_tts.deps' and reload the integration.",
                err,
            )
            raise

        _LOGGER.info("Loading the Qwen3-TTS model from %s", model_path)

        load_kwargs: dict[str, Any] = {}
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "cuda:0"
            load_kwargs["dtype"] = torch.bfloat16
        else:
            load_kwargs["dtype"] = torch.float32

        try:
            model = Qwen3TTSModel.from_pretrained(model_path, **load_kwargs)
        except Exception:
            _LOGGER.exception(
                "Unable to load the Qwen3-TTS model from %s", model_path
            )
            raise

        _model = model
        _loaded_model_path = model_path
        _LOGGER.info("Qwen3-TTS model loaded successfully")
        return model


def _preload_model(model_path: str) -> None:
    """Warm the model cache up in the background."""
    try:
        _load_model(model_path)
    except Exception:  # noqa: BLE001 - reported in _load_model as well
        _LOGGER.exception(
            "Could not preload the Qwen3-TTS model from %s", model_path
        )


def _synthesize(
    model_path: str,
    ref_audio: str,
    ref_text: str,
    message: str,
    language: str,
) -> bytes:
    """Synthesize `message` and return a mono 16 bit WAV file.

    This runs in the executor, it must never be called from the event loop.
    """
    # Imported lazily: in server mode Home Assistant never needs numpy (it is
    # not a Home Assistant dependency), only this local code path does.
    import numpy as np

    model = _load_model(model_path)

    # Qwen3-TTS expects a full language name such as "Italian", not "it".
    language_name = LANGUAGE_NAMES.get(language, "Italian")

    generate_kwargs: dict[str, Any] = {
        "text": message,
        "language": language_name,
        "ref_audio": ref_audio,
        "max_new_tokens": 4096,
    }
    if ref_text:
        # In-context learning: the transcript of the reference audio gives
        # the highest voice similarity.
        generate_kwargs["ref_text"] = ref_text
        generate_kwargs["x_vector_only_mode"] = False
    else:
        # Speaker embedding only: no transcript of the reference audio needed.
        generate_kwargs["x_vector_only_mode"] = True

    wavs, sample_rate = model.generate_voice_clone(**generate_kwargs)

    if not wavs:
        raise RuntimeError("Qwen3-TTS did not return any audio")

    audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise RuntimeError("Qwen3-TTS returned an empty audio buffer")

    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())

    return buffer.getvalue()


def _read_file(path: str) -> bytes:
    """Read a file, blocking; runs in the executor."""
    with open(path, "rb") as handle:
        return handle.read()


async def _ping_server(
    hass: HomeAssistant, session: aiohttp.ClientSession, url: str
) -> None:
    """Log whether the remote server is usable, without blocking setup."""
    del hass  # only kept for a stable signature
    try:
        async with session.get(
            f"{url}/health", timeout=aiohttp.ClientTimeout(total=HEALTH_TIMEOUT)
        ) as response:
            if response.status != 200:
                _LOGGER.warning(
                    "The Qwen3-TTS server at %s answered %s on /health: speech "
                    "synthesis will fail until the server works again.",
                    url,
                    response.status,
                )
                return
            payload = await response.json(content_type=None)
    except (aiohttp.ClientError, TimeoutError, ValueError, TypeError) as err:
        _LOGGER.warning(
            "The Qwen3-TTS server at %s is not reachable (%s). Speech "
            "synthesis will fail until it is back online.",
            url,
            err,
        )
        return

    if not isinstance(payload, dict) or payload.get("name") != _SERVER_NAME:
        _LOGGER.warning(
            "Nothing that looks like a Qwen3-TTS server answers at %s "
            "(check the URL in the integration options).",
            url,
        )
        return

    if payload.get("status") == "error":
        _LOGGER.error(
            "The Qwen3-TTS server at %s cannot load the model: %s",
            url,
            payload.get("load_error"),
        )
        return

    _LOGGER.info(
        "Qwen3-TTS server at %s is reachable (model=%s, device=%s, loaded=%s)",
        url,
        payload.get("model"),
        payload.get("device"),
        payload.get("model_loaded"),
    )


class _BaseQwen3TTSEntity(TextToSpeechEntity):
    """Properties shared by the local and the remote TTS entity."""

    _attr_name = "Qwen3-TTS"
    _attr_unique_id = "qwen3_tts"

    def __init__(
        self,
        ref_audio: str,
        ref_text: str,
        voice: str,
        language: str,
    ) -> None:
        """Initialize the entity."""
        self._ref_audio = ref_audio or DEFAULT_REF_AUDIO
        self._ref_text = ref_text or ""
        self._default_voice = voice
        self._default_language = language

    @property
    def default_language(self) -> str:
        """Return the default language."""
        return self._default_language

    @property
    def supported_languages(self) -> list[str]:
        """Return the list of supported languages."""
        return SUPPORTED_LANGUAGES

    @property
    def default_options(self) -> dict[str, str]:
        """Return the default options."""
        return {CONF_VOICE: self._default_voice}

    @property
    def supported_options(self) -> list[str]:
        """Return the supported options."""
        return SUPPORTED_OPTIONS

    def _language_code(self, language: str) -> str | None:
        """Normalize the requested language, or None when unsupported."""
        lang = (language or self._default_language).split("-")[0].lower()
        if lang not in SUPPORTED_LANGUAGES:
            _LOGGER.error("Language %s is not supported by Qwen3-TTS", language)
            return None
        return lang

    def _reference_audio_missing(self) -> bool:
        """Report a missing reference audio file and return True when absent."""
        if os.path.isfile(self._ref_audio):
            return False
        _LOGGER.error(
            "The reference audio %s does not exist. Set a valid file in "
            "the Qwen3-TTS integration options.",
            self._ref_audio,
        )
        return True


class Qwen3TTSServerEntity(_BaseQwen3TTSEntity):
    """TTS entity that asks a Qwen3-TTS server in the network."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        server_url: str,
        ref_audio: str,
        ref_text: str,
        voice: str,
        language: str,
    ) -> None:
        """Initialize the entity."""
        super().__init__(ref_audio, ref_text, voice, language)
        self._session = session
        self._server_url = server_url.rstrip("/")

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> tuple[str | None, bytes | None]:
        """Return a tuple of file extension and data as bytes."""
        lang = self._language_code(language)
        if lang is None:
            return None, None
        if not message or not message.strip():
            _LOGGER.error("Qwen3-TTS received an empty message")
            return None, None
        if self._reference_audio_missing():
            return None, None

        try:
            raw = await self.hass.async_add_executor_job(
                _read_file, self._ref_audio
            )
        except OSError as err:
            _LOGGER.error(
                "Unable to read the reference audio %s: %s", self._ref_audio, err
            )
            return None, None

        payload = {
            "text": message,
            "language": lang,
            "ref_audio": base64.b64encode(raw).decode("ascii"),
            "ref_text": self._ref_text,
        }

        try:
            async with self._session.post(
                f"{self._server_url}/synthesize",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=SERVER_TIMEOUT),
            ) as response:
                if response.status != 200:
                    body = (await response.text())[:500]
                    _LOGGER.error(
                        "The Qwen3-TTS server at %s answered %s: %s",
                        self._server_url,
                        response.status,
                        body,
                    )
                    return None, None
                data = await response.read()
        except TimeoutError:
            _LOGGER.error(
                "The Qwen3-TTS server at %s did not answer within %s seconds",
                self._server_url,
                int(SERVER_TIMEOUT),
            )
            return None, None
        except aiohttp.ClientError as err:
            _LOGGER.error(
                "Unable to reach the Qwen3-TTS server at %s: %s",
                self._server_url,
                err,
            )
            return None, None

        if not data:
            _LOGGER.error("The Qwen3-TTS server returned an empty response")
            return None, None

        return "wav", data


class Qwen3TTSLocalEntity(_BaseQwen3TTSEntity):
    """TTS entity that runs the model inside Home Assistant."""

    def __init__(
        self,
        model_path: str,
        ref_audio: str,
        ref_text: str,
        voice: str,
        language: str,
    ) -> None:
        """Initialize the entity."""
        super().__init__(ref_audio, ref_text, voice, language)
        self._model_path = model_path

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> tuple[str | None, bytes | None]:
        """Return a tuple of file extension and data as bytes."""
        lang = self._language_code(language)
        if lang is None:
            return None, None
        if self._reference_audio_missing():
            return None, None

        try:
            data = await self.hass.async_add_executor_job(
                _synthesize,
                self._model_path,
                self._ref_audio,
                self._ref_text,
                message,
                lang,
            )
        except Exception:  # noqa: BLE001 - reported to the log for debugging
            _LOGGER.exception("Qwen3-TTS failed to generate speech")
            return None, None

        return "wav", data


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Qwen3-TTS TTS platform."""
    data = config_entry.data
    options = config_entry.options

    def _get(key: str, default: Any) -> Any:
        return options.get(key, data.get(key, default))

    ref_audio = _get(CONF_REF_AUDIO, "") or DEFAULT_REF_AUDIO
    ref_text = _get(CONF_REF_TEXT, "") or ""
    voice = _get(CONF_VOICE, DEFAULT_VOICE)
    language = _get(CONF_LANGUAGE, DEFAULT_LANGUAGE)
    # The mode is part of the entry data and cannot be changed in options.
    mode = data.get(CONF_MODE, MODE_LOCAL)

    if mode == MODE_SERVER:
        server_url = (_get(CONF_SERVER_URL, "") or "").strip()
        if not server_url:
            _LOGGER.error(
                "No Qwen3-TTS server URL is configured. Set it in the "
                "integration options."
            )
            return

        session = aiohttp_client.async_get_clientsession(hass)
        async_add_entities(
            [
                Qwen3TTSServerEntity(
                    session, server_url, ref_audio, ref_text, voice, language
                )
            ]
        )
        hass.async_create_background_task(
            _ping_server(hass, session, server_url.rstrip("/")),
            "qwen3_tts_server_ping",
        )
        return

    model_path = _get(CONF_MODEL_PATH, DEFAULT_MODEL_PATH)
    async_add_entities(
        [Qwen3TTSLocalEntity(model_path, ref_audio, ref_text, voice, language)]
    )

    if not os.path.isdir(model_path):
        _LOGGER.warning(
            "The model directory %s does not exist, Qwen3-TTS cannot work "
            "until it is fixed in the integration options.",
            model_path,
        )
        return

    # Load the weights in the background so the first synthesis does not have
    # to wait for them.
    hass.async_create_background_task(
        hass.async_add_executor_job(_preload_model, model_path),
        "qwen3_tts_model_preload",
    )
