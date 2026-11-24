#!/usr/bin/env python3
"""Standalone HTTP server for Qwen3-TTS voice cloning.

This is the counterpart of the Home Assistant integration running in
"network server" mode: Home Assistant sends the text to speak together with
a reference audio and receives a mono 16 bit WAV file.

Endpoints
---------
``GET /health``
    Liveness/model information. Never loads the model.

``POST /synthesize``
    JSON body::

        {
          "text": "Buongiorno",             # required
          "language": "it",                 # optional, default "it"
          "ref_audio": "<base64 audio>",     # required unless --default-ref
          "ref_text": "trascrizione",        # optional, improves cloning
          "max_new_tokens": 4096            # optional
        }

    Response: ``200 audio/wav`` on success, or a JSON ``{"error": ...}``.

Only the Python standard library plus the packages required by ``qwen-tts``
are used, so the server runs in the same virtual environment as the model.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import logging
import os
import sys
import tempfile
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar
from urllib.parse import urlparse

API_NAME = "qwen3-tts-server"
API_VERSION = "1.0"

MAX_TEXT_LENGTH = 4000
MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_NEW_TOKENS_DEFAULT = 4096
MAX_NEW_TOKENS_LIMIT = 8192

#: Number of reference audios kept with their extracted voice prompt.
PROMPT_CACHE_SIZE = 8

LOG = logging.getLogger(API_NAME)

# Home Assistant sends the ISO 639-1 code; the model wants the full name.
ISO_TO_LANGUAGE: dict[str, str] = {
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


class ServerError(Exception):
    """An error that is reported to the client with an HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        """Store the HTTP status and the human readable message."""
        super().__init__(message)
        self.status = status


def _encode_wav(wavs: list[Any], sample_rate: int) -> bytes:
    """Convert the model output into a mono 16 bit WAV file."""
    import numpy as np

    if not wavs:
        raise ServerError(500, "the model returned no audio")

    audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ServerError(500, "the model returned an empty audio buffer")

    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


class Engine:
    """Loads the model once and performs the synthesis."""

    def __init__(self, model: str, device: str, default_ref: str | None) -> None:
        """Prepare an engine; the model is loaded lazily on first use."""
        self.model_id = model
        self.device_request = device
        self.default_ref = default_ref
        self.default_ref_audio: str | None = None

        self._model: Any = None
        self._load_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._prompts: dict[tuple[str, str], list[Any]] = {}

        self.device = "unknown"
        self.load_error: str | None = None
        self.generations = 0
        self.started_at = time.time()

    @property
    def loaded(self) -> bool:
        """Return True when the model is in memory."""
        return self._model is not None

    def health(self) -> dict[str, Any]:
        """Return the JSON payload served by /health."""
        return {
            "status": "error" if self.load_error else "ok",
            "name": API_NAME,
            "version": API_VERSION,
            "model": self.model_id,
            "model_loaded": self.loaded,
            "device": self.device,
            "load_error": self.load_error,
            "generations": self.generations,
            "languages": sorted(ISO_TO_LANGUAGE),
            "uptime": int(time.time() - self.started_at),
        }

    def load(self) -> Any:
        """Load the model into memory (thread safe, runs at most once)."""
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model

            try:
                import torch
                from qwen_tts import Qwen3TTSModel
            except Exception as err:  # noqa: BLE001 - reported to the client
                self.load_error = f"missing dependency: {err}"
                raise ServerError(503, self.load_error) from err

            use_cuda = (
                torch.cuda.is_available()
                if self.device_request == "auto"
                else self.device_request == "cuda"
            )
            load_kwargs: dict[str, Any] = {}
            if use_cuda:
                load_kwargs["device_map"] = "cuda:0"
                load_kwargs["dtype"] = torch.bfloat16
                self.device = "cuda"
            else:
                load_kwargs["dtype"] = torch.float32
                self.device = "cpu"

            LOG.info(
                "Loading the Qwen3-TTS model from %s on %s (this can take a "
                "minute and several GB of RAM)...",
                self.model_id,
                self.device,
            )
            started = time.monotonic()
            try:
                model = Qwen3TTSModel.from_pretrained(self.model_id, **load_kwargs)
            except Exception as err:  # noqa: BLE001 - reported to the client
                self.load_error = f"{type(err).__name__}: {err}"
                LOG.exception("Unable to load the model from %s", self.model_id)
                raise ServerError(503, f"unable to load the model: {err}") from err

            self._model = model
            self.load_error = None
            LOG.info("Model loaded in %.1fs", time.monotonic() - started)
            return model

    @staticmethod
    def decode_audio(ref_audio: str) -> tuple[Any, int]:
        """Decode a base64 audio payload into ``(waveform, sample_rate)``."""
        import numpy as np

        payload = ref_audio.split(",", 1)[1] if ref_audio.startswith("data:") else ref_audio
        try:
            raw = base64.b64decode(payload, validate=False)
        except (binascii.Error, ValueError) as err:
            raise ServerError(400, f"invalid base64 reference audio: {err}") from err
        if not raw:
            raise ServerError(400, "the reference audio is empty")

        audio: Any = None
        sample_rate = 0
        try:
            import soundfile as sf

            audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception:  # noqa: BLE001 - fall back to librosa/ffmpeg below
            path: str | None = None
            try:
                descriptor, path = tempfile.mkstemp(suffix=".audio")
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(raw)
                import librosa

                audio, sample_rate = librosa.load(path, sr=None, mono=True)
            except Exception as err:  # noqa: BLE001 - reported to the client
                raise ServerError(
                    400, f"unable to decode the reference audio: {err}"
                ) from err
            finally:
                if path:
                    try:
                        os.unlink(path)
                    except OSError:  # pragma: no cover - best effort cleanup
                        pass

        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)
        if waveform.size == 0:
            raise ServerError(400, "the reference audio contains no samples")
        return waveform, int(sample_rate)

    def _default_reference(self) -> str:
        """Return the base64 payload of --default-ref, reading it once."""
        if self.default_ref_audio is not None:
            return self.default_ref_audio
        if not self.default_ref:
            raise ServerError(400, "no reference audio provided")
        try:
            with open(self.default_ref, "rb") as handle:
                raw = handle.read()
        except OSError as err:
            raise ServerError(
                500, f"unable to read the default reference {self.default_ref}: {err}"
            ) from err
        self.default_ref_audio = base64.b64encode(raw).decode("ascii")
        return self.default_ref_audio

    def synthesize(
        self,
        text: str,
        language: str,
        ref_audio: str | None,
        ref_text: str,
        max_new_tokens: int,
    ) -> bytes:
        """Synthesize `text` and return a WAV file."""
        model = self.load()

        ref_text = (ref_text or "").strip()
        if not ref_audio:
            ref_audio = self._default_reference()

        digest = hashlib.sha256(ref_audio.encode("utf-8")).hexdigest()
        key = (digest, ref_text)

        with self._model_lock:
            # The prompt holds the speaker embedding and the speech codes of
            # the reference: extracting them is expensive, so it is cached.
            prompt = self._prompts.get(key)
            if prompt is None:
                waveform, sample_rate = self.decode_audio(ref_audio)
                prompt = model.create_voice_clone_prompt(
                    ref_audio=(waveform, sample_rate),
                    ref_text=ref_text or None,
                    x_vector_only_mode=not ref_text,
                )
                if len(self._prompts) >= PROMPT_CACHE_SIZE:
                    self._prompts.pop(next(iter(self._prompts)))
                self._prompts[key] = prompt
                LOG.info("Reference voice prepared (ref_text=%r)", ref_text[:60])

            language_name = ISO_TO_LANGUAGE.get(language)
            if language_name is None:
                raise ServerError(
                    400,
                    f"unsupported language {language!r}; "
                    f"use one of: {', '.join(sorted(ISO_TO_LANGUAGE))}",
                )

            LOG.info("Synthesizing %d characters in %s", len(text), language_name)
            started = time.monotonic()
            wavs, sample_rate = model.generate_voice_clone(
                text=text,
                language=language_name,
                voice_clone_prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            data = _encode_wav(wavs, sample_rate)
            self.generations += 1
            LOG.info("Synthesized in %.1fs (%d bytes)", time.monotonic() - started, len(data))
            return data


class Handler(BaseHTTPRequestHandler):
    """HTTP routes of the server."""

    engine: ClassVar[Engine]
    server_version = f"{API_NAME}/{API_VERSION}"

    def log_message(self, fmt: str, *args: Any) -> None:
        """Send the default access log to our logger."""
        LOG.info("%s %s", self.address_string(), fmt % args)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        """Send a JSON response."""
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _wav(self, data: bytes) -> None:
        """Send the synthesized WAV file."""
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - dictated by BaseHTTPRequestHandler
        """Serve /health (and / for convenience)."""
        path = urlparse(self.path).path
        if path in ("/", "/health", "/healthz"):
            self._json(200, self.engine.health())
        else:
            self._json(404, {"error": f"unknown path {path}"})

    def do_POST(self) -> None:  # noqa: N802 - dictated by BaseHTTPRequestHandler
        """Serve /synthesize."""
        path = urlparse(self.path).path
        if path != "/synthesize":
            self._json(404, {"error": f"unknown path {path}"})
            return

        try:
            payload = self._read_json()
            data = self._synthesize(payload)
        except ServerError as err:
            LOG.warning("Request rejected (%s): %s", err.status, err)
            self._json(err.status, {"error": str(err)})
        except Exception as err:  # noqa: BLE001 - never crash the server
            LOG.exception("Unexpected failure while synthesizing")
            self._json(500, {"error": f"{type(err).__name__}: {err}"})
        else:
            self._wav(data)

    def _read_json(self) -> dict[str, Any]:
        """Read and parse the JSON request body."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as err:
            raise ServerError(400, "invalid Content-Length") from err
        if length <= 0:
            raise ServerError(400, "the request body is empty")
        if length > MAX_BODY_BYTES:
            raise ServerError(413, "the request body is too large")

        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            raise ServerError(400, f"invalid JSON body: {err}") from err
        if not isinstance(payload, dict):
            raise ServerError(400, "the JSON body must be an object")
        return payload

    def _synthesize(self, payload: dict[str, Any]) -> bytes:
        """Validate the payload and run the synthesis."""
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ServerError(400, "the 'text' field is required")
        text = text.strip()
        if len(text) > MAX_TEXT_LENGTH:
            raise ServerError(
                400, f"the text is too long ({len(text)} > {MAX_TEXT_LENGTH} characters)"
            )

        language = payload.get("language", "it")
        if not isinstance(language, str) or language not in ISO_TO_LANGUAGE:
            raise ServerError(
                400,
                f"unsupported language {language!r}; "
                f"use one of: {', '.join(sorted(ISO_TO_LANGUAGE))}",
            )

        ref_audio = payload.get("ref_audio")
        if ref_audio is not None and not isinstance(ref_audio, str):
            raise ServerError(400, "'ref_audio' must be a base64 string")

        ref_text = payload.get("ref_text") or ""
        if not isinstance(ref_text, str):
            raise ServerError(400, "'ref_text' must be a string")

        max_new_tokens = payload.get("max_new_tokens", MAX_NEW_TOKENS_DEFAULT)
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
            raise ServerError(400, "'max_new_tokens' must be an integer")
        max_new_tokens = max(1, min(max_new_tokens, MAX_NEW_TOKENS_LIMIT))

        return self.engine.synthesize(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            max_new_tokens=max_new_tokens,
        )


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run the server."""
    parser = argparse.ArgumentParser(
        prog="qwen3-tts-server",
        description="HTTP server exposing Qwen3-TTS voice cloning on the LAN.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Address to bind (default: all)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Local folder or Hugging Face repo id of the model",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: cuda when available)",
    )
    parser.add_argument(
        "--default-ref",
        default=None,
        help="Optional reference audio used when the request has no ref_audio",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load the model at startup instead of on the first request",
    )
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ...")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if args.default_ref and not os.path.isfile(args.default_ref):
        LOG.error("The default reference audio %s does not exist", args.default_ref)
        return 2

    engine = Engine(model=args.model, device=args.device, default_ref=args.default_ref)
    Handler.engine = engine

    if args.preload:
        threading.Thread(target=engine.load, name="model-preload", daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    LOG.info(
        "%s %s listening on http://%s:%s (model=%s, device=%s)",
        API_NAME,
        API_VERSION,
        args.host,
        args.port,
        args.model,
        args.device,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive shutdown
        LOG.info("Shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
