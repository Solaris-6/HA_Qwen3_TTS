#!/usr/bin/env python3
"""Check that a Qwen3-TTS server answers and produces valid audio.

Usage:

    python tools/test_server.py                        # server on localhost:8000
    python tools/test_server.py --url http://192.168.1.139:8000
    python tools/test_server.py --text "Buongiorno" --lang it

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REF = REPO_ROOT / "custom_components" / "qwen3_tts" / "audio" / "claudia.wav"
DEFAULT_OUT = REPO_ROOT / "tools" / "output" / "server_sample.wav"

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Print a check result and remember a failure."""
    print(f"[{'OK ' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def get(url: str, timeout: float = 10.0) -> tuple[int, bytes]:
    """GET `url`, returning status and body (HTTP errors included)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def post(url: str, payload: dict, timeout: float = 900.0) -> tuple[int, str, bytes, float]:
    """POST a JSON payload, returning status, content type, body and elapsed seconds."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return (
                response.status,
                response.headers.get("Content-Type", ""),
                response.read(),
                time.monotonic() - started,
            )
    except urllib.error.HTTPError as err:
        return (
            err.code,
            err.headers.get("Content-Type", ""),
            err.read(),
            time.monotonic() - started,
        )


def main() -> int:
    """Run every check against the server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--ref", default=str(DEFAULT_REF), help="Reference audio (WAV/MP3)")
    parser.add_argument("--text", default="Buongiorno! Questo è un test del server Qwen3-TTS.")
    parser.add_argument("--lang", default="it", help="Language code, e.g. it, en, de")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Where to save the WAV")
    args = parser.parse_args()

    base = args.url.rstrip("/")

    # 1) health, waiting for the model if it is still loading
    health = None
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        status, body = get(f"{base}/health")
        if status == 200:
            health = json.loads(body)
            if health.get("model_loaded"):
                break
        time.sleep(3)

    check("health returns 200", health is not None, json.dumps(health, ensure_ascii=False) if health else "")
    check("it is a qwen3-tts server", bool(health and health.get("name") == "qwen3-tts-server"))
    check("model loaded", bool(health and health.get("model_loaded")), str(health and health.get("device")))

    ref_path = Path(args.ref)
    if not ref_path.is_file():
        check("reference audio exists", False, str(ref_path))
        print("\nFAILURES:", failures)
        return 1
    ref_b64 = base64.b64encode(ref_path.read_bytes()).decode("ascii")

    # 2) synthesis
    status, content_type, body, elapsed = post(
        f"{base}/synthesize",
        {"text": args.text, "language": args.lang, "ref_audio": ref_b64, "ref_text": ""},
    )
    check("synthesize returns 200", status == 200, f"{status} {body[:200] if status != 200 else ''}")
    check("content type is audio/wav", content_type == "audio/wav", content_type)

    if status == 200:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(body)
        with wave.open(str(out), "rb") as wf:
            channels, width, rate, frames = (
                wf.getnchannels(),
                wf.getsampwidth(),
                wf.getframerate(),
                wf.getnframes(),
            )
        duration = frames / float(rate)
        check("wav is mono 16 bit", channels == 1 and width == 2, f"ch={channels} width={width}")
        check("wav has audio", duration > 0.5, f"{duration:.2f}s @ {rate} Hz -> {out}")
        print(f"       synthesis took {elapsed:.1f}s")

    # 3) error handling
    status, _, body, _ = post(f"{base}/synthesize", {"language": args.lang, "ref_audio": ref_b64})
    check("missing text -> 400", status == 400, str(status))

    status, _, body, _ = post(
        f"{base}/synthesize", {"text": "ciao", "language": "zz", "ref_audio": ref_b64}
    )
    check("bad language -> 400", status == 400, str(status))

    print()
    print("FAILURES:", failures if failures else "none")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
