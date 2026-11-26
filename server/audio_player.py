#!/usr/bin/env python3
"""Simple HTTP audio playback server for Raspberry Pi 4.

Accepts POST /play with JSON {"file": "name.wav"} and plays it via mpv
on the 3.5 mm headphone jack.  Also accepts GET /play?file=name.wav.

Endpoints
---------
GET /health     -> {"status": "ok"}
POST /play      -> {"file": "x.wav"}  plays and returns {"status": "ok"}
GET /play?file= -> plays and returns {"status": "ok"}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

AUDIO_DIR = os.path.expanduser("~/audio")
LOG = logging.getLogger("audio-player")

# Only one playback at a time.
_play_lock = threading.Lock()


def play_file(filename: str) -> bool:
    """Play a file with mpv on the headphone jack. Returns True on success."""
    path = os.path.join(AUDIO_DIR, filename)
    if not os.path.isfile(path):
        LOG.error("Audio file not found: %s", path)
        return False

    with _play_lock:
        LOG.info("Playing %s", filename)
        try:
            subprocess.run(
                [
                    "mpv",
                    "--no-video",
                    "--really-quiet",
                    "--audio-device=alsa/plughw:0,0",
                    path,
                ],
                check=True,
                timeout=120,
            )
            LOG.info("Finished playing %s", filename)
            return True
        except subprocess.CalledProcessError as err:
            LOG.error("mpv failed for %s: %s", filename, err)
            return False
        except subprocess.TimeoutExpired:
            LOG.error("mpv timed out for %s", filename)
            return False


class Handler(BaseHTTPRequestHandler):
    server_version = "audio-player/1.0"

    def log_message(self, fmt, *args):
        LOG.info("%s %s", self.address_string(), fmt % args)

    def _json(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/health", "/healthz"):
            self._json(200, {"status": "ok", "audio_dir": AUDIO_DIR})
            return
        if parsed.path == "/play":
            qs = parse_qs(parsed.query)
            filename = qs.get("file", [None])[0]
            if not filename:
                self._json(400, {"error": "missing 'file' parameter"})
                return
            ok = play_file(filename)
            self._json(200 if ok else 404, {"status": "ok" if ok else "error", "file": filename})
            return
        self._json(404, {"error": "unknown path"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/play":
            self._json(404, {"error": "unknown path"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length)) if length else {}
        except (json.JSONDecodeError, ValueError):
            self._json(400, {"error": "invalid JSON"})
            return
        filename = payload.get("file")
        if not filename:
            self._json(400, {"error": "missing 'file' field"})
            return
        ok = play_file(filename)
        self._json(200 if ok else 404, {"status": "ok" if ok else "error", "file": filename})


def main():
    global AUDIO_DIR
    parser = argparse.ArgumentParser(description="HTTP audio playback server for Pi 4")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--audio-dir", default=AUDIO_DIR)
    args = parser.parse_args()

    AUDIO_DIR = os.path.expanduser(args.audio_dir)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    LOG.info("Audio player listening on http://%s:%s (dir=%s)", args.host, args.port, AUDIO_DIR)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
