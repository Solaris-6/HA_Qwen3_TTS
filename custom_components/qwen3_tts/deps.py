"""Installazione su richiesta delle dipendenze di Qwen3-TTS.

Home Assistant installa i requisiti di ``manifest.json`` **mentre viene caricato
il config flow** (``config_entries._load_integration`` →
``async_process_deps_reqs``).  ``qwen-tts`` trascina con sé un albero di
dipendenze molto pesante (PyTorch, transformers, gradio, librosa, numba,
onnxruntime, ...): se quell'installazione fallisce Home Assistant solleva
``RequirementsNotFound`` che nessuno cattora nel percorso del config flow, e
l'interfaccia mostra un generico «500 Internal Server Error».

Tenendo il requisito fuori dal manifest e installandolo qui, durante il setup
della config entry:

* il config flow si apre sempre, senza scaricare gigabyte in anticipo;
* l'installazione avviene dove gli errori vengono registrati, ritentati e
  mostrati allo stato della scheda («Setup retry»);
* ogni tentativo viene loggato per intero, così la causa reale è visibile in
  **Impostazioni → Sistema → Log**.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
import time
from typing import Final

_LOGGER = logging.getLogger(__name__)

REQUIREMENTS: Final[tuple[str, ...]] = ("qwen-tts==0.1.1",)

#: Massimo numero di tentativi consecutivi per ogni esecuzione di Home Assistant.
MAX_ATTEMPTS: Final = 3

#: Secondi di attesa prima di concedere un nuovo ciclo di tentativi.
RETRY_COOLDOWN: Final = 600.0

#: Timeout (secondi) per singolo comando di installazione.
INSTALL_TIMEOUT: Final = 1800

_attempts = 0
_last_error: str | None = None
_last_error_at = 0.0


def is_installed() -> bool:
    """True quando il pacchetto ``qwen_tts`` è importabile."""
    try:
        return importlib.util.find_spec("qwen_tts") is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _constraints() -> list[str]:
    """Restituisce gli argomenti ``--constraint`` di Home Assistant, se presenti.

    Usare lo stesso file di vincoli di Home Assistant evita di far slittare le
    versioni dei pacchetti già installati nel core.
    """
    try:
        import homeassistant
    except ImportError:  # pragma: no cover - impossibile dentro Home Assistant
        return []

    if not homeassistant.__file__:  # pragma: no cover
        return []

    constraints = os.path.join(
        os.path.dirname(homeassistant.__file__), "package_constraints.txt"
    )
    if os.path.isfile(constraints):
        return ["--constraint", constraints]
    return []


def _commands() -> list[tuple[str, list[str]]]:
    """Restituisce (etichetta, comando) dei tentativi di installazione, in ordine."""
    reqs = list(REQUIREMENTS)
    constraints = _constraints()
    return [
        (
            "uv",
            [sys.executable, "-m", "uv", "pip", "install", "--python", sys.executable, *constraints, *reqs],
        ),
        (
            "pip",
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *constraints, *reqs],
        ),
        (
            "pip --user",
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--user", *reqs],
        ),
    ]


def _run(cmd: list[str]) -> tuple[bool, str]:
    """Esegue un comando e restituisce (riuscito, ultime righe dell'output)."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=INSTALL_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout dopo {INSTALL_TIMEOUT}s: {' '.join(cmd)}"
    except OSError as err:
        return False, f"{type(err).__name__}: {err}"

    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if len(output) > 4000:
        output = "…" + output[-4000:]
    return proc.returncode == 0, output


def ensure_installed() -> str | None:
    """Garantisce che ``qwen-tts`` sia installato.

    Da chiamare **dall'executor** (non dal loop degli eventi).
    Restituisce ``None`` quando tutto è pronto, oppure una breve descrizione
    dell'errore.
    """
    global _attempts, _last_error, _last_error_at

    if is_installed():
        return None

    now = time.monotonic()
    if _last_error is not None and now - _last_error_at > RETRY_COOLDOWN:
        # È passato abbastanza tempo: ricomincia un nuovo ciclo di tentativi.
        _attempts = 0
    if _last_error is not None and _attempts >= MAX_ATTEMPTS:
        return _last_error

    _attempts += 1
    _LOGGER.info(
        "Installing %s (attempt %d/%d). This downloads several GB "
        "(PyTorch, transformers, ...) and can take a long time.",
        ", ".join(REQUIREMENTS),
        _attempts,
        MAX_ATTEMPTS,
    )

    for label, cmd in _commands():
        started = time.monotonic()
        _LOGGER.debug("Running: %s", " ".join(cmd))
        ok, output = _run(cmd)
        elapsed = int(time.monotonic() - started)
        if ok and is_installed():
            _LOGGER.info("Dependencies installed with %s in %ss.", label, elapsed)
            _last_error = None
            return None
        _LOGGER.error(
            "Installing the Qwen3-TTS dependencies with %s failed after %ss.\n%s",
            label,
            elapsed,
            output or "(nessun output)",
        )

    _last_error = "impossibile installare qwen-tts==0.1.1 (vedi log per il dettaglio)"
    _last_error_at = time.monotonic()
    return _last_error
