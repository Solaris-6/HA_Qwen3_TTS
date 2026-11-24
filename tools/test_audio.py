#!/usr/bin/env python3
"""Diagnostica audio per l'integrazione Qwen3 TTS di Home Assistant.

Questo script aiuta a separare due problemi diversi:

  1) La RIProduzione audio (browser/HA non riproduce un WAV)
  2) La SINTESI (il modello Qwen3-TTS non genera audio valido)

Comandi
-------
  tone   Genera un file WAV di prova (tono da 440 Hz) in tools/output/
  check  Valida un file WAV (header, durata, ampiezza, silenzio)
  synth  Esegue la sintesi REALE con il codice dell'integrazione

Esempi
------
  python tools/test_audio.py tone
  python tools/test_audio.py check tools/output/test_tone.wav
  python tools/test_audio.py synth "Ciao, prova di sintesi." --model D:/modelli/Qwen3-TTS-12Hz-0.6B-Base

Per il comando `synth` servono numpy + qwen-tts (+ torch). Se il Python
corrente non li ha, usa l'ambiente che li contiene, ad esempio:

  E:\\ClaudiaAI\\.venv\\Scripts\\python.exe tools/test_audio.py synth ...

Procedura consigliata per capire dove sta il problema
-----------------------------------------------------
  1) python tools/test_audio.py tone
  2) copia tools/output/test_tone.wav in /config/www/ di Home Assistant
  3) apri in un browser: http://<ha>:8123/local/test_tone.wav
     - si sente il tono  -> la riproduzione funziona: il problema e' la sintesi
     - non si sente      -> il problema e' HA/browser, non l'integrazione
  4) python tools/test_audio.py synth "..." --model /percorso/modello
     -> se anche qui fallisce, il log dell'eccezione dice perche'
"""

from __future__ import annotations

import argparse
import array
import math
import os
import sys
import time
import types
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

SAMPLE_RATE = 24000  # frequenza campionamento nativa Qwen3-TTS
TONE_FREQ = 440.0    # nota La


def _print_utf8_setup() -> None:
    """Evita problemi di codifica su console Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass


def _ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"[...]  {msg}")


# ---------------------------------------------------------------------------
# check: valida un file WAV
# ---------------------------------------------------------------------------


def _check_wav(path: Path) -> bool:
    """Analizza un file WAV e restituisce True se valido e riproducibile."""
    if not path.is_file():
        _fail(f"File non trovato: {path}")
        return False

    size = path.stat().st_size
    if size < 44:
        _fail(f"{path.name}: file troppo piccolo ({size} byte) per essere un WAV")
        return False

    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.getnframes()
            duration = frames / rate if rate else 0.0
            raw = wf.readframes(frames)
    except wave.Error as err:
        _fail(f"{path.name}: header WAV non valido ({err})")
        return False
    except EOFError as err:
        _fail(f"{path.name}: file WAV troncato ({err})")
        return False

    print(f"       file      : {path}")
    print(f"       dimensione: {size} byte")
    print(f"       canali    : {channels}")
    print(f"       bit       : {width * 8}")
    print(f"       sample rate: {rate} Hz")
    print(f"       durata    : {duration:.2f} s ({frames} frame)")

    if rate <= 0 or frames <= 0 or not raw:
        _fail("Il WAV non contiene alcun frame audio")
        return False

    # ampiezza massima (16 bit)
    if width == 2:
        samples = array.array("h")
        samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
        peak = max((abs(s) for s in samples), default=0)
        peak_norm = peak / 32768.0
    else:
        peak = 0
        peak_norm = 0.0

    print(f"       ampiezza  : {peak} / 32767 ({peak_norm * 100:.1f}%)")

    if peak == 0:
        _fail("Il WAV e' tutto silenzio: non verrà sentito nulla")
        return False

    _ok(f"{path.name} e' un WAV valido, {duration:.2f} s, {rate} Hz, "
        f"ampiezza {peak_norm * 100:.0f}%")
    return True


# ---------------------------------------------------------------------------
# tone: genera un file di prova
# ---------------------------------------------------------------------------


def cmd_tone(args: argparse.Namespace) -> int:
    """Genera un tono di prova e ne valida l'output."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "test_tone.wav"
    seconds = args.seconds
    total = int(SAMPLE_RATE * seconds)
    fade = int(SAMPLE_RATE * 0.02)  # 20 ms di dissolvenza anti-clangore

    samples = array.array("h")
    for i in range(total):
        env = 1.0
        if i < fade:
            env = i / fade
        elif i > total - fade:
            env = (total - i) / fade
        value = int(0.6 * 32767 * env * math.sin(2 * math.pi * TONE_FREQ * i / SAMPLE_RATE))
        samples.append(value)

    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())

    _ok(f"tono da {TONE_FREQ:.0f} Hz generato: {out} "
        f"({out.stat().st_size} byte, {seconds} s)")
    if not _check_wav(out):
        return 1

    print()
    print("Prossimi passi per testare la RIproduzione in Home Assistant:")
    print(f"  1) copia  {out}  in /config/www/ di Home Assistant")
    print("     (File editor, add-on SSH/SFTP o cartella condivisa)")
    print("  2) apri in un browser:")
    print("       http://<indirizzo-ha>:8123/local/test_tone.wav")
    print("  3) se senti il tono -> la riproduzione funziona, il problema e' la sintesi")
    print("     se non lo senti  -> il problema e' HA/browser, non l'integrazione")
    return 0


# ---------------------------------------------------------------------------
# check: valida un file esistente
# ---------------------------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> int:
    """Valida un file WAV indicato dall'utente."""
    path = Path(args.file).expanduser()
    return 0 if _check_wav(path) else 1


# ---------------------------------------------------------------------------
# synth: sintesi reale usando il codice dell'integrazione
# ---------------------------------------------------------------------------


def _install_homeassistant_stubs() -> None:
    """Inserisce moduli Home Assistant fittizi per importare tts.py fuori da HA.

    tts.py importa solo tipi da homeassistant all'avvio; la sintesi vera e
    propria non tocca HA, quindi basta rendere gli import disponibili.
    """

    def _mod(name: str, **attrs: object) -> types.ModuleType:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
        return module

    ha = _mod("homeassistant")
    components = _mod("homeassistant.components")
    ha.components = components
    components.tts = _mod(
        "homeassistant.components.tts",
        TextToSpeechEntity=object,
        TTSAudioType=tuple,
    )
    ha.config_entries = _mod("homeassistant.config_entries", ConfigEntry=object)
    ha.core = _mod("homeassistant.core", HomeAssistant=object)
    helpers = _mod("homeassistant.helpers")
    ha.helpers = helpers
    helpers.entity_platform = _mod(
        "homeassistant.helpers.entity_platform", AddEntitiesCallback=object
    )
    helpers.config_validation = _mod(
        "homeassistant.helpers.config_validation",
        config_entry_only_config_schema=lambda _domain: object(),
    )


def _check_model_dir(model_path: Path) -> list[str]:
    """Restituisce la lista dei file mancanti nella cartella del modello."""
    required = [
        model_path / "config.json",
        model_path / "generation_config.json",
        model_path / "speech_tokenizer" / "config.json",
    ]
    missing = [str(p.relative_to(model_path)) for p in required if not p.is_file()]
    if not any(
        p.name.startswith(("model", "pytorch_model")) and p.is_file()
        for p in model_path.glob("*")
    ):
        missing.append("model.safetensors (o file pesi simile)")
    return missing


def cmd_synth(args: argparse.Namespace) -> int:
    """Esegue la sintesi con il codice reale dell'integrazione."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = Path(
        args.model
        or os.environ.get("QWEN3_TTS_MODEL", "")
        or "/config/models/Qwen3-TTS-12Hz-0.6B-Base"
    ).expanduser()

    _info(f"cartella modello: {model_path}")
    if not model_path.is_dir():
        _fail(f"La cartella del modello non esiste: {model_path}")
        print("       Scarica il modello con: hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base "
              "--local-dir <cartella>")
        return 1
    missing = _check_model_dir(model_path)
    if missing:
        _fail("Cartella del modello incompleta, manca:")
        for item in missing:
            print(f"         - {item}")
        print("       Riscarica l'intero repository del modello da Hugging Face.")
        return 1
    _ok("cartella del modello completa")

    # dipendenze pesanti: numpy serve per importare tts.py
    try:
        import numpy  # noqa: F401
    except ImportError:
        _fail("numpy non installato in questo Python: pip install numpy")
        return 1

    _install_homeassistant_stubs()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    try:
        from custom_components.qwen3_tts import tts as tts_mod
    except Exception:  # noqa: BLE001
        _fail("Impossibile importare custom_components.qwen3_tts.tts:")
        import traceback

        traceback.print_exc()
        return 1
    _ok("modulo dell'integrazione importato correttamente")

    ref_audio = args.ref or str(
        REPO_ROOT / "custom_components" / "qwen3_tts" / "audio" / "claudia.wav"
    )
    if not Path(ref_audio).is_file():
        _fail(f"Audio di riferimento non trovato: {ref_audio}")
        return 1
    _info(f"audio di riferimento: {ref_audio}")

    message = args.text
    language = args.lang
    print()
    _info(f"sintesi in corso di: {message!r} (lingua {language}) ...")
    started = time.time()
    try:
        wav_bytes = tts_mod._synthesize(
            str(model_path),
            ref_audio,
            args.ref_text or "",
            message,
            language,
        )
    except ImportError as err:
        _fail(f"Dipendenza mancante: {err}")
        print("       Serve qwen-tts==0.1.1 (con transformers==4.57.3) e torch.")
        print("       Esempio: pip install qwen-tts==0.1.1")
        print("       Oppure esegui lo script con il python di un ambiente "
              "che li contiene gia'.")
        return 1
    except Exception:  # noqa: BLE001
        _fail("La sintesi e' fallita:")
        import traceback

        traceback.print_exc()
        return 1
    elapsed = time.time() - started

    out = OUTPUT_DIR / "qwen3_tts_sample.wav"
    out.write_bytes(wav_bytes)
    _ok(f"sintesi completata in {elapsed:.1f} s -> {out}")

    if not _check_wav(out):
        return 1

    print()
    print("Per ascoltare il risultato:")
    print(f"  - riproduci direttamente il file: {out}")
    print("  - oppure copialo in /config/www/ di HA e apri "
          f"http://<indirizzo-ha>:8123/local/{out.name}")
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    _print_utf8_setup()
    parser = argparse.ArgumentParser(
        description="Diagnostica audio Qwen3-TTS: tono di prova, "
                    "validazione WAV e sintesi reale.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_tone = sub.add_parser("tone", help="genera un WAV di prova (tono 440 Hz)")
    p_tone.add_argument("--seconds", type=float, default=3.0,
                        help="durata del tono in secondi (default: 3)")
    p_tone.set_defaults(func=cmd_tone)

    p_check = sub.add_parser("check", help="valida un file WAV")
    p_check.add_argument("file", help="percorso del file WAV da validare")
    p_check.set_defaults(func=cmd_check)

    p_synth = sub.add_parser("synth", help="sintesi reale con il codice "
                                           "dell'integrazione")
    p_synth.add_argument("text", nargs="?", default="Ciao, questa è una prova di sintesi.",
                         help="testo da convertire in voce")
    p_synth.add_argument("--model", help="cartella del modello Qwen3-TTS")
    p_synth.add_argument("--ref", help="audio di riferimento da clonare "
                                       "(default: claudia.wav allegata)")
    p_synth.add_argument("--ref-text", default="",
                         help="trascrizione dell'audio di riferimento (opzionale, "
                              "aumenta la fedeltà)")
    p_synth.add_argument("--lang", default="it", help="lingua (default: it)")
    p_synth.set_defaults(func=cmd_synth)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
