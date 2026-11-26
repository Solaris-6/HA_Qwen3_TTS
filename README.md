# Qwen3 TTS per Home Assistant

Integrazione **HACS** per Home Assistant che aggiunge un motore di sintesi vocale (Text-to-Speech) basato su **Qwen3-TTS**, con **clonazione della voce** da un file audio di riferimento.

Il modello può girare in due modi:

- **🌐 Server di rete (consigliato)**: il modello gira su un server della tua rete (Raspberry Pi, Mini PC, NAS, PC) e Home Assistant gli invia solo il testo. Funziona su **qualsiasi** Home Assistant.
- **🏠 Locale**: il modello gira dentro Home Assistant, ma solo se HA riesce ad installare PyTorch (vedi sotto).

## Perché serve un server

> ⚠️ Le immagini **ufficiali** di Home Assistant (OS, Container, Supervised) girano su **Alpine Linux/musl**, mentre PyTorch pubblica soltanto wheel per **glibc**. Dentro Home Assistant ufficiale, `pip install qwen-tts` fallisce sempre con errori del tipo `No matching distribution found for torchaudio`.
>
> La modalità server risolve il problema: un piccolo processo standalone (una riga di comando, oppure un container Docker) carica il modello e lo espone su HTTP. Home Assistant non installa **nessuna** dipendenza pesante.

## Caratteristiche

- Entità TTS `tts.qwen3_tts` utilizzabile in assistente vocale, automazioni e servizi `tts.speak`
- Clonazione della voce: nessuna chiamata a servizi cloud
- 10 lingue supportate (italiano incluso, predefinito)
- Server standalone in Python (solo dipendenze di `qwen-tts`), con API `/health` e `/synthesize`
- GPU NVIDIA (CUDA, bfloat16) automatica, altrimenti CPU (float32)
- Setup tramite Config Flow (nessuna modifica manuale di `configuration.yaml`)
- Traduzioni in italiano, inglese e altre 8 lingue
- Validazione CI automatica (HACS + hassfest)

## Requisiti

- Home Assistant **2024.12.0** o superiore
- Per la modalità **server** (consigliata):
  - un computer raggiungibile in rete (va bene una Raspberry Pi 4/5, un Mini PC, un NAS o il tuo PC)
  - **4 GB di RAM liberi** per il modello base (8 GB consigliati)
  - circa **3 GB di spazio** per il modello e le dipendenze
  - Python 3.10–3.13 oppure Docker
  - GPU opzionale ma consigliata (senza GPU ogni frase richiede decine di secondi)
- Per la modalità **locale**: la stessa macchina di Home Assistant deve poter installare PyTorch (Home Assistant Core in un venv su Debian/Ubuntu/Windows). **Non funziona** su HA OS / Container.

## Installazione

### 1. Installa l'integrazione tramite HACS

1. Apri **HACS** → **Integrazioni** → menu (⋮) → **Repository personalizzati**
2. Inserisci l'URL `https://github.com/Solaris-6/HA_Qwen3_TTS` e scegli tipo **Integration**
3. Installa **Qwen3 TTS** e riavvia Home Assistant

Installazione manuale alternativa: copia la cartella `custom_components/qwen3_tts` dentro `/config/custom_components/` e riavvia Home Assistant.

### 2a. Avvia il server Qwen3-TTS (consigliato)

#### La tua rete (Home Assistant su Raspberry Pi 5 a `tuo IP`, server su Raspberry Pi 4+ a `tuo IP`)

Il modello (`Qwen3-TTS-12Hz-0.6B-Base`) gira interamente sul **Raspberry Pi 4**: Home Assistant (sul Pi 5) non installa nulla di pesante e si limita a inviare il testo. Il Pi 4 deve avere **almeno 4 GB di RAM** (8 GB consigliati); con 4 GB aggiungi una swap di 2 GB per non avere interruzioni.

Sul Pi 4, da un terminale (SSH o schermo):

```bash
# 1) dipendenze di sistema
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential \
    libsndfile1 sox ffmpeg

# 2) crea il venv
mkdir -p ~/qwen3-tts && cd ~/qwen3-tts
python3 -m venv .venv
source .venv/bin/activate

# 3) dipendenze Python (torch CPU per aarch64)
pip install --upgrade pip
pip install -r /percorso/del/repo/server/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 4) scarica il modello (~2,5 GB) sul Pi 4
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --local-dir ~/qwen3-tts/models/Qwen3-TTS-12Hz-0.6B-Base

# 5) avvia il server (resta in primo piano)
python -u /percorso/del/repo/server/qwen3_tts_server.py \
    --model ~/qwen3-tts/models/Qwen3-TTS-12Hz-0.6B-Base \
    --preload
```

Per farlo ripartire automaticamente dopo un riavvio del Pi 4, crea `/etc/systemd/system/qwen3-tts.service`:

```ini
[Unit]
Description=Qwen3-TTS server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/qwen3-tts
ExecStart=/home/pi/qwen3-tts/.venv/bin/python -u \
    /home/pi/qwen3-tts/server/qwen3_tts_server.py \
    --model /home/pi/qwen3-tts/models/Qwen3-TTS-12Hz-0.6B-Base \
    --preload
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now qwen3-tts
sudo systemctl status qwen3-tts
journalctl -u qwen3-tts -f   # log in tempo reale
```

Verifica dal Pi 5 (o da qualsiasi PC della rete):

```bash
python tools/test_server.py --url http://tuo IP:8000
```

> ⚠️ Il Pi 4 è **solo CPU**: ogni frase di 10 parole richiede circa 60–180 s. Il caricamento del modello all'avvio richiede circa 60 s (anticipato da `--preload`).

#### Installazione generica (PC / NAS / Mini PC)

Sul computer che ospiterà il modello (es. `tuo IP`):"oldString">
### 2a. Avvia il server Qwen3-TTS (consigliato)

Sul computer che ospiterà il modello (es. `tuo IP`):

```bash
git clone https://github.com/Solaris-6/HA_Qwen3_TTS
cd HA_Qwen3_TTS
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate

# dipendenze (torch CPU, ~700 MB)
pip install -r server/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# modello (~2,5 GB): scaricalo da Hugging Face...
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir ./models/Qwen3-TTS-12Hz-0.6B-Base

# avvia il server (resta in primo piano, log su stdout)
python -u server/qwen3_tts_server.py --model ./models/Qwen3-TTS-12Hz-0.6B-Base --preload
```

In alternativa con **Docker** (utile su NAS e Raspberry Pi):

```bash
cd server
mkdir -p models
# copia il modello dentro server/models/Qwen3-TTS-12Hz-0.6B-Base
docker compose up -d --build
```

Verifica che il server risponda:

```bash
python tools/test_server.py --url http://tuo IP:8000
```

Argomenti utili di `server/qwen3_tts_server.py`:

| Argomento | Descrizione |
|---|---|
| `--host` / `--port` | Indirizzo e porta (default `0.0.0.0:8000`) |
| `--model` | Cartella locale **oppure** repo Hugging Face (scaricata al primo avvio) |
| `--device` | `auto` (default), `cpu` oppure `cuda` |
| `--preload` | Carica il modello già all'avvio |
| `--default-ref` | Audio di riferimento usato se la richiesta non ne include uno |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, ... |

> 🔒 Il server non ha autenticazione: usalo solo nella tua rete locale e non esporlo su internet.

### 2b. Alternativa: modello dentro Home Assistant (solo HA Core)

Se il tuo Home Assistant può installare pacchetti Python (HA Core in venv, non HA OS/Container):

1. Scarica il modello in `/config/models/Qwen3-TTS-12Hz-0.6B-Base` (con `hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir ...`)
2. Durante la configurazione **lascia vuoto** il campo del server: l'integrazione installerà `qwen-tts` (circa **3 GB**, diversi minuti) al primo setup della scheda
3. Se l'installazione fallisce, in **Impostazioni → Sistema → Log** trovi il dettaglio cercando `qwen3_tts.deps`

### 3. Configura l'integrazione

**Impostazioni → Dispositivi e servizi → Aggiungi integrazione → Qwen3 TTS**

- **Primo passo**: inserisci l'indirizzo del server, ad esempio `http://tuo IP:8000`. Lascia il campo **vuoto** per la modalità locale. L'integrazione verifica subito che il server risponda.
- **Secondo passo**:

| Opzione | Descrizione | Predefinito |
|---|---|---|
| Audio di riferimento | File audio (WAV/MP3) la cui voce verrà clonata | `audio/claudia.wav` inclusa |
| Trascrizione del riferimento | Testo esatto pronunciato nell'audio di riferimento (opzionale) | vuoto |
| Lingua | Lingua predefinita della sintesi | `it` |

- **Trascrizione del riferimento**: se compilata l'audio usa la modalità ICL, con maggiore fedeltà della voce; se lasciata vuota viene usata solo l'impronta vocale (funziona comunque, con risultato leggermente meno simile).
- In modalità locale compare anche il **percorso del modello** (`/config/models/Qwen3-TTS-12Hz-0.6B-Base`).
- Le opzioni sono modificabili in qualsiasi momento dalla rotella ⚙️ della scheda integrazione.

## Utilizzo

- **Prova la sintesi**: finestra «Prova text-to-speech» nelle impostazioni di assistente vocale, scegliendo l'entità `Qwen3 TTS`
- **Assistente vocale**: seleziona `Qwen3 TTS` come motore TTS nella pipeline
- **Servizi**: `tts.speak` con entità media player, o le automazioni con `service: tts.speak`

Tempi indicativi per una frase di ~10 parole:

| Ambiente | Tempo |
|---|---|
| GPU (CUDA) | 5–15 s |
| CPU desktop | 20–60 s |
| Raspberry Pi (CPU) | 60–180 s |

La **prima** richiesta richiede in più il caricamento del modello (30–60 s): usa `--preload` per anticiparlo.

### Verifica della riproduzione audio

Lo script `tools/test_audio.py` aiuta a capire se il problema è la **sintesi** o la **riproduzione**:

```bash
# 1) Genera un file WAV di prova (tono) e validalo
python tools/test_audio.py tone

# 2) Copia tools/output/test_tone.wav in /config/www/ di Home Assistant,
#    poi apri in un browser: http://<ha>:8123/local/test_tone.wav
#    → Se suona: la riproduzione funziona, il problema è la sintesi
#    → Se non suona: il problema è HA/browser, non l'integrazione

# 3) Sintesi reale con il codice dell'integrazione (serve qwen-tts installato)
python tools/test_audio.py synth "Ciao, questa è una prova di sintesi." --model /percorso/del/modello

# 4) Verifica il server (solo standard library)
python tools/test_server.py --url http://tuo IP:8000

# 5) Valida un file WAV qualsiasi
python tools/test_audio.py check tools/output/test_tone.wav
```

## Log e debug

Per attivare il log dettagliato, modifica `configuration.yaml`:

```yaml
logger:
  default: info
  logs:
    custom_components.qwen3_tts: debug
    qwen_tts: debug
```

Riavvia Home Assistant. I log sono consultabili in **Impostazioni → Sistema → Log**, filtrando per `qwen3_tts`. Sul server, i log della sintesi sono su stdout (o `docker logs qwen3-tts-server`).

### Problemi comuni

| Sintomo | Causa probabile |
|---|---|
| «Impossibile caricare il flusso di configurazione: 500 Internal Server Error» | Versione dell'integrazione vecchia (< 1.1.1) che installava `qwen-tts` mentre apriva il flusso: aggiorna |
| Il config flow rifiuta l'indirizzo del server (`cannot_connect`) | Server spento, porta sbagliata o indirizzo di un altro servizio: verifica con `python tools/test_server.py --url ...` |
| «The Qwen3-TTS server ... is not reachable» nel log | Il server si è spento dopo la configurazione: riavvialo, poi ricarica l'integrazione |
| Scheda bloccata in «Setup retry» (solo modalità locale) | Installazione di `qwen-tts` non riuscita: nel log cerca `qwen3_tts.deps` |
| «Error playing audio» | Guarda il log: modello incompleto, audio di riferimento mancante o server non raggiungibile |
| Sintesi lentissima sul server | Normale su CPU/Raspberry Pi: valuta una GPU o accetta i tempi nella tabella sopra |
| Errore «modello incompleto» in configurazione | Mancano `speech_tokenizer/` o `generation_config.json`: riscarica l'intero repository |
| Errore «audio di riferimento non esiste» | Il percorso in `ref_audio` non è valido |

## Struttura del repository

```text
HA_Qwen3_TTS/
├── .github/workflows/          # CI: validazione HACS + hassfest
├── brand/icon.png              # Icona dell'integrazione (per HACS/brands)
├── custom_components/
│   └── qwen3_tts/
│       ├── __init__.py         # Setup della config entry
│       ├── audio/              # Voci di riferimento incluse (claudia.wav)
│       ├── config_flow.py      # Configurazione da interfaccia (server o locale)
│       ├── const.py            # Costanti
│       ├── deps.py             # Installazione di qwen-tts (solo modalità locale)
│       ├── manifest.json       # Metadati (requirements vuoti)
│       ├── services.yaml       # Documentazione servizi
│       ├── strings.json        # Stringhe di origine (inglese)
│       ├── tts.py              # Piattaforma TTS (client del server o motore locale)
│       └── translations/       # 10 lingue
├── server/                     # Server standalone per la modalità di rete
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── qwen3_tts_server.py     # API HTTP: /health e /synthesize
│   └── requirements.txt
├── hacs.json                   # Metadati HACS
├── LICENSE                     # MIT
├── README.md
├── requirements.txt
└── tools/                      # Diagnostica (test_audio.py, test_server.py)
```

## Aggiornamenti

- **HACS**: Integrazioni → **Aggiorna** se disponibile → riavvia Home Assistant
- **Manuale**: `git pull` nella cartella dell'integrazione → riavvia Home Assistant

## Licenza

Progetto distribuito con licenza **MIT**. Vedi il file [`LICENSE`](LICENSE).

---

**Autore**: Solaris-6
