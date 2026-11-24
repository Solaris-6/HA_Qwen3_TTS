# Qwen3 TTS per Home Assistant

Integrazione **HACS** per Home Assistant che aggiunge un motore di sintesi vocale (Text-to-Speech) **locale e offline** basato su Qwen3-TTS, con **clonazione della voce** da un file audio di riferimento.

## Caratteristiche

- Entità TTS `tts.qwen3_tts` utilizzabile in assistente vocale, automazioni e servizi `tts.speak`
- Clonazione della voce in locale: nessuna chiamata a servizi cloud
- 10 lingue supportate (italiano incluso, predefinito)
- Supporto GPU NVIDIA (CUDA, bfloat16) automatico, altrimenti CPU (float32)
- Setup tramite Config Flow (nessuna modifica manuale di `configuration.yaml`)
- Traduzioni in italiano, inglese e altre 8 lingue
- Validazione CI automatica (HACS + hassfest)

## Requisiti

- Home Assistant **2024.12.0** o superiore
- Circa **4 GB di RAM liberi** per il modello base (8 GB consigliati)
- Circa **2,5 GB di spazio** per il modello
- GPU NVIDIA opzionale ma consigliata per risposte rapide
- Il modello deve già essere scaricato in Home Assistant prima della configurazione

## Installazione

###1. Scarica il modello (una tantum)

Scarica l'intero repository del modello (~2,5 GB) in una cartella di Home Assistant, ad esempio `/config/models/Qwen3-TTS-12Hz-0.6B-Base`.

Da un computer con [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli):

```bash
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir ./Qwen3-TTS-12Hz-0.6B-Base
```

poi copia la cartella in `/config/models/` di Home Assistant (add-on Terminal/SSH, File editor, SFTP...).

La cartella del modello **deve** contenere tra gli altri:

```text
Qwen3-TTS-12Hz-0.6B-Base/
├── config.json
├── generation_config.json
├── model.safetensors
└── speech_tokenizer/
    ├── config.json
    └── model.safetensors
```

> ⚠️ Senza la cartella `speech_tokenizer/` e `generation_config.json` l'integrazione non funziona: durante la configurazione riceverai l'errore «modello incompleto».

###2. Installa l'integrazione tramite HACS

1. Apri **HACS** → **Integrazioni** → menu (⋮) → **Repository personalizzati**
2. Inserisci l'URL `https://github.com/Solaris-6/HA_Qwen3_TTS` e scegli tipo **Integration**
3. Installa **Qwen3 TTS** e riavvia Home Assistant

Installazione manuale alternativa: copia la cartella `custom_components/qwen3_tts` dentro `/config/custom_components/` e riavvia Home Assistant.

> 💾 Al primo avvio Home Assistant installa automaticamente la dipendenza `qwen-tts==0.1.1` (PyTorch e compagni): il download può richiedere **diversi minuti**. Controlla il log se sembra bloccato.

###3. Configura l'integrazione

**Impostazioni → Dispositivi e servizi → Aggiungi integrazione → Qwen3 TTS**

| Opzione | Descrizione | Predefinito |
|---|---|---|
| Percorso del modello | Cartella del modello scaricato | `/config/models/Qwen3-TTS-12Hz-0.6B-Base` |
| Audio di riferimento | File audio (WAV/MP3) la cui voce verrà clonata | `audio/claudia.wav` inclusa |
| Trascrizione del riferimento | Testo esatto pronunciato nell'audio di riferimento (opzionale) | vuoto |
| Lingua | Lingua predefinita della sintesi | `it` |

- **Trascrizione del riferimento**: se compilata l'audio usa la modalità ICL, con maggiore fedeltà della voce; se lasciata vuota viene usata solo l'impronta vocale (funziona comunque, con risultato leggermente meno simile).
- Le opzioni sono modificabili in qualsiasi momento dalla rotella ⚙️ della scheda integrazione.

## Utilizzo

- **Prova la sintesi**: finestra «Prova text-to-speech» nelle impostazioni di assistente vocale, scegliendo l'entità `Qwen3 TTS`
- **Assistente vocale**: seleziona `Qwen3 TTS` come motore TTS nella pipeline
- **Servizi**: `tts.speak` con entità media player, o le automazioni con `service: tts.speak`

La prima generazione richiede 10–60 secondi (caricamento del modello in RAM); in seguito il modello resta in memoria e le generazioni successive sono più rapide.

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

# 4) Valida un file WAV qualsiasi
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

Riavvia Home Assistant. I log sono consultabili in **Impostazioni → Sistema → Log**, filtrando per `qwen3_tts`.

### Problemi comuni

| Sintomo | Causa probabile |
|---|---|
| «Error playing audio» | Guarda il log: di solito mancano i requisiti `qwen-tts` (primo avvio non completato) o la cartella del modello è incompleta |
| Errore «modello incompleto» in configurazione | Mancano `speech_tokenizer/` o `generation_config.json`: riscarica l'intero repository |
| Errore «audio di riferimento non esiste» | Il percorso in `ref_audio` non è valido |
| Primo avvio lentissimo | Normale: pip sta installando PyTorch e dipendenze |

## Struttura del repository

```text
HA_Qwen3_TTS/
├── .github/workflows/          # CI: validazione HACS + hassfest
├── brand/icon.png              # Icona dell'integrazione (per HACS/brands)
├── custom_components/
│   └── qwen3_tts/
│       ├── __init__.py         # Setup della config entry
│       ├── audio/              # Voci di riferimento incluse (claudia.wav)
│       ├── config_flow.py      # Configurazione da interfaccia
│       ├── const.py            # Costanti
│       ├── manifest.json       # Metadati + requisiti Python
│       ├── services.yaml       # Documentazione servizi
│       ├── strings.json        # Stringhe di origine (inglese)
│       ├── tts.py              # Piattaforma TTS
│       └── translations/       # 10 lingue
├── hacs.json                   # Metadati HACS
├── LICENSE                     # MIT
├── README.md
├── requirements.txt
└── tools/                      # Script di diagnostica audio (test_audio.py)
```

## Aggiornamenti

- **HACS**: Integrazioni → **Aggiorna** se disponibile → riavvia Home Assistant
- **Manuale**: `git pull` nella cartella dell'integrazione → riavvia Home Assistant

## Licenza

Progetto distribuito con licenza **MIT**. Vedi il file [`LICENSE`](LICENSE).

---

**Autore**: Solaris-6
