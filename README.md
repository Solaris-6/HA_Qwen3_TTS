# Qwen3-TTS Project 🎙️

Integrazione **moderna** per Home Assistant che supporta la sintesi vocale (Text-to-Speech) con il modello Qwen3-TTS:

## ✨ Caratteristiche Principali

✅ Setup moderno con ConfigEntry (compatibile Home Assistant 2024.11+)  
✅ Configurazione tramite interfaccia utente  
✅ Supporto per 10 lingue diverse  
✅ GPU acceleration opzionale (CUDA)  
✅ Traduzioni in italiano ed inglese  

---

## 🌍 Lingue Supportate

- **Italiano** (it) - Default
- **Inglese** (en) - Default
- **Cinese** (zh)
- **Tedesco** (de)
- **Portoghese** (pt)
- **Spagnolo** (es)
- **Giapponese** (ja)
- **Coreano** (ko)
- **Francese** (fr)
- **Russo** (ru)

---

## 📦 Installazione

### Opzione A: Tramite HACS (Consigliato)

1. Apri **HACS** nel tuo Home Assistant
2. Vai su **Integrations** → **+** → **Custom repositories**
3. Inserisci URL repository: `https://github.com/Solaris-6/Qwen3-TTS`
4. Seleziona **Integration** come tipo di repository
5. Installa e riavvia Home Assistant

### Opzione B: Installazione Manuale

```bash
# Crea directory nel config di Home Assistant
mkdir -p /config/custom_components/qwen3_tts

# Clona il repository (opzionale)
git clone https://github.com/Solaris-6/Qwen3-TTS.git /config/custom_components/qwen3_tts
```

3. Riavvia Home Assistant

4. Vai su **Impostazioni** → **Dispositivi e servizi** → **Aggiungi integrazione** → **Qwen3-TTS**

---

## ⚙️ Configurazione

### Primo Uso

1. Aggiungi l'integrazione Qwen3-TTS
2. Configura il percorso del modello: `/config/models/Qwen3-TTS-12Hz-0.6B-Base`
3. Seleziona la voce predefinita (default)
4. Scegli la lingua predefinita (italiano o inglese)

### Dispositivi Supportati

L'integrazione supporta automaticamente:

- **TTS**: Sintesi vocale in 10 lingue
- **Audio Output**: WAV format a 24kHz
- **GPU Acceleration**: CUDA opzionale per sintesi più veloce

---

## 🌐 Protocolli di Comunicazione

### Modello Qwen3-TTS
```
Modello: Qwen3-TTS-12Hz-0.6B-Base
Formato: Hugging Face Transformers
Precisione: float16 (GPU) o float32 (CPU)
Sample Rate: 24000 Hz
```

### GPU Support (Opzionale)
```
CUDA Version: 12.x+
Driver NVIDIA: 535.0+
RAM VRAM: Minimo 8GB per modello base
```

---

## 📋 Struttura Repository

```
Qwen3-TTS-Project/
├── custom_components/
│   └── qwen3_tts/
│       ├── __init__.py        # Setup entry + discovery
│       ├── config_flow.py     # Configurazione UI
│       ├── const.py           # Costanti
│       ├── manifest.json      # Metadata
│       ├── hacs.json          # HACS metadata
│       ├── tts.py             # Piattaforma TTS
│       └── translations/       # File traduzioni (opzionale)
├── README.md
├── pyproject.toml            # Build system config
├── requirements.txt           # Dipendenze runtime
└── LICENSE
```

---

## 🐛 Debug e Log

### Abilitare log dettagliati

Modifica `configuration.yaml`:
```yaml
logger:
  default: debug
  logs:
    homeassistant.components.qwen3_tts: debug
```

Riavvia Home Assistant per applicare.

---

## ⚠️ Note Importanti

### Compatibilità Home Assistant

- **Minima**: Home Assistant 2024.11.0 o superiore
- **Consigliata**: Home Assistant 2025.x o versioni successive
- **Piattaforma**: Linux, Docker, Supervised, OSM

### Requisiti Sistemistici

- Python 3.11+ (per requisiti di Home Assistant)
- RAM: Minimo 8GB per modello base
- GPU: NVIDIA con CUDA 12.x+ (opzionale ma consigliato)
- Storage: ~500MB per il modello Qwen3-TTS

---

## 🔄 Aggiornamenti e Manutenzione

### Aggiornamento HACS

1. Apri HACS → Integrations
2. Seleziona **Qwen3-TTS** dall'elenco
3. Clicca su **Update** se disponibile

### Aggiornamento Manuale

```bash
# Nel repository GitHub
cd /config/custom_components/qwen3_tts
git pull origin main

# Riavvia Home Assistant
```

---

## 📊 Supporto e Risorse

### GitHub Repository
- **Main**: https://github.com/Solaris-6/Qwen3-TTS
- **Issues**: https://github.com/Solaris-6/Qwen3-TTS/issues
- **Wiki**: Documentazione completa disponibile su GitHub

### Home Assistant Community
- Forum ufficiale: https://community.home-assistant.io/
- Discussione HACS: https://community.home-assistant.io/t/hacs-discussion-qwen3-tts-integration/
- Discord: https://discord.gg/home-assistant

---

## 📜 Licenza

Questo progetto è distribuito con licenza MIT. Vedi file `LICENSE` per dettagli.

```
MIT License

Copyright (c) 2024-2025 Solaris-6

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📞 Contatti e Contributi

Vuoi contribuire? Apri una Issue su GitHub o crea un Pull Request!

**Autore**: Solaris-6  
**Data Ultima Revisione**: 2025-11-06  
**Versione**: 2.0.0 (Corretto)

---

## 🎉 Grazie per l'uso!

Buon automazione con Home Assistant! 🏠🎙️