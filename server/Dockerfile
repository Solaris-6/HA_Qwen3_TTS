# Qwen3-TTS server used by the Home Assistant integration in server mode.
#
# The image is Debian based (glibc), so PyTorch wheels install normally:
# this is exactly what Home Assistant OS/Container cannot do.
FROM python:3.12-slim-bookworm

WORKDIR /app

# Installing qwen-tts also pulls torch/torchaudio; the PyTorch CPU index keeps
# the image small because pip prefers the "+cpu" builds.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cpu

COPY qwen3_tts_server.py ./

EXPOSE 8000

# /health answers as soon as the HTTP server is up, even while the model is
# still loading, so the check below only proves the process is alive.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["python", "-u", "qwen3_tts_server.py", "--host", "0.0.0.0", "--port", "8000", "--preload"]
