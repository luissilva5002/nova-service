# NOVA — Personal AI Assistant

Single-brain, local-first personal assistant: one 3B LLM (Qwen2.5-3B or
Llama-3.2-3B) as the "brain", whisper.cpp as the "ears", Piper TTS as the
"mouth", SQLite + ChromaDB as dual-layer memory, and a pluggable
`skills/` directory for integrations (YouTube Music, Flutter workspace
inspection, etc). Runs entirely in one Docker container, on Windows
(via WSL2) or Linux, capped at ~6GB RAM for a Ryzen 3 / 8GB box.

## 1. Prerequisites

- **Windows:** Docker Desktop with WSL2 backend enabled.
- **Linux:** Docker Engine + Docker Compose plugin (`sudo apt install docker.io docker-compose-plugin`).
- `bash`/Git-Bash or WSL on Windows, to run the model-download script.
- ~3GB free disk for model weights (not included in this zip — see step 2).

## 2. Download the models

Models are **not** baked into the Docker image (keeps the image small and
avoids re-downloading gigabytes every rebuild). They're host-mounted via
`./models`. Run once:

```bash
bash scripts/download_models.sh
```

This fetches into `models/`:
- `brain/qwen2.5-3b-instruct-q4_k_m.gguf` (~2.2GB) — swap for a Llama 3.2 3B
  GGUF if you prefer that brain (just update `NOVA_LLM_MODEL_PATH`)
- `whisper/ggml-base.en.bin` (~150MB)
- `piper/en_US-lessac-medium.onnx` + `.json` (~60MB) — browse more voices at
  https://rhasspy.github.io/piper-samples/

If the download step fails on your network, download manually from the
URLs in `scripts/download_models.sh` and place the files in the same
paths — nothing else needs to change.

## 3. Configure your projects folder (optional, for the Flutter/codebase skill)

Copy `.env.example` to `.env` and point `NOVA_HOST_PROJECTS_PATH` at the
folder that contains the project folders you want NOVA to search/read
(e.g. your Flutter apps):

```bash
cp .env.example .env
# then edit .env, e.g.:
# NOVA_HOST_PROJECTS_PATH=C:/Users/Luis/dev        (Windows)
# NOVA_HOST_PROJECTS_PATH=/home/luis/dev           (Linux)
```

This is mounted **read-only** into the container at `/app/host_projects`.

## 4. Build & run

From the project root (same folder as `docker-compose.yml`):

```bash
docker compose build
docker compose up -d
```

Check it booted correctly:

```bash
docker compose logs -f
```

You should see NOVA log the models it loaded (or "STUB mode" warnings if
a model file is missing — the server still boots and is testable without
models, it just echoes input instead of doing real inference).

Open the dashboard: **http://localhost:8000**
WebSocket endpoint (used by the Flutter app too): **ws://<server-ip>:8000/ws/chat**

## Performance tuning

The CPU-oriented defaults use four llama.cpp threads, a 512-token batch, a
4096-token context, and a 512 MB prompt KV cache. These can be adjusted with
`NOVA_LLM_THREADS`, `NOVA_LLM_THREADS_BATCH`, `NOVA_LLM_BATCH`,
`NOVA_LLM_CTX`, `NOVA_LLM_FLASH_ATTN`, `NOVA_LLM_MLOCK`,
`NOVA_LLM_KV_CACHE_TYPE`, `NOVA_LLM_CACHE_MB`, `NOVA_LLM_TEMP_TOOLS`,
`NOVA_LLM_MAX_TOKENS_CHAT`, `NOVA_LLM_MAX_TOKENS_TOOL`, and
`NOVA_LLM_CONFIRMATIONS`; see `.env.example` for defaults. Streaming is
opt-in per WebSocket text message with `"stream": true`.

After upgrading the host from 8 GB to 12 GB RAM, raise `NOVA_MEM_LIMIT` to
approximately `9g` or `10g` if the host has enough headroom.

NOVA defaults to offline Hugging Face mode. Prefetch complete local model
directories into `models/minilm`, `models/whisper`, and `models/hf_cache`.
Google Calendar, YouTube Music, and the port 8080 OAuth callback are the
components that legitimately require internet access.

## 5. Everyday commands

| Action | Command |
|---|---|
| Build image | `docker compose build` |
| Start in background (auto-restarts on reboot) | `docker compose up -d` |
| Stream live logs | `docker compose logs -f` |
| Stop | `docker compose down` |
| Rebuild after code changes | `docker compose up -d --build` |
| Open a shell inside the container | `docker compose exec nova_server bash` |

Enable Docker to auto-start on Linux boot (only needed once):

```bash
sudo systemctl enable docker
```

`restart: unless-stopped` in `docker-compose.yml` then relaunches NOVA
automatically whenever the Docker daemon starts.

## 6. The Windows → Linux server workflow

This is fully supported and is how you should develop:

1. **Develop on Windows.** Run Docker Desktop (WSL2), build/test at
   `http://localhost:8000`, iterate on the web UI and skills.
2. **Transfer via USB.** Copy the whole `nova_system/` project folder
   (including the now-downloaded `models/` folder — this saves you from
   re-downloading gigabytes on the server) onto a USB drive.
3. **Deploy on Linux.** Plug the USB into your server, copy the folder to
   e.g. `/opt/nova` or `~/nova`, then:
   ```bash
   cd /opt/nova
   docker compose up -d --build
   ```

**Line-ending gotcha:** if you create any shell scripts, save them with
LF (Linux) line endings rather than CRLF (Windows), or just rely
entirely on `docker compose` commands, which handle this automatically
for you (the Dockerfile/compose files don't care about your editor's
line endings).

## 7. Project layout

```
nova_system/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── models/              # host-mounted GGUF/ONNX weights (gitignored)
├── data/                # host-mounted SQLite + ChromaDB storage (gitignored)
├── host_projects/        # host-mounted, read-only, your source projects
├── scripts/
│   └── download_models.sh
├── nova/                 # backend brain & router
│   ├── main.py            # FastAPI app, WebSocket pipeline
│   ├── config.py
│   ├── brain/
│   │   ├── llm_engine.py    # llama.cpp bindings + tool calling
│   │   ├── stt_engine.py    # whisper.cpp + filler-word stripping
│   │   ├── tts_engine.py    # Piper streaming synthesis
│   │   └── tool_router.py   # schema aggregation + skill dispatch
│   ├── memory/
│   │   ├── core_store.py    # SQLite: user facts, project profiles
│   │   ├── vector_store.py  # ChromaDB: project/code embeddings
│   │   └── memory_agent.py  # async background fact extraction
│   ├── ingestion/
│   │   ├── project_scanner.py  # directory tree builder
│   │   ├── llm_filter.py       # LLM-classified ignore rules (cached)
│   │   └── pipeline.py         # orchestrates scan -> filter -> embed
│   └── skills/
│       ├── base_skill.py
│       ├── youtube_music/      # example skill: tools.py + handler.py
│       └── flutter_workspace/  # example skill: codebase search/read
└── web_ui/                # dashboard: index.html, style.css, app.js
```

## 8. Known gaps to close next (flagged honestly, not swept under the rug)

- **Browser mic format:** `web_ui/app.js` records `audio/webm` (Opus) via
  `MediaRecorder`, but `nova/brain/stt_engine.py` expects raw PCM16 for
  whisper.cpp. You'll need an `ffmpeg`-based conversion step (webm → PCM16
  mono 16kHz) in the WebSocket handler before transcription — `ffmpeg` is
  already installed in the Docker image for this. This is a decode step,
  not an architecture change.
- **`ytm_play` / `ytm_skip` / etc. in `skills/youtube_music/handler.py`**
  are stubs with `# TODO` markers — wire up `ytmusicapi` or your device
  bridge of choice.
- **Piper sample rate in `app.js`** is hardcoded to 22050Hz for the WAV
  header on playback — confirm this matches whichever voice's `.onnx.json`
  you actually use (the `sample_rate` field), or read it dynamically.
- **No auth** on the WebSocket/REST endpoints yet — fine on a private LAN,
  but add a token check before exposing port 8000 beyond your home network.
- **llama-cpp-python tool-calling format** varies slightly by version;
  `generate_with_tools()` targets the OpenAI-style `tools=[...]` API — pin
  a known-good version in `requirements.txt` if you hit schema errors.

This is a working skeleton end-to-end (it boots and serves the UI even
without models present, via STUB mode), not a finished product — exactly
where you wanted to start debugging and iterating.
