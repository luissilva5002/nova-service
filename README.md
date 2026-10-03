# NOVA AI

NOVA is a local-first personal assistant with a browser UI, a local GGUF
language model, speech input/output, persistent memory, and optional skills.
Docker Compose runs the service on Windows, macOS, or Linux; you do not need
to install the Python application dependencies on your host.

## Quick start from a clone

### 1. Install prerequisites

- Git.
- Docker Desktop on Windows/macOS, or Docker Engine and the Docker Compose
  plugin on Linux. Start Docker before continuing.
- An internet connection for the initial model downloads.
- Several GB of free disk space for model files and caches. The default
  container memory limit is 6 GB; larger models or workloads may need more.
- NOVA currently runs as a **Linux container** on all host operating systems.
  On Windows and macOS, install Docker Desktop and use Linux containers. Native
  Windows containers are not supported.

### 2. Clone the repository

Run these commands in PowerShell, Terminal, or a Linux shell:

```text
git clone https://github.com/luissilva5002/nova-service.git
cd nova-service
```

### 3. Create the local environment file

Copy the example file using the shell you are using:

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```
**macOS, Linux, or WSL**

```sh
cp .env.example .env
```

The example selects the Qwen2.5-3B model downloaded in the next step and
allows first-run downloads for the speech and intent-classifier models. Review
`.env` before starting. It is ignored by Git and is the place for local paths
and optional API keys; never commit it.

### 4. Download the base model and voice files

From the repository root, use the native script for your operating system:

**Windows PowerShell**

```powershell
.\scripts\download_models.ps1
```

**macOS, Linux, or WSL**

```sh
bash scripts/download_models.sh
```

These scripts download Qwen2.5-3B, a legacy `whisper.cpp` model file, and the
Piper voice into `models/`. NOVA currently uses `faster-whisper`, not the
downloaded `whisper.cpp` file. The example `.env` points faster-whisper at a
separate model directory and sets `NOVA_OFFLINE=0`, so the first container
startup can fetch the compatible faster-whisper and intent-classifier
artifacts. Keep network access enabled for that first startup. For a strictly
offline install, prepare the complete model directories expected by the
application first; the base downloader alone does not prepare them.
Both downloaders skip existing non-empty files and only replace a destination
after a successful, non-empty download.

### 5. Create Compose's shared network

The Compose file uses an external network named `nova-net` so NOVA can
optionally connect to services such as WebObsidian. Create it once per Docker
installation. This step is only needed when using Compose as documented; if
you already have a `nova-net` Docker network, keep using it.

**PowerShell**

```powershell
docker network inspect nova-net *> $null
if ($LASTEXITCODE -ne 0) { docker network create nova-net }
```

**macOS/Linux/WSL**

```sh
docker network inspect nova-net >/dev/null 2>&1 || docker network create nova-net
```

### 6. Build and start NOVA

From the repository root:

```text
docker compose config -q
docker compose up --build -d
docker compose logs -f nova_server
```

Wait for the startup logs, then open **http://localhost:8000**. The API status
endpoint is **http://localhost:8000/api/status**. Stop following logs with
Ctrl+C; the container continues running.

If the status reports a missing model or a component in stub mode, check
`.env`, confirm the corresponding files exist under `models/`, and inspect
`docker compose logs nova_server`.

## Configure your installation

### Local projects (optional)

`NOVA_HOST_PROJECTS_PATH` in `.env` points to the directory containing project
folders that the workspace skill may inspect. The directory is mounted
read-only in the container.

- Windows: use a Docker Desktop-visible path such as
  `C:/Users/you/dev`.
- macOS: use a path such as `/Users/you/dev`.
- Linux: use a path such as `/home/you/dev`.

The default `./host_projects` refers to the repository's own `host_projects`
folder.

### Models and offline operation

`NOVA_LLM_MODEL` selects one of the model IDs in `nova/config.py`. The general
model downloader provides `qwen2.5-3b`; choose another ID only after placing
that model's GGUF under `models/brain/`. The app's code default is `qwen3-1.7b`,
so the provided `.env.example` explicitly selects `qwen2.5-3b`.

For offline operation, set `NOVA_OFFLINE=1` only after preparing the complete
local faster-whisper and MiniLM model assets at the configured paths. The
`.env.example` uses `NOVA_OFFLINE=0` for first-run downloads. The file
`models/whisper/ggml-base.en.bin` downloaded by the base script is a
whisper.cpp model and is not a substitute for faster-whisper's CTranslate2
model directory.

### Optional integrations

- **WebObsidian memory:** set `AGENT_API_KEY` in `.env` using an API key with
  read, write, and search scopes. The default API URL expects a service named
  `webobsidian` on the shared Docker network; set `AGENT_API_BASE_URL` if it
  is elsewhere. `scripts/migrate_knowledge_to_vault.py` can import existing
  local Markdown notes when configured.
- **Google Calendar:** follow the Google OAuth setup for your Google Cloud
  credentials, then use `scripts/get_google_token.py` to save a token. Keep
  credentials and token files local and out of Git.
- **YouTube Music:** the skill is optional and may need account/device
  configuration beyond the base setup.

NOVA has no authentication on its HTTP or WebSocket endpoints. Keep ports
8000 and 8080 on a trusted network; do not expose the service directly to the
public internet.

## Platform support

The supported deployment path is Docker's Linux-container mode on Windows,
macOS, and Linux. The container build does not force x86-only CPU instructions,
so it can target both `linux/amd64` and `linux/arm64` (including Apple Silicon
through Docker Desktop). On ARM Linux, use a 64-bit ARM64 Docker host. CPU-only
inference is the default; performance depends on the host and model.

The repository's GitHub Actions checks Compose configuration and downloader
scripts on Windows, macOS, and Linux. It builds the Linux container for both
supported architectures, checks application imports on each, and starts an
API smoke test on `linux/amd64`. These checks cover the project's supported
deployment path; they do not certify every OS release, Docker version, CPU, or
optional integration. Report the host OS/architecture and Docker version with
environment-specific problems.

## Everyday commands

Run from the repository root:

| Action | Command |
|---|---|
| Check Compose configuration | `docker compose config -q` |
| Start | `docker compose up -d` |
| Follow logs | `docker compose logs -f nova_server` |
| Stop | `docker compose down` |
| Rebuild and restart after code changes | `docker compose up --build -d` |
| Open a shell in the service container | `docker compose exec nova_server bash` |

To update an existing clone, run `git pull` and then
`docker compose up --build -d`. Model and persistent data directories are
host-mounted and are not included in the Git repository.

## Script index

See [`scripts/README.md`](scripts/README.md) for a categorized list of
download, startup, maintenance, diagnostic, and KV-cache scripts. Existing
script paths are kept stable so commands and links do not need to change.

## Repository layout

```text
nova/          Application code: API, engine, memory, skills, ingestion
web_ui/        Browser user interface
models/        Local model weights and caches (not committed)
data/          Local databases and runtime data (not committed)
persistent_memory/
               Local notes and credentials (not committed)
host_projects/ Optional read-only project mount
scripts/       Setup and developer/maintenance scripts
docs/          Protocol, diagnostics, and implementation notes
```
