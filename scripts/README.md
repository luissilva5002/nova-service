# Script guide

Run commands from the repository root unless noted. Host scripts are grouped
here by purpose; filenames and paths remain stable.

## First-time setup and Docker

| Script | Platform | What it does / command |
|---|---|---|
| [`download_models.ps1`](./download_models.ps1) | Windows PowerShell | Downloads the base Qwen2.5-3B GGUF, legacy whisper.cpp weight, and Piper voice: `.\scripts\download_models.ps1`. |
| [`download_models.sh`](./download_models.sh) | macOS, Linux, or WSL | Bash equivalent of the base model/voice downloader: `bash scripts/download_models.sh`. |
| [`download_brain_models.sh`](./download_brain_models.sh) | macOS/Linux/WSL Bash | Downloads selectable GGUF brain models: `bash scripts/download_brain_models.sh [qwen2.5-3b|qwen3-1.7b|qwen3-0.6b|all]`. |
| [`start_nova.sh`](./start_nova.sh) | Linux/WSL Bash | Builds and starts the Docker Compose app, then follows logs. Prefer the cross-platform Docker Compose commands in the root README when using PowerShell or macOS. |
| [`check_compose.sh`](./check_compose.sh) | Linux/WSL Bash | Validates Compose configuration with `docker compose config -q`. On any OS, run that Docker command directly if Bash is unavailable. |

The base model downloaders do not fetch complete faster-whisper or local
MiniLM model directories. The starter `.env.example` allows their initial
online download. Both downloaders skip non-empty files and leave the previous
destination untouched if a download fails. See the root README before
switching NOVA to offline mode.

## Maintenance and integrations

| Script | Requirements | What it does / command |
|---|---|---|
| [`get_google_token.py`](./get_google_token.py) | Host Python with project requirements; Google OAuth client ID and secret in environment | Interactive OAuth flow for Google Calendar; run `python scripts/get_google_token.py`. |
| [`migrate_knowledge_to_vault.py`](./migrate_knowledge_to_vault.py) | Host Python with project requirements; configured WebObsidian API URL/key; source Markdown notes | Copies local knowledge notes to WebObsidian without deleting the originals: `python scripts/migrate_knowledge_to_vault.py`. |
| [`train_classifier.sh`](./train_classifier.sh) | Linux/WSL Bash, Python, and classifier dependencies | Trains the intent classifier in a native WSL working directory: `bash scripts/train_classifier.sh`. |

## Diagnostics and model experiments

These are developer tools, not required for a normal install.

| Script | Requirements | What it does / command |
|---|---|---|
| [`diag_prompt.py`](./diag_prompt.py) | Host Python and project dependencies | Prints prompt component diagnostics: `python scripts/diag_prompt.py`. |
| [`collect_diagnostics.sh`](./collect_diagnostics.sh) | Linux/WSL Bash and Docker Compose | Collects machine/container diagnostics into a local text file. Review and redact output before sharing. |
| [`bench_llm.py`](./bench_llm.py) | Host Python with `llama-cpp-python`; a GGUF path | Runs local generation benchmarks: `python scripts/bench_llm.py models/brain/<model>.gguf`. It records timing metrics. |
| [`kv_baseline.py`](./kv_baseline.py) | Host Python with project dependencies; exact configured GGUF | Measures KV-prefix behavior through the normal `LLMEngine` call path. See its help/source before use; it records timing diagnostics. |
| [`kv_toolcost.py`](./kv_toolcost.py) | Host Python with `llama-cpp-python`; exact configured Qwen2.5-3B GGUF | Counts system-block tokens for tools using the GGUF vocabulary/template: `python scripts/kv_toolcost.py`. |
| [`kv_generate_probe.py`](./kv_generate_probe.py) | Host Python with `llama-cpp-python`; exact configured Qwen2.5-3B GGUF | Compares token-id generation prefix reuse with and without `LlamaRAMCache`: `python scripts/kv_generate_probe.py`. |

## Path and environment notes

- The Python diagnostics add the repository root to `sys.path`; run them with
  the Python environment in which the project dependencies are installed.
- Model experiment scripts read configured model paths from `nova.config`
  unless their usage explicitly takes a model path argument.
- `.env`, API keys, OAuth files, models, and generated diagnostic output may
  contain private data. Do not commit or share them without review.
