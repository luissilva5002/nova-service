# NOVA CPU/Offline Streaming Audit

## A) Audit table (section 0)

| Feature | Status | Location |
|---|---|---|
| LlamaRAMCache/set_cache | present | `nova/brain/llm_engine.py:225-234` |
| `stream=True` path | partial | `nova/brain/llm_engine.py:313-383` |
| `asyncio.to_thread` + engine lock | present | `nova/brain/llm_engine.py:269-274` |
| prompt/TTFT metrics | partial | `nova/brain/llm_engine.py:145-171` |
| static system/dynamic last-user prompt | present | `nova/brain/llm_engine.py:399-407` |
| history as real chat messages | present | `nova/brain/llm_engine.py:400-404` |
| `tools=None` when empty | present | `nova/brain/llm_engine.py:259-267` |
| forced single-tool choice | present | `nova/brain/llm_engine.py:263-266` |
| audio/open_url second call | present, opt-in | `nova/main.py:546-571` |
| flash attention/batch/mlock | present | `nova/config.py:64-75`, `nova/brain/llm_engine.py:190-220` |
| model close on selection | present | `nova/brain/llm_engine.py:67-70` |
| Qwen thinking regex | present | `nova/brain/llm_engine.py:96` |
| WS `text_delta`/audio chunks | partial, legacy audio only | `nova/main.py:650-714` |
| `app.js` streaming handling | missing | `web_ui/app.js:64-87` |
| MiniLM local path loading | missing | `nova/brain/intent_classifier.py:62` |
| faster-whisper local model dir | partial | `nova/brain/stt_engine.py:57-70` |
| Chroma embedding/telemetry control | missing | `nova/memory/vector_store.py:35` |
| HF offline environment | missing | `nova/config.py:1-10` |

## B) Scope

The remaining sections implement the missing/partial items without changing
the public pipeline return shape or legacy WebSocket frames.

## B) Commits and files changed

Follow-up branch: `offline-streaming-audit`.

| Commit | Section | Summary |
|---|---:|---|
| `9d6dd2f` | 0 | Audit report |
| `2644010` | 1 | Offline environment, local model paths, compose limits/capability |
| `8916cce` | 2 | Prompt diagnostics, history cap, cache-aware metrics |
| `297d1d5` | 3 | Optional GGUF page-cache prewarming |
| `328cf7b` | 4 | Dashboard TTFT display |
| `3358a15`, `4b033a5` | 5 | Streaming protocol, client playback, async stream queue |
| `4812d34` | 6 | Benchmark and diagnostic collection scripts |

Changed areas include `nova/config.py`, `nova/main.py`,
`nova/brain/{llm_engine,intent_classifier,stt_engine,sentence_splitter}.py`,
`nova/memory/vector_store.py`, `docker-compose.yml`, `.env.example`,
`README.md`, `web_ui/app.js`, `docs/streaming_protocol.md`, `scripts/`,
and `tests/`.

## C) New environment variables

Defaults: `NOVA_OFFLINE=1`, `NOVA_THREADS=4`,
`NOVA_HISTORY_TURNS=6`, `NOVA_PREWARM_MODELS=false`,
`NOVA_MINILM_MODEL_DIR=./models/minilm`,
`NOVA_WHISPER_MODEL_DIR=./models/whisper`.
Previously added performance variables remain documented in `.env.example`.
Compose also supports `NOVA_MEM_LIMIT=6g`.

## D) Prompt diagnostics

`python3 scripts/diag_prompt.py` reports estimated tokens because no GGUF
tokenizer/model is installed:

| Intent | System chars | Dynamic chars | Estimated tokens |
|---|---:|---:|---:|
| chat | 644 | 72 | 199 |
| action_music | 644 | 75 | 200 |
| action_calendar | 644 | 474 | 311 |
| memory_write | 644 | 92 | 204 |
| memory_recall | 644 | 94 | 205 |

Tool schema tokens and real tokenizer effects are not represented in the
no-model diagnostic.

## E) Test results

- Python compilation: passed.
- Sentence splitter checks: passed.
- STUB engine/config checks: passed.
- `docker compose config -q`: passed in this environment.
- `pytest`: not run; `pytest` is not installed.
- Real model TTFT, cache hits, streaming decode rate, Piper playback, and
  local embedding behavior: not verifiable without model assets/dependencies.

## F) Server deployment and verification

Copy or create these host directories and populate them with complete local
model trees:

```text
models/hf_cache/
models/minilm/
models/whisper/
```

Then rebuild the image rather than only restarting the old image:

```bash
docker compose up -d --build
docker compose logs -f nova_server
bash scripts/check_compose.sh
python3 scripts/diag_prompt.py
bash scripts/collect_diagnostics.sh
```

Verify the startup log contains `offline=true`, the MiniLM and Whisper paths,
and verify the WebSocket with and without `"stream": true`.

## G) Risks and assumptions

- `faster-whisper` and `SentenceTransformer` API support for
  `local_files_only` depends on installed package versions.
- Existing Chroma collections created with the default embedding function may
  need rebuilding with the local MiniLM embedding function.
- Streaming cancellation across an already-running WebSocket turn remains
  limited by the current sequential receive loop.
- No real GGUF or Docker image was available for inference/latency validation.

## H) Important configuration values

- `LLM_THREADS=4`
- `LLM_CONTEXT_SIZE=4096`
- `LLM_GPU_LAYERS=0`
- Models: `qwen3-1.7b`, `qwen3-0.6b`, `qwen2.5-3b`
- `INTENT_CONFIDENCE_THRESHOLD=0.5`
- Persona length: 644 characters, approximately 179 tokens by the diagnostic
  estimate.

## I) PASTE THIS TO CLAUDE

```text
NOVA follow-up branch offline-streaming-audit adds an audit report and local
offline model configuration. Compose now uses NOVA_MEM_LIMIT, fixed numeric
memlock, IPC_LOCK, HF offline/telemetry env, and NOVA_THREADS. MiniLM and
faster-whisper prefer models/minilm and models/whisper; Chroma can use a local
MiniLM embedding function. Prompt history is capped at 6 turns and
scripts/diag_prompt.py reports estimated prompt sizes: chat 199, music 200,
calendar 311, memory_write 204, memory_recall 205 tokens without a GGUF.
Streaming now emits text_delta/audio_chunk/audio_done and the web UI schedules
PCM chunks with WebAudio. Key config remains threads=4, ctx=4096, GPU layers=0,
threshold=0.5, persona=644 chars (~179 estimated tokens). Syntax, STUB, splitter,
and compose checks passed; pytest and real-model latency/cache/audio checks
were unavailable. Remaining risk: cancellation is not fully concurrent with
the sequential WebSocket receive loop, and old Chroma collections may need
reindexing under local embeddings.
```
