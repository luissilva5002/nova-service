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
