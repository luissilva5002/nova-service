# KV-B2-0: ContextManager discovery

## Environment

- OS: Windows 11 Home (local development laptop; Docker Linux engine was unavailable).
- CPU: AMD Ryzen 7 5825U with Radeon Graphics.
- RAM: 15.4 GiB reported physical memory.
- Python: 3.13.7.
- llama-cpp-python: 0.3.36, installed into the local Python 3.13 environment from the project's CPU wheel index.
- `n_threads`: 4 from the local configured default (`nova/config.py:73`).
- Required model: `models/brain/qwen2.5-3b-instruct-q4_k_m.gguf` is present locally; `nova/config.py:40-43` maps `qwen2.5-3b` to this path.
- Runtime status: Docker CLI is installed, but the Docker Desktop Linux engine pipe was unavailable when checked. Both scripts therefore ran in the Windows host environment, not inside the container. Paths resolve from the repository/config; neither script uses a server absolute path.
- Part 2 and Part 3 ran locally using the exact configured GGUF. No timings were collected or interpreted.

## Part 1 — source survey

The references below are repository-relative `file:line` citations.

### a. Model prompt/generation call sites

| Path | Parameters and prompt | End/stop behavior | Result delivery |
|---|---|---|---|
| `LLMEngine.generate_with_tools` (`nova/brain/llm_engine.py:453-514`) | Calls `create_chat_completion` at line 485 with `messages`, configured `max_tokens` (tool default `LLM_MAX_TOKENS_TOOL=96`, chat default `LLM_MAX_TOKENS_CHAT=128`), and `temperature=LLM_TEMP_TOOLS` (default `0.1`) when schemas exist, otherwise `0.4` (`config.py:80-83`). If tools exist it passes `tools` and `tool_choice`; exactly one tool is forced, more than one uses `"auto"` (`llm_engine.py:476-484`). No explicit `stop` parameter. | No engine-level EOS or `<|im_end|>` stopping logic is passed. The GGUF chat template/llama.cpp completion API controls termination. A returned text result passes through `_without_thinking`, which removes `<think>...</think>` blocks, not end markers (`llm_engine.py:117-118, 508-514`). | Returns one complete tool-call or text decision; this path itself does not stream text (`llm_engine.py:486-514`). |
| `LLMEngine.generate_raw_with_metrics` (`nova/brain/llm_engine.py:521-535`) | Calls `create_chat_completion` at lines 529-533 using system+user messages, caller `max_tokens`, and `temperature=0.2`. Callers include optional intent fallback (`intent_router.py:135-149`), memory extraction (`memory_agent.py:36-46`), ingestion classification (`ingestion/llm_filter.py:47-52`), and optional tool confirmation (`main.py:520-527`). There is no explicit `stop`. `generate_raw` delegates to this method (`llm_engine.py:517-518`). | Same library/template-controlled completion ending; text is passed through `_without_thinking` (`llm_engine.py:534`). | Non-streaming helper. |
| `LLMEngine.stream_chat` (`nova/brain/llm_engine.py:537-594`) | Calls `create_chat_completion` at lines 554-555 with the constructed messages, `max_tokens=LLM_MAX_TOKENS_CHAT` (default 128), `temperature=0.4`, and `stream=True`; no explicit `stop`. It runs production in a worker thread and forwards returned deltas through an async queue. | No explicit stop strings or EOS check in NOVA; stream ends when llama.cpp completes/raises and the producer puts its queue sentinel (`llm_engine.py:550-577`). | Each non-empty `delta.content` is added to output and yielded as `{"type":"text_delta","text":...}` (`llm_engine.py:578-583`). `/ws/chat` sends each delta immediately as `text_delta`; it also sentence-splits it to synthesize `audio_chunk`s, then persists and sends the final text/metrics/`audio_done` (`main.py:616-671`). |
| `Llama.generate` | No production call site exists in current application code. The added `scripts/kv_generate_probe.py` calls `llama.generate(prompt_ids, temp=0.0, reset=False)` with a 40-token manual cap and stops when the yielded token id equals the GGUF `<|im_end|>` id. | Probe-specific token-id EOS handling; no new production behavior. | Probe collects generated ids and decodes them locally; no websocket path. |
| `create_completion` | No application-level direct call site. `llm_engine.py:244-245, 320-435` wraps llama.cpp's internal `_create_completion`/chat completion for KV instrumentation; it is not another prompt-producing pipeline call. | Delegates to llama.cpp. | Instrumentation records token/prompt metadata, not websocket text. |

The WebSocket also has a non-streaming path: it awaits `run_pipeline`, which awaits a complete response, then sends one JSON `"text"` message, metrics, optional URL action, and base64 audio (`main.py:674-700`). Streaming must be explicitly requested with `"stream": true`, and only the `"chat"` intent enters `stream_chat` (`main.py:616-629`).

### b. Per-turn prompt contents and placement

| Prompt component | Construction and position | Turn variability |
|---|---|---|
| Persona/system prompt | `LLMEngine._build_messages` starts with exactly one system message containing `NOVA_PERSONA_PROMPT` (`llm_engine.py:618-628`; prompt text/configured override in `config.py:87-95`). The chat-completion tool schemas are passed separately as the `tools` argument, not placed in a history message; the GGUF chat formatter handles that schema rendering (`llm_engine.py:476-484`). | Persona is fixed unless the environment override changes. Schema is fixed for a request/session's selected intent, but can vary between turns because the intent filters it. |
| History | After system message, `_build_messages` appends the most recent configured number of user/assistant messages, discarding other roles (`llm_engine.py:625-629`). `run_pipeline` obtains at most 12 persisted messages (= six user/assistant turns) using `get_context(..., limit=12)` (`main.py:380`). | Changes each turn and is session-specific. |
| Date/time | `_current_date_context` is placed inside the *current user message* as `[Context] ... [/Context]`, immediately before user text (`llm_engine.py:597-615, 630-633`). All requests include the current local date and timezone abbreviation (not clock time). Calendar intent also includes a 14-day date/weekday lookup table. | Recomputed each turn; longer for `action_calendar`. |
| Session summary and facts/vault context | `main.run_pipeline` builds JSON `memory_context`: targeted vault preferences/knowledge/profile fields for `memory_recall`; summary, selected profile fields, and anti-fabrication guidance for `chat`; empty for action and memory-write intents (`main.py:407-439`). `_build_messages` appends this as `Known user context:` in the current user `[Context]` block, following date context and before the current user text (`llm_engine.py:630-633`). Memory recall can answer directly before the LLM (`main.py:386-401`). | Recomputed per turn; depends on intent and vault/session data. Streaming chat instead includes the session summary and an empty relevant-profile object (`main.py:619-629`). |
| Tool schemas | Supplied outside the `messages` list as `tools=tool_schemas`; `tool_choice` is forced for one schema and auto for multiple (`llm_engine.py:480-484`). Router discovers schemas from skills (`tool_router.py:34-68`). | Intent-specific and can change from turn to turn; chat-streaming passes no tools. |
| Explicit per-turn user input | Final user message is `[Context]\n{date and optional memory context}\n[/Context]\n\n{user_text}` (`llm_engine.py:630-633`). | Always changes. |

There is no separate date/time message and no prior generated assistant completion is added to the raw system prompt. The `_build_system_prompt` helper only joins the persona with optional memory context; current `generate_with_tools` uses `_build_messages` (`llm_engine.py:618-633`).

### c. Intent-to-tool mapping

`main.run_pipeline` gets the discovered schemas then filters them using `filter_tools_for_intent` (`main.py:448-454`; filter definitions `intent_router.py:89-95, 247-249`). The mapping actually passed to the LLM is:

| Intent | Tool names exposed |
|---|---|
| `action_music` | `ytm_play`, `ytm_skip`, `ytm_pause`, `ytm_set_volume` |
| `action_calendar` | `gc_list_today`, `gc_find`, `gc_create_event`, `gc_update_event`, `gc_delete_event` |
| `memory_write` | `update_user_preference`, `update_user_knowledge` |
| `memory_recall` | `recall_note` (unless `_answer_from_memory` directly answers before the LLM, `main.py:386-401`) |
| `chat` | None |

`stream_chat` always passes an empty tools list, and is selected only when requested streaming intent is chat (`main.py:616-629`; `llm_engine.py:547-555`). Skill schemas are defined in `nova/skills/{youtube_music,google_calendar,remember,flutter_workspace}/tools.py`; notably, Flutter workspace tools are discovered but are not selected by any current intent filter.

### d. History storage, loading, and trimming

- `session_memory` exports `load_session`, `save_session`, `append_turn`, `get_context`, `get_session_summary`, `get_session_snapshot`, and clear helpers. It stores one JSON file per sanitized session id in `DATA_DIR/session_memory` (`session_memory.py:9-10, 13-18`).
- The JSON object contains `session_id`, `history`, `summary`, `created_at`, and `updated_at`. Each history item has `role`, `content`, and an ISO UTC `timestamp`; `append_turn` updates the summary and persists after each message (`session_memory.py:20-60`).
- Normal and streamed chat append the user and assistant separately (`main.py:545-546, 657-658`). Tool-call internals are not stored as separate roles; only user text and final assistant reply are persisted. Session-command/direct-memory responses are similarly appended in their early-return paths (`main.py:362-363, 393-394`).
- Prompt trimming happens when `main.py` requests `get_context(..., limit=12)` (six user/assistant turns) and again in `_build_messages`, which slices to `LLM_HISTORY_TURNS * 2` (default six turns) and keeps only user/assistant roles (`main.py:380`; `llm_engine.py:627-629`; `config.py:84`). The JSON `history` itself is not trimmed by `append_turn`.
- On restart, `load_session(session_id)` reads the same JSON path and defaults missing keys; malformed/unreadable JSON returns a new empty session (`session_memory.py:29-43`). The active socket selects its id from query parameter or creates a UUID, and accepts a later `"session"` message or per-message `session_id` update (`main.py:579-605`). The web client keeps a session id in local storage and sends it in its WebSocket URL/handshake (`web_ui/app.js:24-48`).
- `CoreStore.conversation_log` separately persists raw user/NOVA role+content+timestamp records in SQLite; `recent_messages` reads it, but LLM prompt history is loaded from `session_memory`, not that log (`core_store.py:40-45, 105-121`).

### e. Tool parsing and single-shot flow

`generate_with_tools` reads the first structured `tool_calls` function response and JSON-decodes its `arguments`; malformed argument JSON falls back to `{}` (`llm_engine.py:490-506`). If a text response contains `<tool_call`, `main._extract_tool_call_from_text` extracts the inner block (or whole string), tries normal JSON, doubled outer braces, then a substring between first/last braces, and accepts a decoded dictionary containing `name` (`main.py:143-177`). Failure is logged and the text reply continues (`main.py:466-484`).

For a recovered or structured call, `main.run_pipeline` alone calls `tool_router.dispatch(name, arguments)` (`main.py:486-489`). The router resolves the owning skill and invokes `skill.execute`; it returns a string or a structured result (`tool_router.py:80-91`). The pipeline does **not** feed the tool result back to the model by default: plain string/generic results are spoken directly; audio/open-url results are described directly unless `LLM_CONFIRMATIONS` is enabled, in which case `generate_raw_with_metrics` receives a confirmation system prompt and the tool result/message as user prompt, with `max_tokens=64` (`main.py:490-538`, `config.py:83`). There is no automatic second tool-call round trip in this flow.

### f. Chat/session concurrency

Multiple `/ws/chat` connections can each hold a different session id and each session's JSON file/history. The shared `llm_engine` is a singleton (`llm_engine.py:636`); all model calls use its `asyncio.Lock` around inference (`llm_engine.py:485-486, 529-533, 549-577`). Thus separate chats can be connected/active concurrently, but their model generation is serialized, not parallel. Reusing the same `session_id` intentionally accesses the same persisted history.

## Part 2 — tokenizer-only tool cost

`scripts/kv_toolcost.py` uses only the `qwen2.5-3b` vocabulary/template (`Llama(..., vocab_only=True)`), `LlamaBackend`, the real persona system prompt, each skill's declarative schemas, and current intent filtering. It prints measured system-block counts; the table was produced on the local Windows host with llama-cpp-python 0.3.36.

| scope | tool names | system-block tokens | delta vs no tools |
|---|---|---:|---:|
| no tools | (none) | 140 | +0 |
| skill: flutter_workspace | `search_codebase`, `read_file` | 415 | +275 |
| skill: google_calendar | `gc_list_today`, `gc_find`, `gc_create_event`, `gc_update_event`, `gc_delete_event` | 1237 | +1097 |
| skill: remember | `update_user_preference`, `update_user_knowledge`, `recall_note` | 760 | +620 |
| skill: youtube_music | `ytm_play`, `ytm_skip`, `ytm_pause`, `ytm_set_volume` | 473 | +333 |
| intent: action_music | `ytm_play`, `ytm_skip`, `ytm_pause`, `ytm_set_volume` | 473 | +333 |
| intent: action_calendar | `gc_list_today`, `gc_find`, `gc_create_event`, `gc_update_event`, `gc_delete_event` | 1237 | +1097 |
| intent: memory_write | `update_user_preference`, `update_user_knowledge` | 662 | +522 |
| intent: memory_recall | `recall_note` | 317 | +177 |
| intent: chat | (none) | 140 | +0 |
| ALL tools | `search_codebase`, `read_file`, `gc_list_today`, `gc_find`, `gc_create_event`, `gc_update_event`, `gc_delete_event`, `update_user_preference`, `update_user_knowledge`, `recall_note`, `ytm_play`, `ytm_skip`, `ytm_pause`, `ytm_set_volume` | 2228 | +2088 |
| memory_search | `memory_search` (fake one-string `query` schema) | 275 | +135 |

The script reads skill schema declarations directly rather than importing each handler, because the local run's `ToolRouter` logged that it could not import the YouTube Music handler. The measured schema is still the declared `ytm_*` schema; actual skill registration requires its handler dependencies.

Run from the repository root inside the model-equipped runtime: `python scripts/kv_toolcost.py`.

## Part 3 — token-id generation probe

`scripts/kv_generate_probe.py` loaded `qwen2.5-3b` via `LLMEngine.load()` with configured production settings and ran four fixed user turns twice (engine-default `LlamaRAMCache` and `llm.set_cache(None)`). It wraps `llm.eval` to count tokens evaluated before the first yielded token, records `llm.n_tokens` before each generation, uses `ContextManager`, `reset=False`, temperature zero, a 40-token cap, and stops at the `<|im_end|>` id. The final variation changes the first 100 token ids of the turn-four prompt before another `generate`. Reuse estimate is prompt-token count minus tokens counted in `eval` before the first generated token.

### Run A — current LlamaRAMCache attached

| Turn | Prompt tokens | Evaluated before first generated token | Reuse estimate | `llm.n_tokens` before generate | Decoded generated text |
|---|---:|---:|---:|---:|---|
| 1 | 163 | 163 | 0 | 0 | Great plan! For lunch, you might want to check out some local eateries or even a quick home-cooked meal. Have you got any specific type of cuisine in mind? |
| 2 | 222 | 222 | 0 | 199 | For an easy walk with a nice view of trees, you might want to head to a local park or a forested area. If you're near a city, a park like Central Park in New |
| 3 | 281 | 281 | 0 | 460 | Perfect! For lunch, you can try a simple vegetarian restaurant or a caf� that serves healthy, plant-based meals. Some local options might be Green Zebra, The Herbivore, or even a |
| 4 | 340 | 340 | 0 | 780 | Sure! For your quiet weekend walk, we decided to head to a local park or forested area with trees for an easy walk. For lunch, you prefer simple vegetarian food. |
| turn 4 prefix changed | 340 | 340 | 0 | 1156 | Sure! For your quiet weekend walk, you want an easy walk with a nice view of trees. You prefer a location like a local park or a forested area. For lunch, you like simple |

Conclusion (i): No prefix reuse was measured for token-id `generate()`; turns 2-4 each evaluated the full prompt (0 reused).

Conclusion (ii): The same per-turn counts without the RAM cache were 0, so `LlamaRAMCache` did not change measured reuse. The changed-prefix variation also completed normally with 0 reused tokens.

### Run B — `llm.set_cache(None)`

| Turn | Prompt tokens | Evaluated before first generated token | Reuse estimate | `llm.n_tokens` before generate | Decoded generated text |
|---|---:|---:|---:|---:|---|
| 1 | 163 | 163 | 0 | 0 | Great plan! For lunch, you might want to check out some local eateries or even a quick home-cooked meal. Have you got any specific type of cuisine in mind? |
| 2 | 222 | 222 | 0 | 199 | For an easy walk with a nice view of trees, you might want to head to a local park or a forested area. If you're near a city, a park like Central Park in New |
| 3 | 281 | 281 | 0 | 460 | Perfect! For lunch, you can try a simple vegetarian restaurant or a caf� that serves healthy, plant-based meals. Some local options might be Green Zebra, The Herbivore, or even a |
| 4 | 340 | 340 | 0 | 780 | Sure! For your quiet weekend walk, we decided to head to a local park or forested area with trees for an easy walk. For lunch, you prefer simple vegetarian food. |
| turn 4 prefix changed | 340 | 340 | 0 | 1156 | Sure! For your quiet weekend walk, you want an easy walk with a nice view of trees. You prefer a location like a local park or a forested area. For lunch, you like simple |

Conclusion (i): No prefix reuse was measured for token-id `generate()`; turns 2-4 each evaluated the full prompt (0 reused).

Conclusion (ii): Counts matched Run A exactly; disabling the `LlamaRAMCache` did not change measured reuse (all 0). The changed-prefix variation completed normally with 0 reused tokens.

Run again from the repository root with `python scripts/kv_generate_probe.py`. It exits before loading any alternate model if the configured Qwen2.5-3B GGUF is missing.
