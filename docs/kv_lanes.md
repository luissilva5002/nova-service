# KV lanes

KV lanes keep token history and prompt prefixes in separate "lanes" while one
llama.cpp context is shared. Enable the feature with `NOVA_KV_LANES=1`. It is
disabled by default. Use `NOVA_LLM_MODEL=qwen2.5-3b`; Qwen3 models disable lanes
because the lane path does not apply the Qwen3 non-thinking chat handler.

There are two kinds of lane:

- **Chat lanes** (one per session): the session's token history, reused turn
  after turn so only the new message is evaluated.
- **Action lanes** (one per tool intent): the persona plus the tool schemas of
  `action_calendar`, `action_music` and `memory_write`. This prefix is large
  (the calendar schema alone is about 1,200 tokens) and identical on every
  turn, so it is prefilled once and restored afterwards.

## Action lanes and persisted base states

A tool turn in a multi-tool intent runs `LaneEngine.action_turn()`:

1. The lane's prefix is made resident: from memory, from the persisted base
   state on disk, or prefilled once (and then persisted).
2. The turn (the `[Context]` block plus the user text) is evaluated and the
   model generates the tool call.
3. The turn is discarded. Action lanes keep only their prefix; the exchange is
   added to the session's chat lane through `record_external_turn()`.

Single-tool intents (`memory_recall`) stay on the legacy path, because
llama.cpp forces that one tool call with a grammar. Any action-lane failure
falls back to the legacy path for that turn.

Base states are written to `data/kv_base/` (inside the `./data` volume in
Docker). A state is reused only if all of these match: model fingerprint and
size, llama-cpp-python version, `n_ctx`, flash-attention and KV types, RoPE
settings, the lane name, and a hash of the prompt tokens. Changing the persona,
a tool schema or the model therefore rebuilds the state automatically and
removes the stale file. A corrupt or unreadable file is ignored and rebuilt.
States are tied to one machine's llama-cpp build: do not copy them between
machines.

After startup NOVA builds or restores the action-lane prefixes in the
background (`NOVA_KV_PREWARM_ACTIONS`). With states on disk this takes
milliseconds per lane; without them it is one prefill per lane (measured on the
Ryzen 7 laptop at about 45 tokens/s: calendar 1237 tokens in 27 s, music 473
tokens in 11 s, memory_write 662 tokens in 13 s).

A build only starts after user-facing work has been quiet for 4 seconds, and a
message that arrives while a lane is being built waits for that one build. On the
very first boot, wait for the three `KV action lane ready` lines before chatting.

What is still evaluated on every action turn: the date block (a 14-day lookup
table for the calendar) and the user text. When the previous turn used the
same action lane and the date block is unchanged, this tail is reused too.

Action turns see the last `NOVA_KV_ACTION_HISTORY_TURNS` user/assistant
exchanges (default 1) as plain text in the `[Context]` block, instead of the
six turns of chat messages the legacy path sends. Set it to 0 to send none.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `NOVA_KV_LANES` | off | Master switch for lanes. |
| `NOVA_LLM_CTX` | 4096 | llama context size. |
| `NOVA_KV_LANES_MAX_SNAPSHOT_MB` | 256 | Total in-memory snapshot budget (chat and action). |
| `NOVA_KV_ACTION_LANES` | on | Use action lanes for tool turns (needs `NOVA_KV_LANES`). |
| `NOVA_KV_BASE_STATES` | on | Persist and restore action-lane prefixes. |
| `NOVA_KV_BASE_DIR` | `data/kv_base` | Where base states are stored. |
| `NOVA_KV_PREWARM_ACTIONS` | on | Build or restore action-lane prefixes after startup. |
| `NOVA_KV_ACTION_HISTORY_TURNS` | 1 | Exchanges shown to action turns as text. |

## Observability

A lane-backed turn logs a line starting with `KV-LANE`:
`lane`, `prompt_tokens`, `reused`, `evaluated`, `ttft_s`, `completion_tokens`
and `warm_prefill`. `warm_prefill` is the number of prefix tokens built during
this turn (0 when the prefix was already resident or restored). Action turns
report `lane=action:<intent>`. At startup the log shows one line per action lane:
`KV action lane ready: lane=... source=disk|built|memory ...`.

`/api/status` reports lane counters under `kv_lanes`, including
`base_loaded`, `base_saved`, `base_invalid`, `base_save_failures`,
`action_turns`, `swaps_in`, `swaps_out` and `cold_starts`.

## Rolling back

Set `NOVA_KV_ACTION_LANES=0` to keep chat lanes but send tool turns through the
legacy path. Unset `NOVA_KV_LANES` (or set it to `0`) to disable all lanes.
Deleting `data/kv_base/` is always safe; the states are rebuilt.

## Known limits

- Compaction rebuilds a chat lane from persisted text and starts cold. In-place
  cache shifting is being evaluated (see `scripts/kv_shift_probe.py`) and is not
  used in production.
- The calendar date block is re-evaluated when the previous turn was not in the
  same action lane. A per-day prefix snapshot would remove that cost.
- Qwen3 models are not supported by the lane path.
