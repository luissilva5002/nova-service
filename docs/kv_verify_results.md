# KV lane verification

- server: http://127.0.0.1:8000
- session: kvverify-1795029979

## Restart check (waited 0.0s for the lanes)

| check | result |
|---|---|
| action lanes ready | PASS |
| all prefixes loaded from disk | PASS |
| none rebuilt (base_saved == 0) | PASS |
| no prefill tokens spent | PASS |
| no invalid base states | PASS |

kv_lanes counters: `{"swaps_in": 0, "swaps_out": 0, "cold_starts": 0, "restore_failures": 0, "evictions": 0, "warm_prefill_tokens": 0, "base_loaded": 3, "base_saved": 0, "base_invalid": 0, "base_save_failures": 0, "snapshot_bytes": 87482027, "snapshots": 3, "resident": "action:memory_write", "turns": 0, "compactions": 0, "overflows": 0, "external_turns": 0, "action_turns": 0, "strip_ok": 0, "strip_skipped": 0, "strip_failed": 0, "strip_tokens_removed": 0}`

### Boot log

| lane | source | tokens | prefilled | seconds |
|---|---|---:|---:|---:|

(no 'KV action lane ready' lines found in nova_run2.log; it must be the server's log of the boot you are checking)

## Turns

| # | kind | lane | prompt | reused | evaluated | warm prefill | strip | model ttft s | client first s | client reply s | overhead s | result |
|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| 1 | calendar | action:action_calendar | 1507 | 1237 | 270 | 0 | - | 16.71 | 24.81 | 24.81 | 7.95 | PASS |
| 2 | music | action:action_music | 539 | 473 | 66 | 0 | - | 5.86 | 8.75 | 8.75 | 2.81 | PASS |

## Lane counters (before -> after)

| counter | before | after |
|---|---:|---:|
| base_loaded | 3 | 3 |
| base_saved | 0 | 0 |
| base_invalid | 0 | 0 |
| cold_starts | 0 | 0 |
| restore_failures | 0 | 0 |
| evictions | 0 | 0 |
| swaps_in | 0 | 2 |
| swaps_out | 0 | 0 |
| warm_prefill_tokens | 0 | 0 |
| compactions | 0 | 0 |
| overflows | 0 | 0 |
| strip_ok | 0 | 0 |
| strip_skipped | 0 | 0 |
| strip_failed | 0 | 0 |
| strip_tokens_removed | 0 | 0 |

**Overall: PASS**
