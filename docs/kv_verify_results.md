# KV lane verification

- server: http://127.0.0.1:8000
- session: kvverify-1795033791

## Turns

| # | kind | lane | prompt | reused | evaluated | warm prefill | strip | model ttft s | client first s | client reply s | overhead s | result |
|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| 1 | chat | chat:4:chat:19:kvverify-1795033791 | 207 | 0 | 207 | 0 | stripped | 3.96 | 4.56 | 8.07 | 0.64 | PASS |
| 2 | calendar | action:action_calendar | 1554 | 1237 | 317 | 0 | - | 7.13 | 8.53 | 8.53 | 1.37 | PASS |
| 3 | chat | chat:4:chat:19:kvverify-1795033791 | 288 | 188 | 100 | 0 | stripped | 1.45 | 1.50 | 4.60 | 0.05 | PASS |
| 4 | calendar | action:action_calendar | 1570 | 1237 | 333 | 0 | - | 7.38 | 7.68 | 7.68 | 0.27 | PASS |
| 5 | music | action:action_music | 540 | 473 | 67 | 0 | - | 2.44 | 4.49 | 4.49 | 2.03 | PASS |
| 6 | chat | chat:4:chat:19:kvverify-1795033791 | 436 | 277 | 159 | 0 | stripped | 2.36 | 2.40 | 7.01 | 0.14 | PASS |
| 7 | calendar | action:action_calendar | 1567 | 1237 | 330 | 0 | - | 7.51 | 7.83 | 7.83 | 0.29 | PASS |

## Lane counters (before -> after)

| counter | before | after |
|---|---:|---:|
| base_loaded | 3 | 3 |
| base_saved | 0 | 0 |
| base_invalid | 0 | 0 |
| cold_starts | 0 | 1 |
| restore_failures | 0 | 0 |
| evictions | 0 | 0 |
| swaps_in | 0 | 6 |
| swaps_out | 0 | 3 |
| warm_prefill_tokens | 0 | 0 |
| compactions | 0 | 0 |
| overflows | 0 | 0 |
| strip_ok | 0 | 3 |
| strip_skipped | 0 | 0 |
| strip_failed | 0 | 0 |
| strip_tokens_removed | 0 | 194 |

**Overall: PASS**
