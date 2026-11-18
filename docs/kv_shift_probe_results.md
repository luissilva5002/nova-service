# KV shift probe results

`scripts/kv_shift_probe.py` checks that old chat turns can be dropped from a live
llama.cpp KV cache (`kv_cache_seq_rm`, then `kv_cache_seq_shift` on the tail) and
that generation continues from the shifted cache. It records token counts and
generated-token equality only.

## Environment

| | Laptop (Windows 11, Python 3.13.7) | Server (Docker) |
|---|---|---|
| llama-cpp-python | 0.3.36 | 0.3.35 |
| model | qwen2.5-3b-instruct-q4_k_m | not run yet |
| flash attention / KV type | on / f16 | not run yet |
| n_ctx 4096 | PASS | **pending** |
| n_ctx 1024 | PASS | **pending** |

## Laptop results (identical at n_ctx 4096 and 1024)

All five acceptance criteria passed:

1. **API**: `kv_cache_seq_rm(seq_id, p0, p1) -> bool` and
   `kv_cache_seq_shift(seq_id, p0, p1, shift)` are present; `kv_cache_seq_add` is
   absent in 0.3.36. The shift wraps `llama_memory_seq_add` and `seq_id` must be 0
   (-1 hits a `GGML_ASSERT`).
2. **Cost**: the first generation after a shift evaluated 26 to 27 tokens, against
   369 to 663 for a cold run of the same prompt (dropping 2, 5 or 9 of 14 turns).
3. **Coherence**: every shifted reply was fluent and on topic. Only 2 of 9 single
   trials and 1 of 4 consecutive shifts matched the cold reply exactly; the others
   diverged between token 2 and token 22 into a paraphrase (for example "tends to
   rise" against "rises"). The cold run is therefore not a bit-exact oracle.
4. **Consecutive shifts**: 4 shifts in a row stayed stable (27, 25, 26, 20 evaluated tokens).
5. **Anchor**: the 140-token system prefix was untouched in every trial.

Quantized KV: Q8_0 and Q4_0 pass with `flash_attn=True` and fail at context
creation without it.

## Still to run

```text
python scripts/kv_shift_probe.py                     # server, inside the container (0.3.35)
python scripts/kv_shift_probe.py --n-ctx 1024        # server
```

## Experiment 6: removing a middle span (KV-C-lite)

The operation needed to strip a turn's `[Context]` block is a removal in the
middle of the history. `scripts/kv_strip_probe.py` runs it through the production
code (`LaneEngine`, `nova/brain/kv_edit.py`); the full table and replies are in
`docs/kv_strip_probe_results.md`.

Laptop, n_ctx 4096, 8-turn chat:

- 7 of 7 turns from the second were stripped (turn 1 has no earlier block), and the KV equalled the
  stored history token for token after every turn.
- Evaluated tokens per turn from turn 2: 82 with strip, 146 without (64 fewer).
- All 7 stripped replies were coherent and on topic. They diverged from a cold
  run before token 30 on 7 of 7 turns, against 4 of 7 for a control that
  evaluates the same history in one plain chunk with no shift. The extra
  divergence is expected: the cached states after the removed block were computed
  while the block was visible, so they are an approximation of a cold evaluation.

Verdict: structurally safe to implement and enable; replies are an approximation
of a cold run. Enable `NOVA_KV_STRIP_CONTEXT=1` if about 1.4 s less prefill per
chat turn is worth slightly different wording; leave it off for exact
reproducibility.
