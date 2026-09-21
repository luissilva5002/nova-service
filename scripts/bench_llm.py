"""Measure cold/warm prompt latency and decode throughput for an installed GGUF."""
import argparse
import time

from llama_cpp import Llama


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to a GGUF model file")
    args = parser.parse_args()
    llm = Llama(model_path=args.model, n_ctx=4096, n_threads=4, n_batch=512, verbose=False)
    prompt = " ".join(["Explain this local assistant design clearly."] * 140)
    messages = [{"role": "user", "content": prompt}]
    rows = []
    for label in ("cold", "warm"):
        started = time.perf_counter()
        result = llm.create_chat_completion(messages=messages, max_tokens=64, temperature=0.4)
        elapsed = time.perf_counter() - started
        usage = result.get("usage", {})
        tokens = usage.get("completion_tokens", 0)
        rows.append((label, usage.get("prompt_tokens", 0), elapsed, tokens / elapsed if elapsed else 0))
    stream_start = time.perf_counter()
    stream = llm.create_chat_completion(messages=messages, max_tokens=64, temperature=0.4, stream=True)
    next(stream)
    ttft = time.perf_counter() - stream_start
    print("run   prompt_tokens  total_seconds  decode_tok_s  first_delta_s")
    for label, prompt_tokens, elapsed, rate in rows:
        print(f"{label:<5} {prompt_tokens:>13}  {elapsed:>13.3f}  {rate:>12.2f}  {'-' if label == 'cold' else f'{ttft:.3f}':>13}")


if __name__ == "__main__":
    main()
