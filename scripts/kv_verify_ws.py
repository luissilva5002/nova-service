"""End-to-end KV lane check through the real /ws/chat endpoint of a running NOVA server.

Start the server first (native uvicorn or Docker), with for example:
    NOVA_KV_LANES=1  NOVA_LLM_MODEL=qwen2.5-3b  (and optionally NOVA_KV_STRIP_CONTEXT=1)

Then, from the repository root (needs `pip install websockets`):
    python scripts/kv_verify_ws.py                       # 7-turn sequence
    python scripts/kv_verify_ws.py --expect-strip        # also require KV-C-lite to strip every chat turn
    python scripts/kv_verify_ws.py --restart-check       # after restarting the server: base states from disk?
    python scripts/kv_verify_ws.py --restart-check --log-file server.log

Everything is read from the `metrics` frame of each reply and from /api/status,
so no log scraping is needed (--log-file only adds the boot-time lane lines).
Calendar tool errors (no Google token) are expected and do not matter here.

A turn that the intent classifier sends to another lane is reported INCONCLUSIVE and does not fail
the run (use --strict-routing to make it fail).

Checks per turn:
  chat      lane is a chat lane; from the 2nd chat turn: reused grows, evaluated <= --chat-eval-max,
            model ttft <= --chat-ttft-max; with --expect-strip the KV strip status is "stripped".
  calendar  lane is action:action_calendar, reused >= --base-calendar, warm_prefill == 0.
  music     lane is action:action_music,    reused >= --base-music,    warm_prefill == 0.
"reused >= base" because an action turn right after the same lane also reuses the date block.

The client cannot see model-lock gaps directly; the table reports `overhead` = time to the final
text frame minus the model's own elapsed time (classifier, TTS and queueing included).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None

SEQUENCE = [
    ("chat", "hi, what can you do?"),
    ("calendar", "what's on my calendar tomorrow?"),
    ("chat", "tell me something interesting about octopuses"),
    ("calendar", "what's on my calendar on Friday?"),
    ("music", "play some jazz on youtube music"),
    ("chat", "thanks"),
    ("calendar", "am I free Saturday afternoon?"),
]
ACTION_LANE = {"calendar": "action:action_calendar", "music": "action:action_music"}
# \s+ between the fields: PowerShell hard-wraps the lines it writes to a Tee-Object log at the console width.
LANE_READY = re.compile(
    r"KV action lane ready:\s+lane=(\S+)\s+source=(\w+)\s+tokens=(\d+)\s+prefilled=(\d+)\s+seconds=([\d.]+)"
)


@dataclass
class Turn:
    index: int
    kind: str
    text: str
    lane: str = ""
    prompt: int = 0
    reused: int = 0
    evaluated: int = 0
    warm_prefill: int = 0
    strip: str = "-"
    model_ttft: float = 0.0
    model_total: float = 0.0
    client_first: float = 0.0
    client_reply: float = 0.0
    reply: str = ""
    error: str = ""
    verdict: str = "?"
    note: str = ""
    raw: list = field(default_factory=list)

    @property
    def overhead(self) -> float:
        return max(0.0, self.client_reply - self.model_total)


def http_json(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def lane_counters(http: str) -> dict:
    return http_json(f"{http}/api/status").get("kv_lanes", {})


def wait_ready(http: str, expect_lanes: int, timeout: float) -> tuple[bool, float, dict]:
    """Poll /api/status until the brain is loaded and the action-lane prefixes exist."""
    started = time.monotonic()
    last: dict = {}
    while time.monotonic() - started < timeout:
        try:
            status = http_json(f"{http}/api/status")
            last = status.get("kv_lanes", {})
            ready = status.get("brain_loaded") and last.get("enabled", True) is not False
            lanes_built = int(last.get("base_loaded", 0)) + int(last.get("base_saved", 0))
            if ready and lanes_built >= expect_lanes:
                return True, time.monotonic() - started, last
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(1.0)
    return False, time.monotonic() - started, last


async def run_turn(ws, session: str, turn: Turn, args) -> None:
    started = time.perf_counter()
    await ws.send(
        json.dumps({"type": "text", "text": turn.text, "session_id": session, "stream": not args.no_stream})
    )
    stream_path = False
    metrics: dict | None = None
    first_delta = text_at = last_frame = None
    while True:
        wait = args.turn_timeout if (metrics is None or stream_path) else 2.0
        try:
            raw = await asyncio.wait_for(ws.recv(), wait)
        except asyncio.TimeoutError:
            if metrics is None:
                turn.error = f"no metrics frame within {args.turn_timeout:.0f}s"
            break
        now = time.perf_counter() - started
        last_frame = now
        frame = json.loads(raw)
        kind = frame.get("type")
        if args.show_raw:
            turn.raw.append({k: (v if k != "audio_b64" else "...") for k, v in frame.items()})
        if kind == "text_delta":
            stream_path = True
            first_delta = first_delta if first_delta is not None else now
        elif kind == "audio_chunk":
            stream_path = True
        elif kind == "text":
            turn.reply, text_at = frame.get("text", ""), now
        elif kind == "metrics":
            metrics = frame.get("llm") or {}
        elif kind == "error":
            turn.error = frame.get("text", "error frame")
            break
        elif kind == "audio_done":
            break
    metrics = metrics or {}
    turn.lane = metrics.get("lane") or metrics.get("action_lane") or ""
    turn.prompt = int(metrics.get("prompt_tokens", 0))
    turn.reused = int(metrics.get("cache_hit_tokens", 0))
    turn.evaluated = int(metrics.get("evaluated_tokens", 0))
    turn.warm_prefill = int(metrics.get("warm_prefill_tokens", 0))
    turn.strip = str(metrics.get("kv_strip", "-"))
    turn.model_ttft = float(metrics.get("ttft_seconds", 0.0))
    turn.model_total = float(metrics.get("total_elapsed_seconds", 0.0))
    turn.client_first = first_delta if first_delta is not None else (text_at or 0.0)
    turn.client_reply = text_at if text_at is not None else (last_frame or 0.0)


def judge(turn: Turn, previous_chat: Turn | None, args) -> None:
    if turn.error:
        turn.verdict, turn.note = "FAIL", turn.error
        return
    if not turn.lane:
        turn.verdict, turn.note = "FAIL", "no lane in metrics: the turn ran on the legacy path"
        return
    if turn.kind == "chat":
        if not turn.lane.startswith("chat:"):
            turn.verdict, turn.note = "INCONCLUSIVE", f"routed to {turn.lane}, expected a chat lane"
            return
        problems = []
        if previous_chat is not None:
            if turn.reused <= previous_chat.reused:
                problems.append(f"reused did not grow ({previous_chat.reused} -> {turn.reused})")
            if turn.evaluated > args.chat_eval_max:
                problems.append(f"evaluated {turn.evaluated} > {args.chat_eval_max}")
            if turn.model_ttft > args.chat_ttft_max:
                problems.append(f"ttft {turn.model_ttft:.1f}s > {args.chat_ttft_max}s")
        if args.expect_strip and not turn.strip.startswith("stripped"):
            problems.append(f"strip status {turn.strip!r}, expected stripped")
        turn.verdict, turn.note = ("FAIL", "; ".join(problems)) if problems else ("PASS", "")
        return
    expected_lane, base = ACTION_LANE[turn.kind], (args.base_calendar if turn.kind == "calendar" else args.base_music)
    if turn.lane != expected_lane:
        turn.verdict, turn.note = "INCONCLUSIVE", f"routed to {turn.lane}, expected {expected_lane}"
        return
    problems = []
    if turn.reused < base:
        problems.append(f"reused {turn.reused} < base {base}")
    if turn.warm_prefill:
        problems.append(f"warm_prefill {turn.warm_prefill} (prefix was rebuilt)")
    turn.verdict, turn.note = ("FAIL", "; ".join(problems)) if problems else ("PASS", "")


def read_log_text(path: Path) -> str:
    """Decode a server log. Windows PowerShell 5.1 `Tee-Object` writes UTF-16 LE with a BOM."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    if b"\x00" in raw[:4096]:
        return raw.decode("utf-16-le", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


def parse_boot_log(path: Path) -> list[dict]:
    rows = []
    for match in LANE_READY.finditer(read_log_text(path)):
        lane, source, tokens, prefilled, seconds = match.groups()
        rows.append(
            {"lane": lane, "source": source, "tokens": int(tokens), "prefilled": int(prefilled), "seconds": float(seconds)}
        )
    return rows


def turns_table(turns: list[Turn]) -> list[str]:
    lines = [
        "| # | kind | lane | prompt | reused | evaluated | warm prefill | strip | model ttft s | client first s | client reply s | overhead s | result |",
        "|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for t in turns:
        result = t.verdict + (f": {t.note}" if t.note else "")
        lines.append(
            f"| {t.index} | {t.kind} | {t.lane or '?'} | {t.prompt} | {t.reused} | {t.evaluated} | {t.warm_prefill} | "
            f"{t.strip} | {t.model_ttft:.2f} | {t.client_first:.2f} | {t.client_reply:.2f} | {t.overhead:.2f} | {result} |"
        )
    return lines


async def run_sequence(args, sequence) -> list[Turn]:
    if websockets is None:
        raise SystemExit("The 'websockets' package is required: pip install websockets")
    session = args.session
    turns: list[Turn] = []
    previous_chat: Turn | None = None
    url = f"{args.url}?session_id={session}"
    async with websockets.connect(url, max_size=None, ping_interval=None) as ws:
        for index, (kind, text) in enumerate(sequence, start=1):
            if index > 1:
                await asyncio.sleep(args.idle)
            turn = Turn(index=index, kind=kind, text=text)
            print(f"[{index}/{len(sequence)}] {kind}: {text}", flush=True)
            await run_turn(ws, session, turn, args)
            judge(turn, previous_chat if kind == "chat" else None, args)
            if kind == "chat" and turn.lane.startswith("chat:"):
                previous_chat = turn
            print(
                f"      lane={turn.lane or '?'} prompt={turn.prompt} reused={turn.reused} evaluated={turn.evaluated} "
                f"warm_prefill={turn.warm_prefill} strip={turn.strip} -> {turn.verdict} {turn.note}",
                flush=True,
            )
            turns.append(turn)
    return turns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/chat")
    parser.add_argument("--http", default=None, help="HTTP base URL (default: derived from --url)")
    parser.add_argument("--session", default=f"kvverify-{int(time.time())}")
    parser.add_argument("--idle", type=float, default=5.0, help="seconds of silence between turns (prewarm waits 4 s)")
    parser.add_argument("--turn-timeout", type=float, default=240.0)
    parser.add_argument("--no-stream", action="store_true", help="never send stream=true")
    parser.add_argument("--no-wait", action="store_true", help="do not wait for the action-lane prefixes first")
    parser.add_argument("--ready-timeout", type=float, default=900.0)
    parser.add_argument("--expect-lanes", type=int, default=3)
    parser.add_argument("--base-calendar", type=int, default=1237)
    parser.add_argument("--base-music", type=int, default=473)
    parser.add_argument("--chat-eval-max", type=int, default=300)
    parser.add_argument("--chat-ttft-max", type=float, default=6.0)
    parser.add_argument("--expect-strip", action="store_true")
    parser.add_argument(
        "--strict-routing",
        action="store_true",
        help="count a turn the intent classifier sent to another lane as a failure (default: report it, not a KV failure)",
    )
    parser.add_argument("--restart-check", action="store_true", help="verify base states came from disk, then run one calendar and one music turn")
    parser.add_argument("--log-file", type=Path, default=None, help="server log to read the 'KV action lane ready' lines from")
    parser.add_argument("--parse-log", type=Path, default=None, help="only print the 'KV action lane ready' lines of this log and exit (no server needed)")
    parser.add_argument("--show-raw", action="store_true", help="print every frame (audio omitted) of failing turns")
    parser.add_argument("--out", type=Path, default=Path("docs/kv_verify_results.md"))
    args = parser.parse_args()
    if args.parse_log is not None:
        try:
            rows = parse_boot_log(args.parse_log)
        except OSError as exc:
            print(f"Could not read {args.parse_log}: {exc}")
            return 2
        print("| lane | source | tokens | prefilled | seconds |\n|---|---|---:|---:|---:|")
        for r in rows:
            print(f"| {r['lane']} | {r['source']} | {r['tokens']} | {r['prefilled']} | {r['seconds']:.2f} |")
        if not rows:
            print("(no 'KV action lane ready' lines found)")
        return 0 if rows else 1
    http = args.http or re.sub(r"^ws", "http", args.url).rsplit("/ws/", 1)[0]

    try:
        status = http_json(f"{http}/api/status")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"Cannot reach {http}/api/status: {exc}")
        return 2
    if not status.get("kv_lanes", {}).get("enabled", True):
        print("The server reports KV lanes disabled (set NOVA_KV_LANES=1 and NOVA_LLM_MODEL=qwen2.5-3b).")
        return 2

    report: list[str] = ["# KV lane verification", "", f"- server: {http}", f"- session: {args.session}", ""]
    ok = True

    if args.restart_check:
        ready, waited, counters = wait_ready(http, args.expect_lanes, args.ready_timeout)
        from_disk = int(counters.get("base_loaded", 0))
        built = int(counters.get("base_saved", 0))
        checks = [
            ("action lanes ready", ready),
            ("all prefixes loaded from disk", from_disk >= args.expect_lanes),
            ("none rebuilt (base_saved == 0)", built == 0),
            ("no prefill tokens spent", int(counters.get("warm_prefill_tokens", 0)) == 0),
            ("no invalid base states", int(counters.get("base_invalid", 0)) == 0),
        ]
        report += [f"## Restart check (waited {waited:.1f}s for the lanes)", "", "| check | result |", "|---|---|"]
        for name, passed in checks:
            report.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
            ok &= passed
        report += ["", f"kv_lanes counters: `{json.dumps(counters)}`", ""]
        if built >= args.expect_lanes and from_disk == 0:
            report += [
                "**This was a first boot, not a restart**: the prefixes were built and saved (`base_saved`), not loaded. "
                "Stop the server and start it again WITHOUT deleting `data/kv_base`, then rerun `--restart-check`.",
                "",
            ]
        if args.log_file is not None:
            try:
                rows = parse_boot_log(args.log_file)
            except OSError as exc:
                rows = []
                report += [f"### Boot log", "", f"Could not read `{args.log_file}`: {exc}", ""]
                print(f"Could not read the log file {args.log_file}: {exc}")
            else:
                report += ["### Boot log", "", "| lane | source | tokens | prefilled | seconds |", "|---|---|---:|---:|---:|"]
                report += [f"| {r['lane']} | {r['source']} | {r['tokens']} | {r['prefilled']} | {r['seconds']:.2f} |" for r in rows]
                report.append("")
            if not rows and not report[-2].startswith("Could not read"):
                report += [
                    f"(no 'KV action lane ready' lines found in {args.log_file}; "
                    "it must be the server's log of the boot you are checking)",
                    "",
                ]
        sequence = [SEQUENCE[1], SEQUENCE[4]]
    else:
        if not args.no_wait:
            ready, waited, _ = wait_ready(http, args.expect_lanes, args.ready_timeout)
            print(f"action lanes ready={ready} after {waited:.1f}s")
        sequence = SEQUENCE

    before = lane_counters(http)
    turns = asyncio.run(run_sequence(args, sequence))
    after = lane_counters(http)

    report += ["## Turns", ""] + turns_table(turns) + [""]
    for t in turns:
        if t.verdict != "PASS" and t.raw:
            report += [f"Frames of turn {t.index}:", "```", *[json.dumps(frame, ensure_ascii=False) for frame in t.raw], "```", ""]
    inconclusive = [t.index for t in turns if t.verdict == "INCONCLUSIVE"]
    ok &= not any(t.verdict == "FAIL" for t in turns)
    ok &= not (args.strict_routing and inconclusive)

    watched = [
        "base_loaded", "base_saved", "base_invalid", "cold_starts", "restore_failures", "evictions", "swaps_in",
        "swaps_out", "warm_prefill_tokens", "compactions", "overflows", "strip_ok", "strip_skipped", "strip_failed",
        "strip_tokens_removed",
    ]
    report += ["## Lane counters (before -> after)", "", "| counter | before | after |", "|---|---:|---:|"]
    report += [f"| {name} | {before.get(name, '-')} | {after.get(name, '-')} |" for name in watched if name in before or name in after]
    if not ok:
        overall = "FAIL"
    elif inconclusive:
        overall = (
            f"PASS, with {len(inconclusive)} turn(s) {inconclusive} routed to another lane by the intent classifier "
            "(not a KV failure; fix the classifier or rephrase the turn)"
        )
    else:
        overall = "PASS"
    report += ["", f"**Overall: {overall}**", ""]

    text = "\n".join(report)
    print("\n" + text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
