#!/usr/bin/env python3
"""Benchmark user-visible chat timings over Nova's WebSocket API."""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHAT_MESSAGES = [
    "Why does the moon appear larger near the horizon than it does high in the sky?",
    "What simple habits help keep indoor plants healthy through a dry winter?",
    "How can someone make a small room feel brighter without replacing the furniture?",
    "Why do some lakes freeze on the surface while water below remains liquid?",
    "What are a few useful ways to reduce food waste during a busy week?",
    "How does regular walking help improve general health over time?",
    "Why do leaves change color as autumn progresses in cooler regions?",
    "What makes a good checklist for packing a light day bag?",
    "How can I organize a relaxed afternoon walk through an unfamiliar neighborhood?",
    "Why does bread become stale faster in a refrigerator than at room temperature?",
    "What are practical ways to make a morning routine feel less rushed?",
    "How do clouds form when warm air rises into cooler parts of the atmosphere?",
    "What should someone look for when choosing a comfortable pair of walking shoes?",
    "Why are some bridges built with triangular supports instead of solid walls?",
    "How can a household use less water while keeping a small garden healthy?",
    "What is an easy method for comparing two travel routes before leaving home?",
    "Why does adding salt change the temperature at which water freezes?",
    "What details make a neighborhood park pleasant for a quiet break?",
    "How can I plan a simple picnic that stays enjoyable in changeable weather?",
    "Why do people sometimes see flashes of light during a distant thunderstorm?",
    "What are a few ways to keep a desk tidy during a demanding workday?",
    "How does sunlight help plants turn water and carbon dioxide into sugars?",
    "What makes a public library useful for people who enjoy learning new subjects?",
    "Why do metal objects feel colder than wooden objects in the same room?",
    "How can someone choose a scenic route that is still easy to walk?",
    "What small changes can make a kitchen easier to clean after cooking?",
    "Why are mountain temperatures often lower than temperatures in nearby valleys?",
    "How do bees help flowering plants produce seeds and fruit?",
    "What should I consider when arranging a comfortable reading corner at home?",
    "Why can a clear night become colder than a cloudy night?",
    "How can a simple weekly meal plan leave room for unexpected changes?",
    "What makes a walking path accessible and pleasant for a wide range of people?",
    "Why does a spoon look bent when placed partly underwater in a glass?",
    "How can I make a short visit to a local museum more engaging?",
    "What everyday objects are useful for measuring ingredients without special equipment?",
    "Why do shadows change direction and length during the day?",
    "How can someone make a small balcony more comfortable for sitting outdoors?",
    "What helps keep fresh herbs usable for several days after they are picked?",
    "Why does the air often smell different just before or after rain?",
    "What are thoughtful ways to make a weekend outing comfortable for everyone?",
]
TURN_TIMEOUT_SECONDS = 120.0


def _status_url(websocket_url: str) -> str:
    parsed = urllib.parse.urlsplit(websocket_url)
    if parsed.scheme not in ("ws", "wss"):
        raise ValueError("--url must use ws:// or wss://")
    scheme = "https" if parsed.scheme == "wss" else "http"
    return urllib.parse.urlunsplit((scheme, parsed.netloc, "/api/status", "", ""))


def _read_status(websocket_url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        _status_url(websocket_url),
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("/api/status did not return a JSON object")
    return payload


def _active_model(status: dict[str, Any]) -> str | None:
    models = status.get("models")
    if isinstance(models, list):
        for model in models:
            if isinstance(model, dict) and model.get("active"):
                return model.get("id") or model.get("label")
    return status.get("active_model_id")


def _lanes_enabled(lanes_status: Any) -> bool | None:
    if isinstance(lanes_status, dict):
        if isinstance(lanes_status.get("enabled"), bool):
            return lanes_status["enabled"]
        if lanes_status:
            return True
        return None
    if isinstance(lanes_status, bool):
        return lanes_status
    return None


def _turn_messages(long_run: bool, turns: int | None) -> list[str]:
    count = 40 if long_run else (turns if turns is not None else 16)
    if count < 1:
        raise ValueError("--turns must be at least 1")
    return [CHAT_MESSAGES[index % len(CHAT_MESSAGES)] for index in range(count)]


def _new_row(index: int, user_text: str) -> dict[str, Any]:
    return {
        "turn": index,
        "user_text": user_text,
        "user_words": len(user_text.split()),
        "first_delta_s": None,
        "total_s": None,
        "metrics": {},
        "failed": False,
        "error": None,
    }


def _record_frame(row: dict[str, Any], frame: Any, started: float) -> bool:
    """Consume one server frame; return True once the turn is complete."""
    if isinstance(frame, bytes):
        frame = frame.decode("utf-8", errors="replace")
    if isinstance(frame, str):
        try:
            frame = json.loads(frame)
        except json.JSONDecodeError:
            return False
    if not isinstance(frame, dict):
        return False

    frame_type = frame.get("type")
    now = time.perf_counter()
    if frame_type in ("text_delta", "text") and row["first_delta_s"] is None:
        row["first_delta_s"] = max(0.0, now - started)
    if frame_type == "metrics":
        metrics = frame.get("llm", frame.get("metrics", {}))
        row["metrics"] = metrics if isinstance(metrics, dict) else {}
        row["total_s"] = max(0.0, now - started)
        return True
    if frame_type == "done":
        metrics = frame.get("metrics", {})
        row["metrics"] = metrics if isinstance(metrics, dict) else {}
        row["total_s"] = max(0.0, now - started)
        return True
    if frame_type == "audio_done":
        row["total_s"] = max(0.0, now - started)
        return True
    return False


async def _benchmark_websockets(
    websocket_url: str,
    session_id: str,
    messages: list[str],
) -> list[dict[str, Any]]:
    import websockets

    rows: list[dict[str, Any]] = []
    base = urllib.parse.urlsplit(websocket_url)
    query = urllib.parse.parse_qsl(base.query, keep_blank_values=True)
    query.append(("session_id", session_id))
    uri = urllib.parse.urlunsplit(
        (base.scheme, base.netloc, base.path or "/", urllib.parse.urlencode(query), base.fragment)
    )
    connection = None
    connect = websockets.connect

    try:
        for index, user_text in enumerate(messages, start=1):
            row = _new_row(index, user_text)
            rows.append(row)
            try:
                if connection is None:
                    connection = await connect(uri, open_timeout=10, close_timeout=2)
                else:
                    await _drain_async_frames(connection)
                started = time.perf_counter()
                await connection.send(
                    json.dumps({"type": "text", "text": user_text, "stream": True})
                )

                async def receive_until_complete() -> None:
                    while True:
                        frame = await connection.recv()
                        if _record_frame(row, frame, started):
                            return

                await asyncio.wait_for(
                    receive_until_complete(),
                    timeout=TURN_TIMEOUT_SECONDS,
                )
                row["failed"] = False
            except asyncio.TimeoutError:
                row["failed"] = True
                row["error"] = f"Timed out after {TURN_TIMEOUT_SECONDS:g} seconds"
                if connection is not None:
                    await connection.close()
                    connection = None
            except Exception as exc:
                row["failed"] = True
                row["error"] = f"{type(exc).__name__}: {exc}"
                if connection is not None:
                    try:
                        await connection.close()
                    except Exception:
                        pass
                    connection = None
    finally:
        if connection is not None:
            await connection.close()
    return rows


async def _drain_async_frames(connection) -> None:
    while True:
        try:
            await asyncio.wait_for(connection.recv(), timeout=0.01)
        except asyncio.TimeoutError:
            return
        except Exception:
            return


def _benchmark_websocket_client(
    websocket_url: str,
    session_id: str,
    messages: list[str],
) -> list[dict[str, Any]]:
    import websocket

    base = urllib.parse.urlsplit(websocket_url)
    query = urllib.parse.parse_qsl(base.query, keep_blank_values=True)
    query.append(("session_id", session_id))
    uri = urllib.parse.urlunsplit(
        (base.scheme, base.netloc, base.path or "/", urllib.parse.urlencode(query), base.fragment)
    )
    rows: list[dict[str, Any]] = []
    connection = None
    try:
        for index, user_text in enumerate(messages, start=1):
            row = _new_row(index, user_text)
            rows.append(row)
            try:
                if connection is None:
                    connection = websocket.create_connection(uri, timeout=10)
                else:
                    connection.settimeout(0.01)
                    while True:
                        try:
                            connection.recv()
                        except websocket.WebSocketTimeoutException:
                            break
                        except Exception:
                            break
                connection.settimeout(TURN_TIMEOUT_SECONDS)
                started = time.perf_counter()
                connection.send(
                    json.dumps({"type": "text", "text": user_text, "stream": True})
                )
                while True:
                    frame = connection.recv()
                    if _record_frame(row, frame, started):
                        break
                row["failed"] = False
            except Exception as exc:
                row["failed"] = True
                row["error"] = f"{type(exc).__name__}: {exc}"
                if connection is not None:
                    connection.close()
                    connection = None
    finally:
        if connection is not None:
            connection.close()
    return rows


def _median_window(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    groups = (
        ("1-6", rows[:6]),
        ("7-12", rows[6:12]),
        ("13-end", rows[12:]),
    )
    medians: dict[str, float | None] = {}
    for label, group in groups:
        values = [
            row["first_delta_s"]
            for row in group
            if not row["failed"] and isinstance(row["first_delta_s"], (int, float))
        ]
        medians[label] = statistics.median(values) if values else None
    return medians


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [
        row
        for row in rows
        if not row["failed"] and isinstance(row["first_delta_s"], (int, float))
    ]
    maximum = max(successful, key=lambda row: row["first_delta_s"]) if successful else None
    return {
        "median_first_delta_s": _median_window(rows),
        "maximum_first_delta_s": maximum["first_delta_s"] if maximum else None,
        "maximum_first_delta_turn": maximum["turn"] if maximum else None,
        "turns_with_lane": sum(
            1 for row in rows if isinstance(row["metrics"], dict) and "lane" in row["metrics"]
        ),
        "failed_turns": sum(1 for row in rows if row["failed"]),
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("| turn | user words | first_delta_s | total_s | prompt_tokens | evaluated_tokens | lane? |")
    print("|---:|---:|---:|---:|---:|---:|:---|")
    for row in rows:
        metrics = row["metrics"]
        first = row["first_delta_s"]
        total = row["total_s"]
        fmt = lambda value: f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"
        if row["failed"]:
            first_display = row["error"] or "failed"
        else:
            first_display = fmt(first)
        print(
            f"| {row['turn']} | {row['user_words']} | {first_display} | {fmt(total)} | "
            f"{metrics.get('prompt_tokens', 'n/a')} | {metrics.get('evaluated_tokens', 'n/a')} | "
            f"{metrics.get('lane', 'n/a')} |"
        )


def _print_summary(summary: dict[str, Any]) -> None:
    print("\nSummary")
    for window, value in summary["median_first_delta_s"].items():
        formatted = f"{value:.3f}s" if value is not None else "n/a"
        print(f"- Median first-delta ({window}): {formatted}")
    if summary["maximum_first_delta_s"] is None:
        print("- Maximum first-delta: n/a")
    else:
        print(
            f"- Maximum first-delta: {summary['maximum_first_delta_s']:.3f}s "
            f"(turn {summary['maximum_first_delta_turn']})"
        )
    print(f"- Turns with a lane metric: {summary['turns_with_lane']}")
    print(f"- Failed turns: {summary['failed_turns']}")


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def _run_benchmark(
    websocket_url: str,
    label: str,
    messages: list[str],
    status_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        import websockets  # noqa: F401
    except ImportError:
        try:
            import websocket  # noqa: F401
        except ImportError as exc:
            print(
                "Install a WebSocket client with `python -m pip install websockets` "
                "or `python -m pip install websocket-client`.",
                file=sys.stderr,
            )
            raise RuntimeError("No WebSocket client package is installed.") from exc
        client = "websocket-client"
    else:
        client = "websockets"

    status = (
        status_override
        if status_override is not None
        else await asyncio.to_thread(_read_status, websocket_url)
    )
    lanes_value = status.get("kv_lanes")
    session_id = f"bench-{label}-{int(time.time())}"

    if client == "websocket-client":
        rows = await asyncio.to_thread(
            _benchmark_websocket_client,
            websocket_url,
            session_id,
            messages,
        )
    else:
        rows = await _benchmark_websockets(websocket_url, session_id, messages)

    summary = _summarize(rows)
    return {
        "schema_version": 1,
        "label": label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "websocket_url": websocket_url,
            "websocket_client": client,
            "requested_turns": len(messages),
        },
        "status": {
            "kv_lanes": lanes_value,
            "lanes_enabled": _lanes_enabled(lanes_value),
            "active_model": _active_model(status),
            "models": status.get("models"),
        },
        "rows": rows,
        "summary": summary,
    }


def _number(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"


def _compare(paths: list[str]) -> int:
    documents = []
    for filename in paths:
        try:
            documents.append(json.loads(Path(filename).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Could not read {filename}: {exc}", file=sys.stderr)
            return 1
    if any(not isinstance(doc, dict) or "rows" not in doc for doc in documents):
        print("Each comparison file must contain a benchmark result with rows.", file=sys.stderr)
        return 1

    left, right = documents
    left_label = left.get("label", Path(paths[0]).stem)
    right_label = right.get("label", Path(paths[1]).stem)
    print(f"| statistic | {left_label} | {right_label} |")
    print("|:--|--:|--:|")
    for group in ("1-6", "7-12", "13-end"):
        lvalue = left.get("summary", {}).get("median_first_delta_s", {}).get(group)
        rvalue = right.get("summary", {}).get("median_first_delta_s", {}).get(group)
        print(f"| median first-delta {group} | {_number(lvalue)} | {_number(rvalue)} |")
    for title, key in (
        ("maximum first-delta", "maximum_first_delta_s"),
        ("maximum turn", "maximum_first_delta_turn"),
    ):
        lvalue = left.get("summary", {}).get(key)
        rvalue = right.get("summary", {}).get(key)
        print(f"| {title} | {lvalue if lvalue is not None else 'n/a'} | {rvalue if rvalue is not None else 'n/a'} |")

    print("\nPer-turn first-delta (seconds)")
    left_rows = left.get("rows", [])
    right_rows = right.get("rows", [])
    max_turn = max(len(left_rows), len(right_rows))
    print(f"| turn | {left_label} | {right_label} |")
    print("|---:|---:|---:|")
    for index in range(max_turn):
        lvalue = left_rows[index].get("first_delta_s") if index < len(left_rows) else None
        rvalue = right_rows[index].get("first_delta_s") if index < len(right_rows) else None
        print(f"| {index + 1} | {_number(lvalue)} | {_number(rvalue)} |")
    return 0


async def _run_selftest() -> int:
    try:
        from websockets.asyncio.server import serve
    except ImportError:
        try:
            import websockets

            serve = websockets.serve
        except ImportError:
            print(
                "Self-test requires `websockets`; install it with "
                "`python -m pip install websockets`.",
                file=sys.stderr,
            )
            return 2

    async def fake_handler(connection, *_path):
        turn = 0
        async for _message in connection:
            turn += 1
            if turn == 1:
                await connection.send(json.dumps({"type": "text_delta", "text": "Hello"}))
                await connection.send(json.dumps({"type": "text", "text": "Hello there."}))
                await connection.send(
                    json.dumps(
                        {
                            "type": "metrics",
                            "llm": {
                                "prompt_tokens": 42,
                                "cache_hit_tokens": 30,
                                "evaluated_tokens": 12,
                                "ttft_seconds": 0.01,
                                "completion_tokens": 3,
                                "tokens_per_second": 100.0,
                                "lane": "chat:selftest",
                            },
                        }
                    )
                )
                await connection.send(json.dumps({"type": "audio_done"}))
            else:
                await connection.send(json.dumps({"type": "text", "text": "A plain reply."}))
                await connection.send(
                    json.dumps(
                        {
                            "type": "metrics",
                            "llm": {
                                "prompt_tokens": 55,
                                "cache_hit_tokens": 0,
                                "evaluated_tokens": 55,
                                "ttft_seconds": 0.02,
                                "completion_tokens": 4,
                                "tokens_per_second": 80.0,
                            },
                        }
                    )
                )

    async with serve(fake_handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        websocket_url = f"ws://127.0.0.1:{port}/ws/chat"
        result = await _run_benchmark(
            websocket_url,
            "selftest",
            [
                CHAT_MESSAGES[0],
                CHAT_MESSAGES[1],
            ],
            status_override={
                "kv_lanes": {"enabled": True, "cold_starts": 1},
                "models": [{"id": "qwen2.5-3b", "active": True}],
            },
        )
        _print_table(result["rows"])
        _print_summary(result["summary"])
        first, second = result["rows"]
        if (
            first["metrics"].get("lane") != "chat:selftest"
            or second["metrics"].get("evaluated_tokens") != 55
            or result["summary"]["turns_with_lane"] != 1
        ):
            print("Self-test failed: expected stream and non-stream frame shapes.", file=sys.stderr)
            return 1
        print("\nSelf-test passed: streaming, non-streaming, metrics, and audio_done frames handled.")
        return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Nova chat latency over its WebSocket endpoint."
    )
    parser.add_argument("--url", help="WebSocket URL, for example ws://HOST:8000/ws/chat")
    parser.add_argument("--turns", type=int, default=16, help="Number of turns (default: 16)")
    parser.add_argument("--long", action="store_true", help="Run 40 turns")
    parser.add_argument("--label", default="bench", help="Run label")
    parser.add_argument("--out", help="JSON output path")
    parser.add_argument("--compare", nargs=2, metavar=("FIRST.json", "SECOND.json"))
    parser.add_argument("--selftest", action="store_true", help="Run against local fake servers")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.compare:
        return _compare(args.compare)
    if args.selftest:
        try:
            return asyncio.run(_run_selftest())
        except RuntimeError as exc:
            if "No WebSocket client package" in str(exc):
                print(str(exc), file=sys.stderr)
                return 2
            raise
    if not args.url:
        _parser().error("--url is required unless --compare or --selftest is used")

    try:
        messages = _turn_messages(args.long, args.turns)
        result = asyncio.run(_run_benchmark(args.url, args.label, messages))
    except RuntimeError as exc:
        if "No WebSocket client package" in str(exc):
            print(str(exc), file=sys.stderr)
            return 2
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"Benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    _print_table(result["rows"])
    _print_summary(result["summary"])
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = Path(args.out) if args.out else Path("docs/bench") / f"{args.label}-{timestamp}.json"
    try:
        _write_result(output, result)
    except OSError as exc:
        print(f"Could not save benchmark JSON to {output}: {exc}", file=sys.stderr)
        return 1
    print(f"\nSaved benchmark to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
