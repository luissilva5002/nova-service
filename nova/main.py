"""
nova/main.py
Main server event loop: Audio In -> STT -> Brain (tool routing) -> Skill
Execution -> TTS -> Audio Out. Exposes:

  GET  /                    Web UI dashboard (static/index.html)
  GET  /api/status          Health & model status
  GET  /api/skills          List registered skills/tools
  POST /api/ingest/{id}     Trigger project ingestion pipeline
  WS   /ws/chat             Bi-directional streaming pipe (text + audio)
"""
import base64
import json
import logging
import re
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from nova.config import WEB_UI_DIR, HOST_PROJECTS_DIR
from nova.brain.llm_engine import llm_engine
from nova.brain.stt_engine import stt_engine
from nova.brain.tts_engine import tts_engine
from nova.brain.tool_router import tool_router
from nova.memory.core_store import core_store
from nova.memory import vault_store
from nova.memory.memory_agent import memory_agent
from nova.memory.session_memory import (
    append_turn,
    clear_session,
    get_context,
    get_session_snapshot,
    get_session_summary,
)
from nova.ingestion.pipeline import ingest_project

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("nova.main")

app = FastAPI(title="NOVA Core API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_UI_DIR)), name="static")


@app.on_event("startup")
async def on_startup():
    logger.info("NOVA booting...")
    llm_engine.load()
    stt_engine.load()
    tts_engine.load()
    memory_agent.set_engine(llm_engine)
    tool_router.discover_skills()
    logger.info("NOVA ready. Skills: %s", tool_router.list_skills())


@app.get("/", response_class=HTMLResponse)
async def get_web_ui():
    """Serves the main web browser dashboard."""
    index_path = WEB_UI_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>NOVA</h1><p>web_ui/index.html not found.</p>")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/api/status")
async def get_status():
    """Endpoint for checking NOVA system health & RAM usage."""
    return {
        "status": "online",
        "brain_loaded": llm_engine._loaded,
        "stt_loaded": stt_engine._loaded,
        "tts_loaded": tts_engine._loaded,
        "active_project": vault_store.get_profile_fields().get("active_project"),
        "voices_available": tts_engine.list_available_voices(),
        "models": llm_engine.list_models(),
    }


@app.get("/api/skills")
async def get_skills():
    return {"skills": tool_router.list_skills()}


@app.get("/api/memory/graph")
async def get_memory_graph():
    """Export the persistent Obsidian-vault graph in a UI-friendly shape."""
    return vault_store.get_memory_graph()


@app.get('/api/memory/entity/{entity_id}')
async def get_memory_entity(entity_id: str):
    """Return full note details for a given graph node id.

    The graph node ids are deterministic SHA1 hashes of the vault-relative
    note path (see vault_store.get_memory_graph), so we compute the same id
    and return structured info the client can display.
    """
    import hashlib
    from pathlib import Path as _Path

    for category in ("preferences", "knowledge"):
        notes = vault_store.list_notes(category)
        for note in notes:
            try:
                rel = _Path(note["path"]).relative_to(vault_store.PERSISTENT_MEMORY_VAULT_DIR)
            except Exception:
                rel = _Path(note["path"])
            nid = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()
            if nid == entity_id:
                raw = vault_store.read_note(category, note["box"], note["subcategory"])
                frontmatter = {}
                body = ""
                if raw:
                    try:
                        fm, body = vault_store._split_frontmatter(raw)
                        frontmatter = fm or {}
                    except Exception:
                        body = raw
                excerpt = (body.strip().splitlines()[0] if body and body.strip() else "")[:800]
                return {
                    "id": nid,
                    "label": note.get("title") or note.get("box"),
                    "type": category,
                    "path": note.get("path"),
                    "tags": frontmatter.get("tags", []),
                    "frontmatter": {k: v for k, v in frontmatter.items() if k not in {"linked_notes"}},
                    "linked_notes": frontmatter.get("linked_notes", []),
                    "excerpt": excerpt,
                    "body": body,
                }
    raise HTTPException(status_code=404, detail="Entity not found")


@app.post("/api/models/{model_id}")
async def select_model(model_id: str):
    """Unload the active local model and load an installed selection."""
    try:
        return llm_engine.select_model(model_id)
    except ValueError as exc:
        status_code = 404 if str(exc).startswith("Unknown model:") else 500
        raise HTTPException(status_code=status_code, detail=f"Could not load model: {exc}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - report a failed local model load to the UI
        raise HTTPException(status_code=500, detail=f"Could not load model: {exc}") from exc


@app.post("/api/ingest/{project_id}")
async def post_ingest(project_id: str):
    """Triggers the 3-step dynamic ingestion pipeline for a project folder
    mounted read-only at HOST_PROJECTS_DIR/<project_id>."""
    result = await ingest_project(project_id, llm_engine, HOST_PROJECTS_DIR)
    return result


def _extract_tool_call_from_text(content: str) -> Optional[dict]:
    """
    Recovers a tool call the model emitted as free-form text instead of
    a structured `tool_calls` field (small models like Qwen3-0.6B often
    do this instead of honoring the grammar-constrained tool-call API).

    Handles the common case where the model doubles the outer JSON
    braces, e.g.:
        <tool_call>{{"name": "x", "arguments": {...}}}</tool_call>
    which is syntactically invalid JSON as-is (json.loads raises on the
    doubled braces) - so we peel one layer of outer braces and retry
    before giving up.
    """
    match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.S)
    raw = match.group(1).strip() if match else content.strip()

    candidates = [raw]
    if raw.startswith("{{") and raw.endswith("}}"):
        candidates.append(raw[1:-1].strip())

    first, last = raw.find("{"), raw.rfind("}")
    if first != -1 and last > first:
        trimmed = raw[first:last + 1]
        candidates.append(trimmed)
        if trimmed.startswith("{{") and trimmed.endswith("}}"):
            candidates.append(trimmed[1:-1].strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "name" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def _extract_name_from_text(text: str) -> Optional[str]:
    patterns = [
        r"\bmy name is\s+([A-Za-z][A-Za-z'\- ]{1,40})",
        r"\bi am\s+([A-Za-z][A-Za-z'\- ]{1,40})",
        r"\bcall me\s+([A-Za-z][A-Za-z'\- ]{1,40})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            extracted = match.group(1).strip().rstrip(".?! ")
            if extracted.lower().startswith("working "):
                continue
            return extracted
    return None


def _select_relevant_facts(user_text: str, facts: dict) -> dict:
    lower = user_text.strip().lower()
    relevant: dict = {}
    if re.search(r"\bwhat\s+(?:is|'s)\s+my\s+name\b|\bwho\s+am\s+i\b|\bwhat\s+is\s+my\s+full\s+name\b", lower):
        for key in ("name", "personal_info"):
            if key in facts:
                relevant[key] = facts[key]
    if re.search(r"\bwhat\s+(?:is|'s)\s+my\s+project\b|\bwhat\s+project\s+am\s+i\s+working\s+on\b|\bwhat\s+am\s+i\s+working\s+on\b", lower):
        for key in ("active_project", "project", "project_fact"):
            if key in facts:
                relevant[key] = facts[key]
    if re.search(r"\bwhat\s+(?:is|'s)\s+my\s+favorite\s+artist\b|\bwho\s+is\s+my\s+favorite\s+artist\b", lower):
        for key in ("favorite_artist", "artist"):
            if key in facts:
                relevant[key] = facts[key]
    return relevant


def _answer_from_memory(user_text: str, facts: dict, session_history: list[dict]) -> Optional[str]:
    lower = user_text.strip().lower()

    if re.search(r"\bwhat\s+(?:is|'s)\s+my\s+name\b|\bwho\s+am\s+i\b|\bwhat\s+is\s+my\s+full\s+name\b", lower):
        name = facts.get("name")
        if not name:
            personal = facts.get("personal_info", {})
            name = personal.get("name") if isinstance(personal, dict) else None
        if not name:
            for turn in reversed(session_history):
                if turn.get("role") == "user":
                    extracted = _extract_name_from_text(str(turn.get("content", "")))
                    if extracted:
                        name = extracted
                        break
        if name:
            return f"Your name is {name}."

    if re.search(r"\bwhat\s+(?:is|'s)\s+my\s+project\b|\bwhat\s+project\s+am\s+i\s+working\s+on\b|\bwhat\s+am\s+i\s+working\s+on\b", lower):
        project = facts.get("active_project") or facts.get("project") or facts.get("project_fact")
        if project:
            return f"Your active project is {project}."

    if re.search(r"\bwhat\s+(?:is|'s)\s+my\s+favorite\s+artist\b|\bwho\s+is\s+my\s+favorite\s+artist\b", lower):
        artist = facts.get("favorite_artist") or facts.get("artist")
        if artist:
            return f"Your favorite artist is {artist}."

    return None


def _handle_session_commands(user_text: str, session_id: str) -> Optional[str]:
    """Handle one-shot session commands that should not go through the LLM."""
    lower = user_text.strip().lower()
    if re.search(r"\b(clear|reset)\b.*\b(chat|conversation|history)\b", lower):
        clear_session(session_id)
        return "I cleared this chat history."

    if re.search(r"\b(show|display|what is)\b.*\b(session|chat)\b.*\b(context|memory|history)\b", lower):
        snapshot = get_session_snapshot(session_id, limit=20)
        return json.dumps({
            "summary": snapshot["summary"],
            "history": snapshot["history"],
        }, ensure_ascii=False)

    if re.search(r"\bshow\b.*\bsummary\b", lower) or re.search(r"\bcurrent\b.*\bsummary\b", lower):
        return json.dumps({"summary": get_session_summary(session_id)}, ensure_ascii=False)

    if re.search(r"\bremember\b.*\bproject\b.*\bfact\b", lower) or re.search(r"\bproject\b.*\bfact\b", lower):
        matched = re.search(r"(?:project\s+fact|remember\s+(?:this\s+as\s+)?(?:a\s+)?project\s+fact)[:\s]+(.+)", user_text, re.I)
        if matched:
            fact = matched.group(1).strip().rstrip(".?! ")
            if fact:
                vault_store.write_knowledge("project_notes", fact)
                return "I saved that as a project fact in my persistent memory."

    # Generic remember/save handling is intentionally disabled here.
    # The explicit remember workflow must be resolved by the LLM-based memory
    # agent using surrounding chat context, not by a regex that captures the
    # trailing fragment after "remember".
    return None


async def run_pipeline(user_text: str, session_id: str = "default") -> tuple[str, bytes, dict, Optional[dict]]:
    """
    Runs one full NOVA turn on already-transcribed text:
      Brain (tool intent) -> [Skill execution + feedback loop] -> reply text
    Returns (reply_text, synthesized_audio_bytes, LLM generation metrics,
    client_action). client_action is an optional dict (e.g. {"type":
    "open_url", "url": ..., "message": ...}) the WebSocket handler
    forwards to the client so it can act on it (e.g. open a deep link).
    Kept as a standalone function so both the WebSocket handler and any
    future REST /api/chat endpoint can reuse it.
    """
    session_command = _handle_session_commands(user_text, session_id)
    if session_command is not None:
        reply_text = session_command
        generation_metrics = {"completion_tokens": 0, "elapsed_seconds": 0.0, "tokens_per_second": 0.0}
        audio_bytes = b""
        client_action = None
        append_turn(session_id, "user", user_text)
        append_turn(session_id, "assistant", reply_text)
        core_store.log_message("user", user_text)
        core_store.log_message("nova", reply_text)
        audio_chunks = []
        async for chunk in tts_engine.synthesize_stream(reply_text):
            audio_chunks.append(chunk)
        audio_bytes = b"".join(audio_chunks)
        return reply_text, audio_bytes, generation_metrics, client_action

    core_store.log_message("user", user_text)
    await memory_agent.process_turn(user_text)

    session_history = get_context(session_id, limit=12)
    session_summary = get_session_summary(session_id)
    facts = vault_store.get_profile_fields()
    direct_answer = _answer_from_memory(user_text, facts, session_history)
    if direct_answer is not None:
        reply_text = direct_answer
        generation_metrics = {"completion_tokens": 0, "elapsed_seconds": 0.0, "tokens_per_second": 0.0}
        audio_bytes = b""
        client_action = None
        append_turn(session_id, "user", user_text)
        append_turn(session_id, "assistant", reply_text)
        core_store.log_message("nova", reply_text)
        audio_chunks = []
        async for chunk in tts_engine.synthesize_stream(reply_text):
            audio_chunks.append(chunk)
        audio_bytes = b"".join(audio_chunks)
        return reply_text, audio_bytes, generation_metrics, client_action

    relevant_facts = _select_relevant_facts(user_text, facts)
    vault_context = vault_store.get_context_snippet(query=user_text)
    memory_context = json.dumps(
        {
            "chat_context_summary": session_summary,
            "recent_chat_turns": session_history,
            "relevant_profile_fields": relevant_facts,
            "preferences": vault_context["preferences"],
            "relevant_knowledge": vault_context["relevant_knowledge"],
        },
        ensure_ascii=False,
    )
    decision = await llm_engine.generate_with_tools(
        user_text=user_text,
        tool_schemas=tool_router.tool_schemas,
        memory_context=memory_context,
    )
    # Debug: log the LLM decision so we can trace tool_call vs text paths
    logger.info("LLM decision: %s", json.dumps(decision) if isinstance(decision, dict) else str(decision))

    # Some models output the tool call wrapped in text (e.g. a <tool_call> block)
    # instead of using the structured tool_calls API field. Detect and recover
    # a tool_call embedded in the text output so the skill dispatch path still runs.
    if isinstance(decision, dict) and decision.get("type") == "text":
        content = decision.get("content", "")
        if "<tool_call" in content:
            parsed = _extract_tool_call_from_text(content)
            if parsed:
                decision = {
                    "type": "tool_call",
                    "name": parsed.get("name"),
                    "arguments": parsed.get("arguments", {}),
                    "metrics": decision.get("metrics", {}),
                }
                logger.info("Recovered tool_call from LLM text output: %s", json.dumps(decision))
            else:
                logger.warning("Found <tool_call> in output but failed to parse JSON: %r", content)

    audio_bytes = b""
    generation_metrics = {}
    client_action = None

    if isinstance(decision, dict) and decision.get("type") == "tool_call":
        try:
            tool_result = tool_router.dispatch(decision["name"], decision["arguments"])
        except Exception as exc:
            logger.exception("Tool execution failed for %s with arguments %s", decision.get("name"), decision.get("arguments"))
            reply_text = "I couldn't complete that task. Please try again."
            generation_metrics = decision.get("metrics", {}) if isinstance(decision, dict) else {}
            tool_result = None
        else:
            # Plain-text tool results should be returned directly to the user.
            # A second LLM rewrite is useful for audio/open_url actions, but it
            # causes nonsense like "calendar updated" when the user only wanted
            # a read/query result from a tool.
            try:
                if isinstance(tool_result, dict):
                    # Audio payload produced by the tool
                    if tool_result.get("type") == "audio":
                        try:
                            from pathlib import Path
                            audio_path = Path(tool_result.get("path", ""))
                            if audio_path.exists():
                                audio_bytes = audio_path.read_bytes()
                            else:
                                logger.warning("Tool returned audio path that does not exist: %s", audio_path)
                        except Exception:
                            logger.exception("Failed to read audio file returned by tool.")
                        user_prompt_for_llm = tool_result.get("message") or f"Executed tool {decision['name']} (audio)."
                    elif tool_result.get("type") == "open_url":
                        # Instruct client to open a URL (e.g. YouTube Music deep link)
                        client_action = tool_result
                        user_prompt_for_llm = tool_result.get("message") or f"Executed tool {decision['name']} (open_url)."
                    else:
                        # Generic dict result: prefer human 'message' key if present
                        user_prompt_for_llm = tool_result.get("message") if tool_result.get("message") else json.dumps(tool_result)
                        reply_text = str(user_prompt_for_llm)
                        generation_metrics = decision.get("metrics", {})
                    if tool_result.get("type") in {"audio", "open_url"}:
                        # Response Feedback Loop: ask the brain to confirm / describe the action
                        reply_text, generation_metrics = await llm_engine.generate_raw_with_metrics(
                            system_prompt="Confirm this action result to the user in one natural spoken sentence. If the action produced audio, also say that playback has started.",
                            user_prompt=user_prompt_for_llm,
                            max_tokens=64,
                        )
                        if not isinstance(reply_text, str):
                            reply_text = str(reply_text)
                        reply_text = reply_text.strip()
                else:
                    # Tool result is a string: use it directly so reads and updates are not
                    # reworded by a second model pass into generic or wrong outputs.
                    reply_text = str(tool_result).strip()
                    generation_metrics = decision.get("metrics", {})
            except Exception:
                logger.exception("Error normalizing tool result for feedback loop.")
                reply_text = str(tool_result).strip() if tool_result is not None else "I couldn't complete that task. Please try again."
                generation_metrics = decision.get("metrics", {})

    else:
        reply_text = decision.get("content") if isinstance(decision, dict) else str(decision)
        generation_metrics = decision.get("metrics", {}) if isinstance(decision, dict) else {}

    append_turn(session_id, "user", user_text)
    append_turn(session_id, "assistant", reply_text)
    core_store.log_message("nova", reply_text)

    # If the tool produced audio bytes, prefer them over TTS.
    if audio_bytes:
        return reply_text, audio_bytes, generation_metrics, client_action

    # Otherwise synthesize TTS as before.
    audio_chunks = []
    async for chunk in tts_engine.synthesize_stream(reply_text):
        audio_chunks.append(chunk)
    audio_bytes = b"".join(audio_chunks)

    return reply_text, audio_bytes, generation_metrics, client_action


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """
    Bi-directional streaming pipe for the Web UI and the Flutter mobile app.
    Accepts two message shapes (JSON-encoded text frames):
      {"type": "text",  "text": "..."}                  -> typed input
      {"type": "audio", "audio_b64": "...", "final": true} -> voice input
    Replies with:
      {"type": "text", "text": "..."}                   -> NOVA's spoken reply text
      {"type": "metrics", "llm": {...}}                  -> generation throughput stats
      {"type": "open_url", "url": "...", "text": "..."}  -> ask client to navigate/open a link
      {"type": "audio", "audio_b64": "..."}              -> synthesized speech (streamed)
    """
    await websocket.accept()
    query_session_id = websocket.query_params.get("session_id") if websocket.query_params else None
    session_id = query_session_id or getattr(websocket, "_nova_session_id", None) or str(uuid.uuid4())
    setattr(websocket, "_nova_session_id", session_id)
    logger.info("Client connected to /ws/chat with session_id=%s", session_id)
    audio_buffer = bytearray()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                # Backward-compatible plain-text fallback.
                message = {"type": "text", "text": raw}

            msg_type = message.get("type", "text")
            if msg_type == "session" and isinstance(message.get("session_id"), str) and message["session_id"].strip():
                session_id = message["session_id"].strip()
                setattr(websocket, "_nova_session_id", session_id)
                logger.info("Updated WS session_id from client handshake to %s", session_id)
                continue

            if msg_type == "audio":
                audio_buffer.extend(base64.b64decode(message.get("audio_b64", "")))
                if not message.get("final"):
                    continue  # keep buffering until the client signals end-of-utterance
                user_text = stt_engine.transcribe_media(bytes(audio_buffer))
                logger.info("Transcribed user text: %r", user_text)
                audio_buffer.clear()
            else:
                if isinstance(message.get("session_id"), str) and message["session_id"].strip():
                    session_id = message["session_id"].strip()
                    setattr(websocket, "_nova_session_id", session_id)
                user_text = message.get("text", "")

            if not user_text.strip():
                continue

            reply_text, audio_bytes, generation_metrics, client_action = await run_pipeline(user_text, session_id=session_id)

            # Log what we are about to send to the client (debug)
            logger.info(
                "Sending to WS client: text=%s, open_url=%s, audio_bytes=%s",
                reply_text,
                client_action.get("url") if client_action and isinstance(client_action, dict) else None,
                'yes' if audio_bytes else 'no',
            )

            await websocket.send_text(json.dumps({"type": "text", "text": reply_text}))
            await websocket.send_text(json.dumps({"type": "metrics", "llm": generation_metrics}))

            # If the tool asked the client to open a URL, send that action
            if client_action and isinstance(client_action, dict) and client_action.get("type") == "open_url":
                logger.info("Forwarding open_url to client: %s", client_action.get("url"))
                await websocket.send_text(json.dumps({
                    "type": "open_url",
                    "url": client_action.get("url"),
                    "text": client_action.get("message"),
                }))

            if audio_bytes:
                await websocket.send_text(json.dumps({
                    "type": "audio",
                    "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
                }))

    except WebSocketDisconnect:
        logger.info("Client disconnected.")