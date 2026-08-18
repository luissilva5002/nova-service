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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from nova.config import WEB_UI_DIR, HOST_PROJECTS_DIR
from nova.brain.llm_engine import llm_engine
from nova.brain.stt_engine import stt_engine
from nova.brain.tts_engine import tts_engine
from nova.brain.tool_router import tool_router
from nova.memory.core_store import core_store
from nova.memory.memory_agent import memory_agent
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
        "active_project": core_store.get_fact("active_project"),
        "voices_available": tts_engine.list_available_voices(),
    }


@app.get("/api/skills")
async def get_skills():
    return {"skills": tool_router.list_skills()}


@app.post("/api/ingest/{project_id}")
async def post_ingest(project_id: str):
    """Triggers the 3-step dynamic ingestion pipeline for a project folder
    mounted read-only at HOST_PROJECTS_DIR/<project_id>."""
    result = await ingest_project(project_id, llm_engine, HOST_PROJECTS_DIR)
    return result


async def run_pipeline(user_text: str) -> tuple[str, bytes, dict]:
    """
    Runs one full NOVA turn on already-transcribed text:
      Brain (tool intent) -> [Skill execution + feedback loop] -> reply text
    Returns (reply_text, synthesized_audio_bytes, LLM generation metrics).
    Kept as a standalone function so both the WebSocket handler and any
    future REST /api/chat endpoint can reuse it.
    """
    core_store.log_message("user", user_text)
    await memory_agent.process_turn(user_text)

    memory_context = json.dumps(core_store.all_facts())
    decision = await llm_engine.generate_with_tools(
        user_text=user_text,
        tool_schemas=tool_router.tool_schemas,
        memory_context=memory_context,
    )

    if decision["type"] == "tool_call":
        tool_result = tool_router.dispatch(decision["name"], decision["arguments"])
        # Response Feedback Loop: feed the tool's result back through the
        # brain so NOVA replies naturally instead of reading raw JSON.
        reply_text, generation_metrics = await llm_engine.generate_raw_with_metrics(
            system_prompt="Confirm this action result to the user in one natural spoken sentence.",
            user_prompt=tool_result,
            max_tokens=64,
        )
        reply_text = reply_text.strip() or tool_result
    else:
        reply_text = decision["content"]
        generation_metrics = decision.get("metrics", {})

    core_store.log_message("nova", reply_text)

    audio_chunks = []
    async for chunk in tts_engine.synthesize_stream(reply_text):
        audio_chunks.append(chunk)
    audio_bytes = b"".join(audio_chunks)

    return reply_text, audio_bytes, generation_metrics


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """
    Bi-directional streaming pipe for the Web UI and the Flutter mobile app.
    Accepts two message shapes (JSON-encoded text frames):
      {"type": "text",  "text": "..."}                  -> typed input
      {"type": "audio", "audio_b64": "...", "final": true} -> voice input
    Replies with:
      {"type": "text", "text": "..."}                   -> NOVA's spoken reply text
      {"type": "audio", "audio_b64": "..."}              -> synthesized speech (streamed)
    """
    await websocket.accept()
    logger.info("Client connected to /ws/chat")
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

            if msg_type == "audio":
                audio_buffer.extend(base64.b64decode(message.get("audio_b64", "")))
                if not message.get("final"):
                    continue  # keep buffering until the client signals end-of-utterance
                user_text = stt_engine.transcribe(bytes(audio_buffer))
                audio_buffer.clear()
            else:
                user_text = message.get("text", "")

            if not user_text.strip():
                continue

            reply_text, audio_bytes, generation_metrics = await run_pipeline(user_text)

            await websocket.send_text(json.dumps({"type": "text", "text": reply_text}))
            await websocket.send_text(json.dumps({"type": "metrics", "llm": generation_metrics}))
            if audio_bytes:
                await websocket.send_text(json.dumps({
                    "type": "audio",
                    "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
                }))

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
