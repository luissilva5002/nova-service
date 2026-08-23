"""
nova/brain/intent_router.py

LLM-based intent classification, run as an explicit first pass before any
tool schema or memory context is assembled. Keyword/regex classification
was replaced because lexical rules misfire unpredictably - e.g. "Hey" vs
"Hey, <real request>" can't be reliably told apart by pattern alone, and
short constrained classification is exactly the kind of task small local
models handle well when it's isolated as its own turn.

Two-pass design:
  Pass 1 (this module): classify_intent_llm() - asks the brain ONE
    question: "which single category does this turn belong to?" - no
    tools, minimal tokens, nothing else happens in this call. The
    model's attention is entirely on classification, not on also trying
    to decide what to do about it.
  Pass 2 (nova/main.py's run_pipeline): once the intent is known, a
    SEPARATE second LLM call actually does the work, with only that
    category's tools/context exposed.

A lightweight regex fallback exists ONLY as a safety net for when the
classification call fails outright (model not loaded, empty/garbled
output) - it is never the primary mechanism.
"""
import logging
import re

logger = logging.getLogger("nova.intent_router")

VALID_INTENTS = {
    "action_music",
    "action_calendar",
    "action_code",
    "memory_write",
    "memory_recall",
    "chat",
}

CLASSIFICATION_SYSTEM_PROMPT = (
    "You are an intent classifier. Read the user's message and output "
    "EXACTLY ONE of the following category names, nothing else - no "
    "punctuation, no explanation, no quotes:\n\n"
    "action_music - the user wants to play, pause, skip, resume, or control "
    "music/song/playlist playback right now.\n"
    "action_calendar - the user wants to create, view, update, or delete a "
    "calendar event, meeting, or reminder.\n"
    "action_code - the user wants to search or read source code / files in "
    "a project workspace.\n"
    "memory_write - the user is explicitly asking you to remember, save, "
    "note, or store a fact for later.\n"
    "memory_recall - the user is asking you to recall, look up, or answer "
    "using something you should already know about them or their projects "
    "(including questions about themselves, their hardware, their notes).\n"
    "chat - anything else: greetings, small talk, opinions, general "
    "conversation, or a question that does not require saved memory or a "
    "tool.\n\n"
    "Examples:\n"
    "\"Hey Nova\" -> chat\n"
    "\"Hey Nova, can you tell me what MCU we used in the Smasher?\" -> memory_recall\n"
    "\"Play some music\" -> action_music\n"
    "\"Remember that I prefer dark mode\" -> memory_write\n"
    "\"What's on my calendar today?\" -> action_calendar\n"
    "\"How are you?\" -> chat\n"
    "\"Tell me something about myself based on what you know\" -> memory_recall\n\n"
    "Respond with only the category name."
)

_TOOL_NAME_FILTERS = {
    "action_music": lambda name: name.startswith("ytm_"),
    "action_calendar": lambda name: name.startswith("gc_"),
    "action_code": lambda name: name in ("search_codebase", "read_file"),
    "memory_write": lambda name: name in ("update_user_preference", "update_user_knowledge"),
    "memory_recall": lambda name: name == "recall_note",
    "chat": lambda name: False,
}

# Safety-net only - used when the classification LLM call itself fails
# (model not loaded, empty output). Never the primary classification path.
_FALLBACK_MUSIC = re.compile(r"\b(play|pause|resume|skip|song|track|music|playlist)\b", re.I)
_FALLBACK_CALENDAR = re.compile(r"\b(calendar|event|meeting|appointment|schedule|remind me|reminder|agenda)\b", re.I)
_FALLBACK_CODE = re.compile(r"\b(codebase|source code|read file|search code|repository|repo)\b", re.I)
_FALLBACK_WRITE = re.compile(r"\b(remember|note that|save this|save to memory|store this in memory)\b", re.I)

_SMALL_TALK_PATTERNS = [
    r"^\s*(hey|hi|hello)(\s+nova)?[,.!]?\s*$",
    r"^\s*(how are you|what's up|whats up|how's it going)\s*\??\s*$",
]


def is_small_talk(user_text: str) -> bool:
    """Kept as a small helper for _select_relevant_facts/_answer_from_memory
    in main.py - NOT used for primary intent classification anymore."""
    text = user_text.strip().lower()
    if len(text.split()) > 6:
        return False
    return any(re.match(p, text) for p in _SMALL_TALK_PATTERNS)


def _fallback_classify(user_text: str) -> str:
    if _FALLBACK_WRITE.search(user_text):
        return "memory_write"
    if _FALLBACK_MUSIC.search(user_text):
        return "action_music"
    if _FALLBACK_CALENDAR.search(user_text):
        return "action_calendar"
    if _FALLBACK_CODE.search(user_text):
        return "action_code"
    if user_text.strip().endswith("?"):
        return "memory_recall"
    return "chat"


async def classify_intent_llm(user_text: str, llm_engine) -> str:
    """
    Pass 1: a dedicated LLM call whose only job is picking one category.
    No tools, no memory context - isolating this from the actual work
    (Pass 2) means the model isn't trying to classify and act in the
    same breath, which is what made the regex approach unreliable.
    """
    text = (user_text or "").strip()
    if not text:
        return "chat"

    try:
        raw = await llm_engine.generate_raw(
            system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
            user_prompt=text,
            max_tokens=12,
        )
    except Exception:
        logger.exception("Intent classification LLM call failed; using fallback classifier.")
        return _fallback_classify(text)

    cleaned = re.sub(r"[^a-z_]", "", raw.strip().lower())
    if cleaned in VALID_INTENTS:
        return cleaned

    for intent in VALID_INTENTS:
        if intent in raw.lower():
            return intent

    logger.warning("Intent classifier returned unrecognized output %r; using fallback classifier.", raw)
    return _fallback_classify(text)


def filter_tools_for_intent(all_tools: list, intent: str) -> list:
    keep = _TOOL_NAME_FILTERS.get(intent, lambda name: False)
    return [t for t in all_tools if keep(t.get("function", {}).get("name", ""))]


def wants_vault_context(intent: str) -> bool:
    return intent == "memory_recall"