"""
nova/brain/intent_router.py

Intent detection pipeline, in priority order:
  1. Trained classifier (nova/brain/intent_classifier.py) - MiniLM
     embeddings -> LogisticRegression, cheap and CPU-only. Used whenever
     it's available AND its confidence clears INTENT_CONFIDENCE_THRESHOLD.
  2. LLM fallback (classify_intent_llm, this module) - a dedicated Qwen3
     call, used ONLY when the trained classifier is unavailable or
     under-confident. This keeps the (slower) generative model out of
     the hot path for the vast majority of turns.
  3. Regex override (apply_memory_reference_override) - always runs
     last, regardless of which path produced the intent. It's a narrow,
     free safety net that only upgrades 'chat' to 'memory_recall' when
     the user explicitly references NOVA's own memory/files/notes -
     catching phrasings neither the classifier nor the LLM caught.

Once an intent is resolved here, everything downstream (tool-schema
filtering, vault-context gating, and the actual Qwen3 response
generation) is unchanged.
"""
import logging
import re

from nova.brain.intent_classifier import intent_classifier
from nova.config import INTENT_CONFIDENCE_THRESHOLD

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

# Used only when BOTH the trained classifier and the LLM fallback fail
# outright (model not loaded, empty/garbled output, exception).
_FALLBACK_MUSIC = re.compile(r"\b(play|pause|resume|skip|song|track|music|playlist)\b", re.I)
_FALLBACK_CALENDAR = re.compile(r"\b(calendar|event|meeting|appointment|schedule|remind me|reminder|agenda)\b", re.I)
_FALLBACK_CODE = re.compile(r"\b(codebase|source code|read file|search code|repository|repo)\b", re.I)
_FALLBACK_WRITE = re.compile(r"\b(remember|note that|save this|save to memory|store this in memory)\b", re.I)

_SMALL_TALK_PATTERNS = [
    r"^\s*(hey|hi|hello)(\s+nova)?[,.!]?\s*$",
    r"^\s*(how are you|what's up|whats up|how's it going)\s*\??\s*$",
]

_MEMORY_REFERENCE_OVERRIDE = re.compile(
    r"\b(your memory|in memory|check your (?:memory|notes|files)|"
    r"access (?:it|that|this) (?:on|in|from) your memory|"
    r"look (?:into|at) (?:those|the|your) files|"
    r"(?:the )?record(?:s)? (?:that )?you have|"
    r"already written in the files|"
    r"analyze (?:them|that|this) (?:on|in) your memory|"
    r"what you (?:have|saved|stored) (?:on|about))\b",
    re.I,
)


def is_small_talk(user_text: str) -> bool:
    """Small helper used elsewhere (e.g. main.py's _select_relevant_facts) -
    NOT part of primary intent classification anymore."""
    text = user_text.strip().lower()
    if len(text.split()) > 6:
        return False
    return any(re.match(p, text) for p in _SMALL_TALK_PATTERNS)


def _regex_fallback_classify(user_text: str) -> str:
    """Last-resort classifier, used only if BOTH the trained classifier
    and the LLM fallback are unavailable/fail."""
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
    Dedicated Qwen3 classification call - used ONLY as a fallback when
    the trained classifier is unavailable or under-confident, not as
    the primary mechanism anymore.
    """
    text = (user_text or "").strip()
    if not text:
        return "chat"

    try:
        raw = await llm_engine.generate_raw(
            system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
            user_prompt=text,
            max_tokens=16,
        )
    except Exception:
        logger.exception("LLM intent classification call failed; using regex fallback.")
        return _regex_fallback_classify(text)

    logger.info("LLM classifier raw output: %r", raw)
    cleaned = re.sub(r"[^a-z_]", "", raw.strip().lower())
    if cleaned in VALID_INTENTS:
        return cleaned

    for intent in VALID_INTENTS:
        if intent in raw.lower():
            return intent

    logger.warning("LLM classifier returned unrecognized output %r; using regex fallback.", raw)
    return _regex_fallback_classify(text)


def apply_memory_reference_override(user_text: str, classified_intent: str) -> str:
    """
    Narrow safety net: only upgrades an already-classified 'chat' result
    to 'memory_recall' when the user explicitly tells NOVA to check its
    own memory/files/notes. Never overrides action/write intents from
    either classification path. Always runs, regardless of which
    mechanism (trained classifier or LLM) produced the intent.
    """
    if classified_intent != "chat":
        return classified_intent
    if _MEMORY_REFERENCE_OVERRIDE.search(user_text):
        logger.info("Memory-reference override: upgrading chat -> memory_recall for %r", user_text)
        return "memory_recall"
    return classified_intent


async def classify_intent(user_text: str, llm_engine) -> dict:
    """
    The single entry point main.py should call. Returns a dict with the
    resolved intent plus diagnostic metadata for logging:
      {"intent": str, "source": "classifier"|"llm"|"regex_fallback",
       "confidence": float|None, "all_scores": dict|None}

    Resolution order:
      1. Trained classifier, if available and confidence >= threshold.
      2. Qwen3 LLM classification, if the trained classifier declined.
      3. Regex-only fallback, if the LLM call itself also fails.
      4. Memory-reference override applied on top of whichever won.
    """
    text = (user_text or "").strip()
    if not text:
        return {"intent": "chat", "source": "empty", "confidence": None, "all_scores": None}

    result = intent_classifier.classify(text)
    if result is not None:
        logger.info(
            "Trained classifier: intent=%s confidence=%.3f all_scores=%s",
            result["intent"], result["confidence"], result["all_scores"],
        )
        if result["confidence"] >= INTENT_CONFIDENCE_THRESHOLD and result["intent"] in VALID_INTENTS:
            intent = apply_memory_reference_override(text, result["intent"])
            return {
                "intent": intent,
                "source": "classifier",
                "confidence": result["confidence"],
                "all_scores": result["all_scores"],
            }
        logger.info(
            "Trained classifier confidence %.3f below threshold %.2f - falling back to LLM classification.",
            result["confidence"], INTENT_CONFIDENCE_THRESHOLD,
        )

    llm_intent = await classify_intent_llm(text, llm_engine)
    intent = apply_memory_reference_override(text, llm_intent)
    return {
        "intent": intent,
        "source": "llm" if result is None or result["confidence"] < INTENT_CONFIDENCE_THRESHOLD else "classifier",
        "confidence": result["confidence"] if result else None,
        "all_scores": result["all_scores"] if result else None,
    }


def filter_tools_for_intent(all_tools: list, intent: str) -> list:
    keep = _TOOL_NAME_FILTERS.get(intent, lambda name: False)
    return [t for t in all_tools if keep(t.get("function", {}).get("name", ""))]


def wants_vault_context(intent: str) -> bool:
    return intent == "memory_recall"