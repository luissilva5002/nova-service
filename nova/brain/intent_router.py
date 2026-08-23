"""
nova/brain/intent_router.py

Classifies each user turn into exactly one intent BEFORE any memory
lookup or tool-schema exposure happens, so a command like "play a song"
can never accidentally surface unrelated vault notes or be offered
memory-write/recall tools it has no business seeing.

Intents:
  action_music     -> only ytm_* tools exposed, no vault context at all
  action_calendar  -> only gc_* tools exposed, no vault context at all
  action_code      -> only search_codebase/read_file exposed, no vault context
  memory_write     -> only update_user_preference/update_user_knowledge exposed
  memory_recall    -> only recall_note exposed, vault search scoped to the query
  chat             -> NO tools exposed at all, no vault knowledge search
                       (small talk, opinions, general conversation)

This is deliberately keyword/regex based rather than an extra LLM call -
it needs to be instant and needs to be reliable at gating what the model
even sees, not itself be a source of ambiguity.
"""
import re

_SMALL_TALK_PATTERNS = [
    r"^\s*(hey|hi|hello)\b",
    r"\b(?:how are you|how\s+you\s+doing|what's up|whats up|how's it going|how is it going)\b",
    r"\b(?:good morning|good afternoon|good evening|have a good day|nice to meet you)\b",
    r"\b(?:thanks|thank you|thanks a lot|bye|goodbye)\b",
]

_MEMORY_WRITE_PATTERN = re.compile(
    r"\b(remember|note that|save this|save to memory|store this in memory|"
    r"create a memory|make a memory|keep in mind|write down)\b",
    re.I,
)

_MEMORY_RECALL_PATTERN = re.compile(
    r"\b(what do you know about|do you remember|what did i tell you about|"
    r"recall|look up|what's my|what is my|who am i|tell me about my|"
    r"where do i|which\s+(?:mcu|microcontroller|chip|processor|imu|sensor|battery|antenna)|"
    r"what\s+(?:mcu|microcontroller|chip|processor|imu|sensor|battery|antenna))\b",
    re.I,
)

_MUSIC_PATTERN = re.compile(
    r"\b(play|pause|resume|skip|next track|previous track|volume|song|track|music|playlist)\b",
    re.I,
)

_CALENDAR_PATTERN = re.compile(
    r"\b(calendar|event|meeting|appointment|schedule|remind me|reminder|agenda)\b",
    re.I,
)

_CODE_PATTERN = re.compile(
    r"\b(codebase|source code|read file|search code|repository|repo|"
    r"project file|flutter project workspace)\b",
    re.I,
)

_TOOL_NAME_FILTERS = {
    "action_music": lambda name: name.startswith("ytm_"),
    "action_calendar": lambda name: name.startswith("gc_"),
    "action_code": lambda name: name in ("search_codebase", "read_file"),
    "memory_write": lambda name: name in ("update_user_preference", "update_user_knowledge"),
    "memory_recall": lambda name: name == "recall_note",
    "chat": lambda name: False,  # chat gets zero tools - forces a plain reply
}


def is_small_talk(user_text: str) -> bool:
    lower = user_text.strip().lower()
    return any(re.search(p, lower) for p in _SMALL_TALK_PATTERNS)


def classify_intent(user_text: str) -> str:
    text = user_text.strip()
    if not text:
        return "chat"
    if is_small_talk(text):
        return "chat"

    # Memory write is checked first: "remember to play music later" should
    # still be treated as a note to save, not an action.
    if _MEMORY_WRITE_PATTERN.search(text):
        return "memory_write"

    if _MUSIC_PATTERN.search(text):
        return "action_music"
    if _CALENDAR_PATTERN.search(text):
        return "action_calendar"
    if _CODE_PATTERN.search(text):
        return "action_code"

    if _MEMORY_RECALL_PATTERN.search(text) or text.rstrip().endswith("?"):
        return "memory_recall"

    return "chat"


def filter_tools_for_intent(all_tools: list, intent: str) -> list:
    """all_tools is tool_router.tool_schemas - a list of OpenAI-style
    {"type": "function", "function": {"name": ..., ...}} dicts."""
    keep = _TOOL_NAME_FILTERS.get(intent, lambda name: False)
    return [t for t in all_tools if keep(t.get("function", {}).get("name", ""))]


def wants_vault_context(intent: str) -> bool:
    """Only memory_recall turns should ever trigger a knowledge-vault
    search. Action turns and chat turns never see vault knowledge -
    action turns because it's irrelevant noise, chat turns because
    small talk shouldn't drag in unrelated notes."""
    return intent == "memory_recall"