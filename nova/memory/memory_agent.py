"""
nova/memory/memory_agent.py

The "Self-Updating Loop": an async background job that watches
conversation turns, and — only when the user explicitly asks to
remember/save/note something — asks the brain's LLM to extract that into
structured entries for NOVA's persistent Obsidian-vault memory
(nova/memory/vault_store.py). Runs *outside* the main response path so
it never adds latency to NOVA's spoken reply.
"""
import asyncio
import json
import logging
import re
from typing import Optional

from nova.memory import vault_store
from nova.memory.core_store import core_store

logger = logging.getLogger("nova.memory_agent")


def _compute_antecedent(user_text: str, context_text: str = "") -> str:
    """Return a candidate fact only when the current turn itself contains one.

    The main remember flow should rely on the LLM to infer from recent chat
    context. This helper is only a tiny safety net and should not be used to
    force a save on vague prompts like 'remember that'.
    """
    lower = user_text.strip().lower()

    for pattern in [
        r"\b(?:the\s+fact\s+is\s+that|fact\s+is\s+that|that\s+my\s+name\s+is|my\s+name\s+is|i\s+am|call\s+me)\s+(.+)$",
        r"\b(?:remember|save|note)\s+(?:that|this)\s+(?:my\s+name\s+is|i\s+am|call\s+me|that\s+my\s+name\s+is)\s+(.+)$",
    ]:
        match = re.search(pattern, user_text, re.I)
        if match:
            candidate = match.group(1).strip().rstrip(".?! ")
            if candidate and candidate.lower() not in {"that", "this", "it", "fact"}:
                return candidate

    if re.search(r"\bremember\b.*\b(?:that|this|fact)\b(?:\s+please)?$", lower):
        return ""

    return ""

FACT_EXTRACTION_INSTRUCTION = (
    "Only extract memory when the user explicitly asks to remember, save, or "
    "note something important. The fact does not need to appear in the same "
    "message as the remember trigger. Use the recent chat context to infer the "
    "most important fact just discussed, then save a short distilled summary. "
    "If the user says 'remember that' or 'save this' with no clear fact in the "
    "current message, inspect the recent conversation and infer the fact from the "
    "surrounding turns instead of saving the word 'that'.\n\n"
    "Important extraction rules:\n"
    "1. Treat 'remember that' as a request to save the subject of the recent "
    "conversation, not the literal word 'that'.\n"
    "2. If the relevant fact was discussed earlier in the same conversation, use "
    "that context to recover the fact and summarize it.\n"
    "3. Prefer a short, distilled summary of the important fact(s), not a word-for-word quote.\n"
    "4. Decide the category:\n"
    "   - profile_fields: identity or personal facts about the user (name, active project, etc.)\n"
    "   - preferences: behavioral preferences, habits, priorities, or rules\n"
    "   - knowledge: explanatory facts, concepts, project info, notes, or content worth learning\n"
    "5. If no clear fact is present in the recent context, return {}.\n\n"
    "Return ONLY compact JSON with up to three optional keys:\n"
    '  "profile_fields": flat scalar facts about the user identity, e.g. '
    '{"name": "Luis", "active_project": "NOVA"}\n'
    '  "knowledge": tangible facts/subject notes, e.g. {"box_name": "content to save"}\n'
    '  "preferences": behavioral/procedural preferences, e.g. {"box_name": "content to save"}\n'
    "If nothing is explicitly worth storing, return {}. Never include conversational filler or the word 'that' as a fact; only JSON."
)

REMEMBER_KEYWORDS = re.compile(r"\b(remember|note that|save this|keep in mind|write down)\b", re.I)


class MemoryAgent:
    def __init__(self, llm_engine=None):
        # llm_engine is injected lazily to avoid a circular import at module load time.
        self.llm_engine = llm_engine

    def set_engine(self, llm_engine) -> None:
        self.llm_engine = llm_engine

    async def process_turn(self, user_text: str) -> None:
        """Persist durable facts only when the user explicitly asks to remember.

        The remember trigger does not have to be in the same message as the fact.
        We inspect the recent chat context, which is the source of truth for
        requests like 'remember that' after a prior factual statement.
        """
        if not REMEMBER_KEYWORDS.search(user_text):
            return

        recent_messages = core_store.recent_messages(limit=12)
        combined_context = "\n".join(f"{m['role']}: {m['content']}" for m in recent_messages)
        await self._extract_and_store(user_text, combined_context)

    async def _extract_and_store(self, user_text: str, context_text: str = "") -> None:
        if self.llm_engine is None:
            return
        try:
            prompt_text = user_text
            if context_text:
                prompt_text = f"Conversation context:\n{context_text}\n\nCurrent user turn:\n{user_text}"

            raw = await self.llm_engine.generate_raw(
                system_prompt=FACT_EXTRACTION_INSTRUCTION,
                user_prompt=prompt_text,
                max_tokens=192,
            )
            parsed = {}
            try:
                parsed = json.loads(raw.strip())
            except Exception:
                logger.debug("memory_agent: LLM returned non-JSON or empty extraction: %r", raw)

            stored_anything = False

            if isinstance(parsed, dict):
                profile_fields = parsed.get("profile_fields")
                if isinstance(profile_fields, dict) and profile_fields:
                    vault_store.update_profile_fields(profile_fields)
                    stored_anything = True
                    logger.info("memory_agent: updated profile fields %s", list(profile_fields.keys()))

                knowledge = parsed.get("knowledge")
                if isinstance(knowledge, dict) and knowledge:
                    for box, content in knowledge.items():
                        if content:
                            vault_store.write_knowledge(box, str(content))
                            stored_anything = True
                    logger.info("memory_agent: wrote knowledge boxes %s", list(knowledge.keys()))

                preferences = parsed.get("preferences")
                if isinstance(preferences, dict) and preferences:
                    for box, content in preferences.items():
                        if content:
                            vault_store.write_preference(box, str(content))
                            stored_anything = True
                    logger.info("memory_agent: wrote preference boxes %s", list(preferences.keys()))

            # If nothing was extracted by the deterministic paths nor the LLM,
            # attempt to compute a sensible antecedent for the 'remember' request
            # rather than saving the raw trailing text (e.g. 'that').
            if REMEMBER_KEYWORDS.search(user_text) and not stored_anything:
                # No fallback to 'general_notes' for ambiguous requests. If the
                # model cannot decide whether the remembered thing is a profile,
                # preference, or knowledge fact, we intentionally skip storage
                # instead of saving a raw pronoun or a vague fragment like 'that'.
                logger.info("memory_agent: ignored ambiguous remember request with no clear factual content")
        except (json.JSONDecodeError, Exception) as exc:  # noqa: BLE001 - best effort background task
            logger.debug("memory_agent: skipped turn (%s)", exc)


memory_agent = MemoryAgent()