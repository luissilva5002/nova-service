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
    "You are a fact-extraction agent for NOVA's persistent memory vault. "
    "Your job is to read the recent chat context and derive the important facts, "
    "not the raw wording. Use a background analysis pass, not regex-based capture.\n\n"
    "Rules:\n"
    "1. Only save facts when the user clearly indicates a desire to remember or save something important.\n"
    "2. Do not auto-save ordinary casual statements. A stored fact must be tied to an explicit remember/save/note instruction in the current turn or immediately adjacent context.\n"
    "3. The fact does not need to be in the same message as the remember trigger; use the surrounding chat to infer it.\n"
    "4. Do not save vague filler like 'that', 'this', 'it', or 'remember that fact' without a meaningful fact.\n"
    "5. Save each fact as a small, distilled statement. Merge multiple facts into separate entries when appropriate.\n"
    "6. Choose a target_box that matches the fact type:\n"
    "   - user_profile for identity/profile facts like name, role, school, active project\n"
    "   - communication_preferences, study_preferences, or other short preference boxes for preferences\n"
    "   - general_notes or a domain-specific knowledge box for broader knowledge notes\n"
    "7. Prefer concise, human-readable facts such as 'Luis is a computer engineering student' or 'Luis studies at FEUP'.\n"
    "8. Include a confidence score: 'high' for explicit facts, 'medium' for likely facts, 'low' for tentative/weak inferences.\n"
    "9. Include a short source_turn_hint that names the relevant turn or the topic, e.g. 'user: I am a computer engineering student at FEUP'.\n"
    "10. If no clear fact is present in the recent context, or the user did not explicitly request memory, return an empty list [].\n\n"
    "Return ONLY compact JSON structured exactly like this:\n"
    '[{"target_box": "user_profile", "fact": "Luis is a computer engineering student.", "confidence": "high", "source_turn_hint": "user: I am a computer engineering student at FEUP"}, {"target_box": "study_preferences", "fact": "Luis prefers to learn by doing practical exercises.", "confidence": "medium", "source_turn_hint": "user: I like hands-on learning"}]\n\n'
    "Do not wrap the JSON in markdown fences. Do not include commentary. Only a JSON array of objects with 'target_box', 'fact', 'confidence', and 'source_turn_hint'."
)

REMEMBER_KEYWORDS = re.compile(
    r"\b(?:remember|note that|save this|save to memory|store this in memory|create a memory|make a memory|keep in mind|write down)\b",
    re.I,
)
PERSONAL_FACT_PATTERNS = [
    r"\bmy\s+name\s+is\b",
    r"\bi\s+am\s+(?:a\s+)?[A-Za-z][A-Za-z'\- ]{2,80}\b",
    r"\bi\s+study\s+at\b",
    r"\bi\s+work\s+as\b",
    r"\bmy\s+favorite\b",
    r"\bi\s+prefer\b",
    r"\bi\s+like\b",
    r"\bi'm\s+(?:a\s+)?[A-Za-z][A-Za-z'\- ]{2,80}\b",
    r"\bcall\s+me\b",
]


def _looks_like_personal_fact(user_text: str) -> bool:
    lower = user_text.lower()
    if not lower:
        return False
    if REMEMBER_KEYWORDS.search(user_text):
        return True

    if re.search(r"\b(?:create|make|save|store)\s+(?:a\s+)?memory\b", user_text, re.I):
        return True

    return False


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
        if not (_looks_like_personal_fact(user_text) or REMEMBER_KEYWORDS.search(user_text)):
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
            parsed = []
            try:
                parsed = json.loads(raw.strip())
            except Exception:
                logger.debug("memory_agent: LLM returned non-JSON or empty extraction: %r", raw)

            stored_anything = False
            extracted = []

            if isinstance(parsed, dict):
                if isinstance(parsed.get("facts"), list):
                    extracted = parsed["facts"]
                elif isinstance(parsed.get("profile_fields"), dict):
                    extracted = [{"target_box": "user_profile", "fact": v, "confidence": "high", "source_turn_hint": "profile_fields"} for _, v in parsed["profile_fields"].items() if v]
                elif isinstance(parsed.get("knowledge"), dict):
                    extracted = [{"target_box": k, "fact": v, "confidence": "medium", "source_turn_hint": "knowledge"} for k, v in parsed["knowledge"].items() if v]
                elif isinstance(parsed.get("preferences"), dict):
                    extracted = [{"target_box": k, "fact": v, "confidence": "medium", "source_turn_hint": "preferences"} for k, v in parsed["preferences"].items() if v]
            elif isinstance(parsed, list):
                extracted = parsed

            for item in extracted:
                if not isinstance(item, dict):
                    continue
                target_box = str(item.get("target_box") or item.get("box") or "").strip()
                fact = str(item.get("fact") or "").strip().rstrip(".?! ")
                confidence = str(item.get("confidence") or "medium").strip().lower()
                source_turn_hint = str(item.get("source_turn_hint") or "").strip()
                if not target_box or not fact:
                    continue
                if confidence == "low":
                    logger.info("memory_agent: skipped low-confidence fact target_box=%s fact=%s source=%s", target_box, fact, source_turn_hint)
                    continue

                if target_box in {"user_profile", "profile"}:
                    if re.search(r"\bname\b", fact, re.I):
                        m = re.search(r"(?:my\s+name\s+is|i\s+am|call\s+me)\s+(.+)$", fact, re.I)
                        if m:
                            vault_store.update_profile_fields({"name": m.group(1).strip()})
                        else:
                            vault_store.write_knowledge(target_box, fact)
                    else:
                        vault_store.write_knowledge(target_box, fact)
                    stored_anything = True
                    logger.info("memory_agent: stored structured profile fact target_box=%s fact=%s confidence=%s source=%s", target_box, fact, confidence, source_turn_hint)
                    continue

                if target_box in {"name"}:
                    name = fact
                    if re.search(r"\bname\b", fact, re.I):
                        m = re.search(r"(?:my\s+name\s+is|i\s+am|call\s+me)\s+(.+)$", fact, re.I)
                        if m:
                            name = m.group(1).strip()
                    vault_store.update_profile_fields({"name": name})
                    stored_anything = True
                    logger.info("memory_agent: stored structured name fact target_box=%s fact=%s confidence=%s source=%s", target_box, fact, confidence, source_turn_hint)
                    continue

                if "preference" in target_box.lower() or "preferences" in target_box.lower():
                    vault_store.write_preference(target_box, fact)
                else:
                    vault_store.write_knowledge(target_box, fact)
                stored_anything = True
                logger.info("memory_agent: stored structured fact target_box=%s fact=%s confidence=%s source=%s", target_box, fact, confidence, source_turn_hint)

            if REMEMBER_KEYWORDS.search(user_text) and not stored_anything:
                logger.info("memory_agent: ignored ambiguous remember request with no clear factual content")
        except (json.JSONDecodeError, Exception) as exc:  # noqa: BLE001 - best effort background task
            logger.debug("memory_agent: skipped turn (%s)", exc)


memory_agent = MemoryAgent()