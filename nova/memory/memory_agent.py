"""
nova/memory/memory_agent.py

The "Self-Updating Loop" from the architecture doc: an async background
job that watches conversation turns, asks the brain's LLM to extract any
durable personal facts worth remembering (in dense, compressed form -
see "Dense Fact Compression"), and writes them into the Tier 2 core
store. This runs *outside* the main response path so it never adds
latency to NOVA's spoken reply.
"""
import asyncio
import json
import logging
import re

from nova.memory.core_store import core_store

logger = logging.getLogger("nova.memory_agent")

FACT_EXTRACTION_INSTRUCTION = (
    "Only extract a fact into durable persistent memory when the user explicitly "
    "asks to remember, save, or note something important. Do not auto-store "
    "everyday chat information such as names, dates, calendar events, or casual "
    "conversation. For ordinary chat context, rely on session memory instead. "
    "Return ONLY compact JSON of the explicit remembered facts; if nothing is "
    "explicitly worth storing, return {}. Never include conversational text, only JSON."
)

REMEMBER_KEYWORDS = re.compile(r"\b(remember|note that|save this|keep in mind|write down)\b", re.I)


class MemoryAgent:
    def __init__(self, llm_engine=None):
        # llm_engine is injected lazily to avoid a circular import at module load time.
        self.llm_engine = llm_engine

    def set_engine(self, llm_engine) -> None:
        self.llm_engine = llm_engine

    async def process_turn(self, user_text: str) -> None:
        """Only extract durable facts when the user explicitly asks to remember them."""
        if not REMEMBER_KEYWORDS.search(user_text):
            return
        recent_messages = core_store.recent_messages(limit=8)
        context_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent_messages)
        await self._extract_and_store(user_text, context_text)

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
                max_tokens=128,
            )
            facts = json.loads(raw.strip())
            if isinstance(facts, dict):
                for key, value in facts.items():
                    core_store.set_fact(key, value)
                if facts:
                    logger.info("memory_agent: stored facts %s", facts)

            if REMEMBER_KEYWORDS.search(user_text) and not facts:
                core_store.set_fact("remembered_note", user_text)
                logger.info("memory_agent: stored explicit reminder %s", user_text)
        except (json.JSONDecodeError, Exception) as exc:  # noqa: BLE001 - best effort background task
            logger.debug("memory_agent: skipped turn (%s)", exc)


memory_agent = MemoryAgent()
