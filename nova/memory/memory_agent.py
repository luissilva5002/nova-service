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

from nova.memory.core_store import core_store

logger = logging.getLogger("nova.memory_agent")

FACT_EXTRACTION_INSTRUCTION = (
    "You extract durable personal facts from a single conversation turn. "
    "Return ONLY a compact JSON object of key/value pairs worth remembering "
    "long-term (e.g. active project, stack preferences, standing rules). "
    "If nothing is worth storing, return {}. Never include conversational "
    "text, only JSON."
)


class MemoryAgent:
    def __init__(self, llm_engine=None):
        # llm_engine is injected lazily to avoid a circular import at module load time.
        self.llm_engine = llm_engine

    def set_engine(self, llm_engine) -> None:
        self.llm_engine = llm_engine

    async def process_turn(self, user_text: str) -> None:
        """Fire-and-forget: schedule fact extraction without blocking the reply path."""
        asyncio.create_task(self._extract_and_store(user_text))

    async def _extract_and_store(self, user_text: str) -> None:
        if self.llm_engine is None:
            return
        try:
            raw = await self.llm_engine.generate_raw(
                system_prompt=FACT_EXTRACTION_INSTRUCTION,
                user_prompt=user_text,
                max_tokens=128,
            )
            facts = json.loads(raw.strip())
            if isinstance(facts, dict):
                for key, value in facts.items():
                    core_store.set_fact(key, value)
                if facts:
                    logger.info("memory_agent: stored facts %s", facts)
        except (json.JSONDecodeError, Exception) as exc:  # noqa: BLE001 - best effort background task
            logger.debug("memory_agent: skipped turn (%s)", exc)


memory_agent = MemoryAgent()
