"""Print prompt component sizes for representative intents."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nova.brain.llm_engine import LLMEngine


SAMPLES = {
    "chat": "How are you?",
    "action_music": "Play some music",
    "action_calendar": "What's on my calendar tomorrow?",
    "memory_write": "Remember that I prefer dark mode",
    "memory_recall": "What do you know about my project?",
}


def main():
    engine = LLMEngine()
    for intent, text in SAMPLES.items():
        messages = engine._build_messages(text, "", [], [], intent)
        system = messages[0]["content"]
        dynamic = messages[-1]["content"]
        estimate = "estimated"
        print(json.dumps({
            "intent": intent, "system_chars": len(system),
            "history_chars": 0, "dynamic_chars": len(dynamic),
            "tools_chars": 0, "tokens": round((len(system) + len(dynamic)) / 3.6),
            "token_mode": estimate,
        }))


if __name__ == "__main__":
    main()
