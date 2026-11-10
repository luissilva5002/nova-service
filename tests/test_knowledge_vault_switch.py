"""The vault switch: NOVA_VAULT_ENABLED = auto | on | off, decided per call."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nova.memory import knowledge_client  # noqa: E402
from nova.memory.knowledge_client import KnowledgeApiError, KnowledgeDisabledError  # noqa: E402


def api_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(payload).encode("utf-8")
    return response


class VaultSwitchTests(unittest.TestCase):
    def setUp(self):
        knowledge_client._disabled_warned = False

    @staticmethod
    def settings(mode: str, key: str):
        return patch.multiple(knowledge_client, VAULT_MODE=mode, AGENT_API_KEY=key)

    def test_auto_without_a_key_reads_empty_and_never_touches_the_network(self):
        with self.settings("auto", ""), patch.object(knowledge_client, "urlopen") as urlopen:
            self.assertIsNone(knowledge_client.read_knowledge("knowledge/user_profile.md"))
            self.assertEqual(knowledge_client.search_knowledge("Smasher nRF52832"), [])
            urlopen.assert_not_called()

    def test_writes_are_refused_with_a_clear_error(self):
        with self.settings("auto", ""), patch.object(knowledge_client, "urlopen") as urlopen:
            with self.assertRaises(KnowledgeDisabledError) as caught:
                knowledge_client.write_knowledge("knowledge/a.md", "text")
            self.assertIn("not enabled", str(caught.exception))
            with self.assertRaises(KnowledgeDisabledError):
                knowledge_client.append_knowledge_entry("knowledge/a.md", "a fact")
            urlopen.assert_not_called()
        # Existing handlers for API failures also cover the disabled case.
        self.assertTrue(issubclass(KnowledgeDisabledError, KnowledgeApiError))

    def test_auto_with_a_key_uses_the_api(self):
        note = {"path": "knowledge/a.md", "content": "body", "frontmatter": {}}
        with self.settings("auto", "test-key"), patch.object(
            knowledge_client, "urlopen", return_value=api_response(note)
        ) as urlopen:
            self.assertEqual(knowledge_client.read_knowledge("knowledge/a.md"), note)
            urlopen.assert_called_once()

    def test_off_disables_the_vault_even_when_a_key_is_configured(self):
        with self.settings("off", "test-key"), patch.object(knowledge_client, "urlopen") as urlopen:
            self.assertIsNone(knowledge_client.read_knowledge("knowledge/a.md"))
            with self.assertRaises(KnowledgeDisabledError):
                knowledge_client.write_knowledge("knowledge/a.md", "text")
            urlopen.assert_not_called()

    def test_on_without_a_key_still_fails_loudly(self):
        with self.settings("on", ""), patch.object(knowledge_client, "urlopen") as urlopen:
            with self.assertRaises(KnowledgeApiError) as caught:
                knowledge_client.read_knowledge("knowledge/a.md")
            self.assertNotIsInstance(caught.exception, KnowledgeDisabledError)
            self.assertIn("AGENT_API_KEY is not configured", str(caught.exception))
            urlopen.assert_not_called()

    def test_the_disabled_warning_is_logged_once(self):
        with self.settings("auto", ""), patch.object(knowledge_client, "logger") as logger:
            knowledge_client.read_knowledge("knowledge/a.md")
            knowledge_client.read_knowledge("knowledge/b.md")
            knowledge_client.search_knowledge("Smasher nRF52832")
            self.assertEqual(logger.warning.call_count, 1)

    def test_path_validation_still_applies_when_disabled(self):
        with self.settings("auto", ""):
            with self.assertRaises(ValueError):
                knowledge_client.read_knowledge("../etc/passwd")


if __name__ == "__main__":
    unittest.main()
