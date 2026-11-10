"""Tests for surrogate sanitizing and the locale-independent date context."""
from __future__ import annotations

import datetime
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_DATA_DIR = tempfile.TemporaryDirectory(prefix="nova-text-dates-", ignore_cleanup_errors=True)
os.environ.setdefault("NOVA_DATA_DIR", _DATA_DIR.name)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import types  # noqa: E402

from nova.text_utils import has_surrogates, sanitize_text  # noqa: E402
from nova.brain.context_manager import LlamaBackend  # noqa: E402
from nova.brain import llm_engine as llm_engine_module  # noqa: E402
from nova.brain.llm_engine import LLMEngine  # noqa: E402


class _SurrogateZone(datetime.tzinfo):
    """A timezone whose name contains a lone surrogate (Portuguese Windows %Z)."""

    def utcoffset(self, dt):
        return datetime.timedelta(0)

    def dst(self, dt):
        return datetime.timedelta(0)

    def tzname(self, dt):
        return "Hora padr\udce3o de GMT"


class SanitizeTextTests(unittest.TestCase):
    def test_replaces_lone_surrogates_and_result_encodes(self):
        dirty = "Hora padr\udce3o de GMT"
        self.assertTrue(has_surrogates(dirty))
        with self.assertRaises(UnicodeEncodeError):
            dirty.encode("utf-8")
        clean = sanitize_text(dirty, "test")
        self.assertFalse(has_surrogates(clean))
        self.assertEqual(clean, "Hora padr\ufffdo de GMT")
        clean.encode("utf-8")

    def test_clean_text_and_non_strings_pass_through(self):
        self.assertEqual(sanitize_text("Olá, não é problema"), "Olá, não é problema")
        self.assertIsNone(sanitize_text(None))
        self.assertEqual(sanitize_text(5), 5)

    def test_warns_once_per_source(self):
        with patch("nova.text_utils.logger") as logger:
            sanitize_text("a\udce3", "unique-source-1")
            sanitize_text("b\udce3", "unique-source-1")
            self.assertEqual(logger.warning.call_count, 1)


class TokenizerBoundaryTests(unittest.TestCase):
    def test_backend_encode_replaces_surrogates_instead_of_raising(self):
        seen: list[bytes] = []

        def tokenize(data, add_bos, special):
            seen.append(data)
            return [1, 2, 3]

        backend = object.__new__(LlamaBackend)  # skip the real model-backed __init__
        backend._llama = types.SimpleNamespace(tokenize=tokenize)
        self.assertEqual(backend.encode("Hora padr\udce3o"), [1, 2, 3])
        self.assertEqual(seen[0], "Hora padr\ufffdo".encode("utf-8"))

    def test_clean_text_is_passed_through_unchanged(self):
        seen: list[bytes] = []
        backend = object.__new__(LlamaBackend)
        backend._llama = types.SimpleNamespace(
            tokenize=lambda data, add_bos, special: seen.append(data) or [7]
        )
        backend.encode("<|im_start|>user\nOlá, não<|im_end|>\n")
        self.assertEqual(seen[0], "<|im_start|>user\nOlá, não<|im_end|>\n".encode("utf-8"))


class DateContextTests(unittest.TestCase):
    def test_header_is_ascii_and_ignores_timezone_name(self):
        now = datetime.datetime(2026, 11, 8, 12, 30, tzinfo=_SurrogateZone())
        text = LLMEngine._format_date_context(now)
        self.assertEqual(text, "Current date: Sunday, 2026-11-08 (UTC+00:00)")
        text.encode("ascii")

    def test_utc_offset_labels(self):
        cases = {
            datetime.timedelta(hours=1): "UTC+01:00",
            datetime.timedelta(hours=-3, minutes=-30): "UTC-03:30",
            datetime.timedelta(hours=5, minutes=45): "UTC+05:45",
            datetime.timedelta(0): "UTC+00:00",
        }
        for offset, expected in cases.items():
            moment = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone(offset))
            self.assertEqual(LLMEngine._utc_offset_label(moment), expected)

    def test_lookup_table_has_fourteen_days_across_a_month_boundary(self):
        now = datetime.datetime(2026, 11, 28, 9, 0, tzinfo=datetime.timezone.utc)
        lines = LLMEngine._format_date_context(now, include_lookup=True).splitlines()
        self.assertEqual(len(lines), 2 + 14)
        self.assertIn("Today: 2026-11-28", lines[2])
        self.assertIn("Tomorrow: 2026-11-29", lines[3])
        self.assertIn("Monday: 2026-11-30", lines[4])
        self.assertIn("Tuesday: 2026-12-01", lines[5])
        self.assertIn("2026-12-11", lines[-1])

    def test_current_date_context_is_always_encodable(self):
        weird = datetime.datetime(2026, 11, 8, 12, tzinfo=_SurrogateZone())
        with patch.object(llm_engine_module.datetime, "datetime") as fake_datetime:
            fake_datetime.now.return_value.astimezone.return_value = weird
            text = LLMEngine._current_date_context(True)
        text.encode("utf-8")


class RecentExchangeTests(unittest.TestCase):
    HISTORY = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second   question\nwith breaks"},
        {"role": "assistant", "content": "second answer " + "x" * 400},
    ]

    def test_last_exchange_only_and_truncated(self):
        text = LLMEngine._recent_exchange_text(self.HISTORY, 1)
        lines = text.splitlines()
        self.assertEqual(lines[0], "User: second question with breaks")
        self.assertTrue(lines[1].startswith("Assistant: second answer"))
        self.assertTrue(lines[1].endswith("..."))
        self.assertLessEqual(len(lines[1]), len("Assistant: ") + 243)
        self.assertNotIn("first", text)

    def test_zero_turns_and_empty_history(self):
        self.assertEqual(LLMEngine._recent_exchange_text(self.HISTORY, 0), "")
        self.assertEqual(LLMEngine._recent_exchange_text([], 1), "")
        self.assertEqual(LLMEngine._recent_exchange_text(None, 1), "")

    def test_surrogates_are_removed(self):
        history = [{"role": "user", "content": "Hora padr\udce3o"}]
        LLMEngine._recent_exchange_text(history, 1).encode("utf-8")


if __name__ == "__main__":
    unittest.main()
