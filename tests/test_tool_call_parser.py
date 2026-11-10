"""Tests for recovering tool calls from malformed model text."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nova.brain.tool_call_parser import extract_tool_call  # noqa: E402


class ToolCallParserTests(unittest.TestCase):
    def test_observed_unbalanced_calendar_call(self):
        # Seen from Qwen2.5-3B: three opening braces, two closing.
        text = (
            '<tool_call>\n{{"name": "gc_list_today", "arguments": '
            '{"calendar_id": "primary"}}\n</tool_call>'
        )
        self.assertEqual(
            extract_tool_call(text),
            {"name": "gc_list_today", "arguments": {"calendar_id": "primary"}},
        )

    def test_valid_json_is_unchanged(self):
        text = '<tool_call>\n{"name": "ytm_play", "arguments": {"query": "daft punk"}}\n</tool_call>'
        self.assertEqual(
            extract_tool_call(text),
            {"name": "ytm_play", "arguments": {"query": "daft punk"}},
        )

    def test_balanced_doubled_braces(self):
        text = '<tool_call>{{"name": "x", "arguments": {"a": 1}}}</tool_call>'
        self.assertEqual(extract_tool_call(text), {"name": "x", "arguments": {"a": 1}})

    def test_triple_leading_braces(self):
        text = '<tool_call>{{{"name": "ytm_pause", "arguments": {}}}}</tool_call>'
        self.assertEqual(extract_tool_call(text), {"name": "ytm_pause", "arguments": {}})

    def test_extra_closing_brace(self):
        text = '<tool_call>{"name": "ytm_pause", "arguments": {}}}</tool_call>'
        self.assertEqual(extract_tool_call(text), {"name": "ytm_pause", "arguments": {}})

    def test_missing_closing_tag_and_braces(self):
        text = '<tool_call>\n{"name": "ytm_pause", "arguments": {}'
        self.assertEqual(extract_tool_call(text), {"name": "ytm_pause", "arguments": {}})

    def test_trailing_comma(self):
        text = '<tool_call>{"name": "ytm_skip", "arguments": {},}</tool_call>'
        self.assertEqual(extract_tool_call(text), {"name": "ytm_skip", "arguments": {}})

    def test_code_fence_inside_tags(self):
        text = '<tool_call>```json\n{"name": "ytm_pause", "arguments": {}}\n```</tool_call>'
        self.assertEqual(extract_tool_call(text), {"name": "ytm_pause", "arguments": {}})

    def test_arguments_given_as_json_string(self):
        text = '<tool_call>{"name": "ytm_play", "arguments": "{\\"query\\": \\"abba\\"}"}</tool_call>'
        self.assertEqual(extract_tool_call(text), {"name": "ytm_play", "arguments": {"query": "abba"}})

    def test_parameters_alias(self):
        text = '<tool_call>{"name": "ytm_set_volume", "parameters": {"level": 30}}</tool_call>'
        self.assertEqual(extract_tool_call(text), {"name": "ytm_set_volume", "arguments": {"level": 30}})

    def test_braces_inside_string_values(self):
        text = '<tool_call>{{"name": "gc_find", "arguments": {"query": "a}b{c"}}</tool_call>'
        self.assertEqual(extract_tool_call(text), {"name": "gc_find", "arguments": {"query": "a}b{c"}})

    def test_nested_array_arguments(self):
        text = (
            '<tool_call>{{"name": "update_user_knowledge", "arguments": '
            '{"box": "b", "content": "c", "tags": ["a", "b"]}}</tool_call>'
        )
        self.assertEqual(
            extract_tool_call(text),
            {
                "name": "update_user_knowledge",
                "arguments": {"box": "b", "content": "c", "tags": ["a", "b"]},
            },
        )

    def test_unknown_tool_is_rejected_when_names_are_given(self):
        text = '<tool_call>{"name": "ytm_play", "arguments": {}}</tool_call>'
        self.assertIsNone(extract_tool_call(text, {"gc_list_today"}))
        self.assertIsNotNone(extract_tool_call(text, {"ytm_play"}))

    def test_empty_allowed_set_rejects_everything(self):
        text = '<tool_call>{"name": "ytm_play", "arguments": {}}</tool_call>'
        self.assertIsNone(extract_tool_call(text, set()))

    def test_garbage_and_empty_return_none(self):
        for text in ("", "   ", "<tool_call>hello there</tool_call>", "no call here", None):
            self.assertIsNone(extract_tool_call(text), text)

    def test_missing_name_returns_none(self):
        self.assertIsNone(extract_tool_call('<tool_call>{"arguments": {"a": 1}}</tool_call>'))


if __name__ == "__main__":
    unittest.main()
