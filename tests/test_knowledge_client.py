import json
import unittest
from urllib.parse import parse_qs, urlparse
from unittest.mock import MagicMock, patch

from nova.memory import knowledge_client


class KnowledgeClientTests(unittest.TestCase):
    def test_search_is_scoped_and_intent_gated(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({"hits": [{"path": "knowledge/a.md"}]}).encode()
        with (
            patch.object(knowledge_client, "AGENT_API_KEY", "test-key"),
            patch.object(knowledge_client, "urlopen", return_value=response) as urlopen,
            patch.object(knowledge_client, "read_knowledge", return_value={
                "path": "knowledge/a.md",
                "title": "A",
                "content": "full note",
                "links": [],
            }) as read,
        ):
            hits = knowledge_client.search_knowledge("Smasher nRF52832")
            self.assertEqual(knowledge_client.search_knowledge("how are you"), [])

        self.assertEqual(hits[0]["content"], "full note")
        read.assert_called_once_with("knowledge/a.md")
        request = urlopen.call_args.args[0]
        params = parse_qs(urlparse(request.full_url).query)
        self.assertEqual(params["q"], ["Smasher nRF52832"])
        self.assertEqual(params["limit"], ["100"])
        self.assertEqual(request.get_header("X-api-key"), "test-key")

    def test_read_returns_agent_response_and_write_uses_put(self):
        get_response = MagicMock()
        get_response.__enter__.return_value.read.return_value = json.dumps(
            {"path": "knowledge/projects/a.md", "content": "---\ntitle: A\n---\nbody", "frontmatter": {"title": "A"}}
        ).encode()
        put_response = MagicMock()
        put_response.__enter__.return_value.read.return_value = b'{"ok":true}'
        with (
            patch.object(knowledge_client, "AGENT_API_KEY", "test-key"),
            patch.object(knowledge_client, "urlopen", side_effect=[get_response, put_response]) as urlopen,
        ):
            note = knowledge_client.read_knowledge("knowledge/projects/a.md")
            knowledge_client.write_knowledge("knowledge/projects/a.md", "replacement")

        self.assertEqual(note["frontmatter"], {"title": "A"})
        self.assertEqual(urlopen.call_args_list[0].args[0].method, "GET")
        self.assertEqual(urlopen.call_args_list[1].args[0].method, "PUT")
        self.assertEqual(json.loads(urlopen.call_args_list[1].args[0].data), {"content": "replacement"})

    def test_paths_cannot_escape_knowledge_prefix(self):
        for path in ("preferences/a.md", "knowledge/../preferences/a.md", "/knowledge/a.md", "../ng-sports/a.md"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                knowledge_client.read_knowledge(path)

    def test_search_discards_hits_outside_knowledge(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"hits": [
                {"path": "ng-sports/smasher/01 Patent & Legal.md"},
                {"path": "knowledge/a.md"},
                {"path": "preferences/b.md"},
            ]}
        ).encode()
        with (
            patch.object(knowledge_client, "AGENT_API_KEY", "test-key"),
            patch.object(knowledge_client, "urlopen", return_value=response) as urlopen,
            patch.object(knowledge_client, "read_knowledge", side_effect=lambda path: {
                "path": path, "title": path, "content": "note", "links": []
            }),
        ):
            hits = knowledge_client.search_knowledge("smasher")
        self.assertEqual(
            [hit["path"] for hit in hits],
            ["ng-sports/smasher/01 Patent & Legal.md", "knowledge/a.md"],
        )
        request = urlopen.call_args.args[0]
        params = parse_qs(urlparse(request.full_url).query)
        self.assertEqual(params["q"], ["smasher"])
        self.assertEqual(params["limit"], ["100"])

    def test_search_fans_out_from_short_index_with_total_content_budget(self):
        index_path = "ng-sports/smasher/00 Smasher.md"
        link_paths = [
            "ng-sports/smasher/01 Patent & Legal.md",
            "ng-sports/smasher/02 PCB.md",
            "ng-sports/smasher/03 Enclosure.md",
            "ng-sports/smasher/04 Battery & Power.md",
        ]
        index_content = "# Smasher\n" + ("Overview. " * 8)
        assert len(index_content) < knowledge_client._SHORT_INDEX_NOTE_CHAR_LIMIT
        notes = {
            index_path: {
                "path": index_path,
                "title": "00 Smasher",
                "content": index_content,
                "links": [path.rsplit("/", 1)[1].removesuffix(".md") for path in link_paths],
            },
            link_paths[0]: {
                "path": link_paths[0],
                "title": "01 Patent & Legal",
                "content": "Patent filing details. " * 30,
                "links": [],
            },
            link_paths[1]: {
                "path": link_paths[1],
                "title": "02 PCB",
                "content": "PCB details. " * 30,
                "links": [],
            },
            link_paths[2]: {
                "path": link_paths[2],
                "title": "03 Enclosure",
                "content": "Enclosure details. " * 30,
                "links": [],
            },
            link_paths[3]: {
                "path": link_paths[3],
                "title": "04 Battery & Power",
                "content": "Battery details. " * 30,
                "links": [],
            },
        }
        with (
            patch.object(knowledge_client, "_request", return_value={
                "hits": [{"path": index_path, "title": "00 Smasher", "snippet": "Smasher overview"}]
            }),
            patch.object(knowledge_client, "read_knowledge", side_effect=lambda path: notes.get(path)),
        ):
            results = knowledge_client.search_knowledge("Smasher")

        self.assertEqual([result["path"] for result in results], [index_path, *link_paths[:3]])
        self.assertIn("Patent filing details.", results[1]["content"])
        self.assertIn("PCB details.", results[2]["content"])
        self.assertLessEqual(
            sum(len(result["content"]) for result in results if "content" in result),
            knowledge_client._KNOWLEDGE_CONTENT_CHAR_BUDGET,
        )

    def test_search_full_content_count_is_configurable(self):
        hits = [
            {"path": f"knowledge/{index}.md", "title": str(index)}
            for index in range(4)
        ]
        with (
            patch.object(knowledge_client, "_request", return_value={"hits": hits}),
            patch.object(knowledge_client, "read_knowledge", side_effect=lambda path: {
                "path": path,
                "title": path,
                "content": "# Note\n" + ("Long body. " * 150),
                "links": [],
            }) as read,
        ):
            results = knowledge_client.search_knowledge("note", max_results=4, full_content_results=2)

        self.assertEqual(read.call_count, 2)
        self.assertTrue(all("content" in result for result in results[:2]))
        self.assertTrue(all("content" not in result for result in results[2:]))

    def test_append_helper_preserves_existing_note_and_uses_put(self):
        existing = {
            "content": "---\ntitle: Smasher\n---\n# Smasher\n\n## Updates\n- Earlier fact\n",
            "frontmatter": {"title": "Smasher"},
        }
        with (
            patch.object(knowledge_client, "read_knowledge", return_value=existing),
            patch.object(knowledge_client, "write_knowledge") as write,
        ):
            knowledge_client.append_knowledge_entry("knowledge/smasher.md", "New fact")

        written = write.call_args.args[1]
        self.assertIn("- Earlier fact", written)
        self.assertIn("New fact", written)

    def test_recall_tool_requires_only_a_search_query(self):
        from nova.skills.remember.handler import skill
        from nova.skills.remember.tools import TOOLS

        schema = next(tool for tool in TOOLS if tool["name"] == "recall_note")["parameters"]
        self.assertEqual(schema["required"], ["query"])
        self.assertEqual(set(schema["properties"]), {"query"})
        with patch("nova.skills.remember.handler.search_knowledge", return_value=[
            {"path": "ng-sports/smasher/01 Patent & Legal.md", "title": "01 Patent & Legal", "content": "Patent filing status."},
            {"path": "ng-sports/smasher/02 PCB.md", "title": "02 PCB", "content": "Board details."},
        ]) as search:
            result = skill.execute("recall_note", {"query": "patent status"})
        search.assert_called_once_with("patent status")
        self.assertIn("01 Patent & Legal", result)
        self.assertIn("Patent filing status.", result)
        self.assertIn("## 02 PCB", result)


if __name__ == "__main__":
    unittest.main()
