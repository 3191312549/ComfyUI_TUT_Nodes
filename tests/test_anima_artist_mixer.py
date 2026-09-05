import json
from pathlib import Path
import sqlite3
import unittest

from ComfyUI_TUT_Nodes.categories import TOOLS_TEXT
from ComfyUI_TUT_Nodes.core.anima_artists import (
    ARTIST_DATABASE_PATH,
    mix_anima_artist_prompt,
    normalize_prompt_punctuation,
    parse_artist_data,
    parse_artist_segment,
    search_anima_artists,
    split_top_level_commas,
)
from ComfyUI_TUT_Nodes.nodes.tools.text import TUT_AnimaArtistPromptMixer


def artist_data(*artists):
    return json.dumps(
        {"version": 1, "artists": [{"name": name, "weight": weight} for name, weight in artists]},
        ensure_ascii=False,
    )


class TUTAnimaArtistPromptMixerTests(unittest.TestCase):
    def test_public_interface(self):
        required = TUT_AnimaArtistPromptMixer.INPUT_TYPES()["required"]
        self.assertTrue(required["prompt"][1]["forceInput"])
        self.assertEqual(required["artist_data"][1]["default"], '{"version":1,"artists":[]}')
        self.assertEqual(TUT_AnimaArtistPromptMixer.RETURN_TYPES, ("STRING",))
        self.assertEqual(TUT_AnimaArtistPromptMixer.RETURN_NAMES, ("mixed_prompt",))
        self.assertEqual(TUT_AnimaArtistPromptMixer.CATEGORY, TOOLS_TEXT)

    def test_chinese_punctuation_and_separator_cleanup_are_idempotent(self):
        source = "，masterpiece，，1girl、微笑。；灯光：柔和！（测试）？,"
        expected = "masterpiece, 1girl, 微笑.;灯光:柔和!(测试)?"
        result = normalize_prompt_punctuation(source)
        self.assertEqual(result, expected)
        self.assertEqual(normalize_prompt_punctuation(result), expected)

    def test_top_level_split_preserves_nested_and_quoted_commas(self):
        source = '(red, blue:1.2), "gold, silver", [a,b], {c,d}, tail'
        self.assertEqual(
            split_top_level_commas(source),
            ["(red, blue:1.2)", '"gold, silver"', "[a,b]", "{c,d}", "tail"],
        )

    def test_official_order_inserts_before_first_general_tag(self):
        prompt = "masterpiece, highres, year 2025, safe, 1girl, fern, sousou no frieren, purple hair, smile"
        result = mix_anima_artist_prompt(prompt, artist_data(("wlop", 1.0)))
        self.assertEqual(
            result,
            "masterpiece, highres, year 2025, safe, 1girl, fern, sousou no frieren, @wlop, purple hair, smile",
        )

    def test_conservative_fallback_and_natural_language(self):
        self.assertEqual(
            mix_anima_artist_prompt("1girl, obscure char, obscure series", artist_data(("wlop", 1.0))),
            "1girl, obscure char, obscure series, @wlop",
        )
        self.assertEqual(
            mix_anima_artist_prompt("A girl stands beside a quiet lake.", artist_data(("wlop", 1.0))),
            "@wlop, A girl stands beside a quiet lake.",
        )

    def test_existing_artists_merge_at_first_artist_and_editor_wins(self):
        prompt = "1girl, char, series, @A, smile, (@b:1.4), red hair"
        result = mix_anima_artist_prompt(prompt, artist_data(("a", 1.7), ("c", 1.0)))
        self.assertEqual(result, "1girl, char, series, (@a:1.7), (@b:1.4), @c, smile, red hair")

    def test_real_colon_parenthesis_and_internal_at_artist_names(self):
        self.assertEqual(parse_artist_segment("@qp:flapper"), {"name": "qp:flapper", "weight": 1.0})
        self.assertEqual(parse_artist_segment("(@qp:flapper:1.5)"), {"name": "qp:flapper", "weight": 1.5})
        self.assertEqual(parse_artist_segment("(@7:08)"), {"name": "7:08", "weight": 1.0})
        self.assertEqual(parse_artist_segment("(@7:08:2)"), {"name": "7:08", "weight": 2.0})
        self.assertEqual(
            parse_artist_segment("(@aoi nagisa (metalder):1.2)"),
            {"name": "aoi nagisa (metalder)", "weight": 1.2},
        )
        self.assertEqual(
            parse_artist_segment("@tomatto (@ma!)"),
            {"name": "tomatto (@ma!)", "weight": 1.0},
        )

    def test_empty_artist_list_still_normalizes_prompt(self):
        node = TUT_AnimaArtistPromptMixer()
        self.assertEqual(node.mix_prompt("masterpiece，1girl，smile。", artist_data()), ("masterpiece, 1girl, smile.",))

    def test_artist_data_validation(self):
        invalid_values = [
            "{bad}",
            json.dumps({"version": True, "artists": []}),
            json.dumps({"version": 1, "artists": {}}),
            json.dumps({"version": 1, "artists": [{"name": "a", "weight": True}]}),
            json.dumps({"version": 1, "artists": [{"name": "a", "weight": float("nan")}]}),
            json.dumps({"version": 1, "artists": [{"name": "a", "weight": 0}]}),
            json.dumps({"version": 1, "artists": [{"name": "a", "weight": 3.1}]}),
            json.dumps({"version": 1, "artists": [{"name": "a", "weight": 1.15}]}),
        ]
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_artist_data(value)


class TUTAnimaArtistDatabaseTests(unittest.TestCase):
    def test_bundled_database_shape_and_metadata(self):
        self.assertTrue(ARTIST_DATABASE_PATH.is_file())
        connection = sqlite3.connect(ARTIST_DATABASE_PATH)
        try:
            count, unique_count, minimum, maximum = connection.execute(
                "SELECT COUNT(*), COUNT(DISTINCT tag), MIN(usage_count), MAX(usage_count) FROM artists"
            ).fetchone()
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            connection.close()
        self.assertEqual((count, unique_count, minimum, maximum), (42196, 42196, 50, 16465))
        self.assertEqual(metadata["schema_version"], "1")
        self.assertEqual(metadata["record_count"], "42196")
        self.assertEqual(integrity, "ok")

    def test_exact_punctuation_folded_fuzzy_and_popular_search(self):
        self.assertEqual(search_anima_artists("@qp:flapper", 1)[0]["tag"], "@qp:flapper")
        self.assertEqual(search_anima_artists("qp flapper", 1)[0]["tag"], "@qp:flapper")
        self.assertEqual(search_anima_artists("qp flaper", 5)[0]["tag"], "@qp:flapper")
        popular = search_anima_artists("", 3)
        self.assertEqual(len(popular), 3)
        self.assertGreaterEqual(popular[0]["usage_count"], popular[1]["usage_count"])

    def test_search_validates_query_and_limit(self):
        with self.assertRaisesRegex(ValueError, "128"):
            search_anima_artists("x" * 129)
        with self.assertRaisesRegex(ValueError, "整数"):
            search_anima_artists("wlop", "bad")
        missing = Path(__file__).with_name("missing.sqlite3")
        with self.assertRaisesRegex(RuntimeError, "数据库不存在"):
            search_anima_artists("wlop", database_path=missing)

    def test_frontend_contract_is_present(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "js" / "tut_anima_artist_mixer.js").read_text(encoding="utf-8")
        self.assertIn("AbortController", source)
        self.assertIn("compositionstart", source)
        self.assertIn("setPointerCapture", source)
        self.assertIn("serializeValue = () => widget.value", source)
        self.assertIn("usage_count", source)
        self.assertIn("0.1–3.0", source)
        self.assertIn("/tut_nodes/anima/artists/search", source)
        self.assertIn('addEventListener("dblclick"', source)
        self.assertIn("beginArtistEdit", source)
        self.assertIn("COMPACT_NODE_HEIGHT = 220", source)
        self.assertIn("scheduleEditorResize", source)
        self.assertIn("getMinHeight: () => COMPACT_EDITOR_HEIGHT", source)
        self.assertIn("afterResize:", source)
        self.assertIn("flex:1 1 auto", source)
        self.assertIn("Math.abs(distance) < 5", source)
        self.assertIn('STATE_KEY = "tut_anima_artist_data"', source)
        self.assertIn("widget.serialize = true", source)
        self.assertIn("widget.options.serialize = true", source)
        self.assertIn("node.onSerialize = function", source)
        self.assertIn("widgets_values_named.artist_data", source)
        self.assertIn("candidates.find(isSerializedArtistData)", source)


if __name__ == "__main__":
    unittest.main()
