"""Parser tests, using output shapes observed in the real results log."""

import pytest

from hansard_llm import schema


class TestParseJson:
    def test_clean_object(self):
        ex = schema.parse_json(
            '{"mentions_topic": true, "subthemes": ["NHS funding", "waiting lists"],'
            ' "evidence_quote": "the health service"}', 5)
        assert ex.parse_ok and ex.mentions_topic is True
        assert ex.subthemes == ["NHS funding", "waiting lists"]
        assert ex.subthemes_raw == ex.subthemes

    def test_code_fence_and_prose_prefix(self):
        ex = schema.parse_json(
            'Here is the JSON:\n```json\n{"mentions_topic": false, '
            '"subthemes": [], "evidence_quote": ""}\n```', 5)
        assert ex.parse_ok and ex.mentions_topic is False

    def test_negative_verdict_keeps_raw_themes(self):
        ex = schema.parse_json(
            '{"mentions_topic": false, "subthemes": ["poor law"], '
            '"evidence_quote": ""}', 5)
        assert ex.subthemes == []           # verdict-gated view
        assert ex.subthemes_raw == ["poor law"]  # nothing discarded

    def test_string_bool_coercion(self):
        ex = schema.parse_json('{"mentions_topic": "Yes", "subthemes": []}', 5)
        assert ex.mentions_topic is True

    def test_no_json(self):
        ex = schema.parse_json("The speech is about railways.", 5)
        assert not ex.parse_ok and ex.mentions_topic is None

    def test_cap_applies(self):
        subs = [f"theme {i}" for i in range(10)]
        ex = schema.parse_json(
            '{"mentions_topic": true, "subthemes": %s}'
            % str(subs).replace("'", '"'), 5)
        assert len(ex.subthemes) == 5


class TestParseFree:
    def test_explicit_yes_with_list(self):
        ex = schema.parse_free(
            "Yes, the speech substantively discusses health care.\n"
            "- hospital funding\n- nurse pay", 5)
        assert ex.mentions_topic is True
        assert ex.presence_inferred is False
        assert ex.subthemes == ["hospital funding", "nurse pay"]

    def test_inferred_presence_is_flagged(self):
        # No yes/no signal, but a bulleted list -> parser INFERS presence and
        # must say so; unflagged, this inflated the free-format positive rate.
        # (fixture avoids verbs like "covers"/"discusses" that the parser
        # legitimately reads as an explicit affirmative)
        ex = schema.parse_free(
            "Key themes:\n- workhouse conditions\n- outdoor relief", 5)
        assert ex.mentions_topic is True
        assert ex.presence_inferred is True

    def test_explicit_no(self):
        ex = schema.parse_free("No. The speech concerns railway gauges.", 5)
        assert ex.mentions_topic is False and not ex.presence_inferred

    def test_negative_with_list_keeps_raw(self):
        ex = schema.parse_free(
            "No, this is not substantively about health care, though it "
            "touches on:\n- army sanitation", 5)
        assert ex.mentions_topic is False
        assert ex.subthemes == []
        assert ex.subthemes_raw == ["army sanitation"]

    def test_labelled_answer(self):
        ex = schema.parse_free("mentions_topic: false\nNothing relevant.", 5)
        assert ex.mentions_topic is False

    def test_no_signal_at_all(self):
        ex = schema.parse_free("An interesting speech about many things.", 5)
        assert not ex.parse_ok and ex.mentions_topic is None


def test_normalize_themes_dedup_and_none_words():
    out = schema.normalize_themes(
        ["  NHS funding ", "nhs funding", "None", "waiting lists"], 5)
    assert out == ["NHS funding", "waiting lists"]


def test_parse_dispatch_rejects_unknown_format():
    with pytest.raises(ValueError):
        schema.parse("x", "yaml", 5)
