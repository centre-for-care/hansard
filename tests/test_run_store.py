"""Results-store tests: cache keys, done-key scanning across run dirs, and
reparse round-trips."""

import json

import pandas as pd
import pytest

from hansard_llm import config, run


def test_cache_key_roundtrip():
    k = run._cache_key(123, "abc123", "m/x", 0.0, 42, 0)
    assert k == "123|abc123|m/x|0.0|42|0"


def test_load_done_keys(tmp_path):
    log = tmp_path / "results.jsonl"
    rows = [
        {"speech_id": 1, "prompt_hash": "h1", "model_id": "m", "temperature": 0.0,
         "seed": 42, "rep": 0},
        {"speech_id": 2, "prompt_hash": "h1", "model_id": "m", "temperature": 0.0,
         "seed": 42, "rep": 0},
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\ncorrupt line\n",
                   encoding="utf-8")
    keys = run._load_done_keys(log)
    assert len(keys) == 2  # corrupt line skipped, not fatal


def test_experiment_done_keys_scans_all_runs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    for rid, sid in (("r1", 1), ("r2", 2)):
        d = runs / "exp" / rid
        d.mkdir(parents=True)
        (d / "results.jsonl").write_text(json.dumps(
            {"speech_id": sid, "prompt_hash": "h", "model_id": "m",
             "temperature": 0.0, "seed": 42, "rep": 0}) + "\n",
            encoding="utf-8")
    monkeypatch.setattr(config, "RUNS_DIR", runs)
    keys = run._experiment_done_keys("exp", include_legacy=False)
    assert keys == {"1|h|m|0.0|42|0", "2|h|m|0.0|42|0"}
    assert run._experiment_done_keys("other", include_legacy=False) == set()


def test_reparse_results_recomputes_from_raw_text():
    df = pd.DataFrame([
        {"raw_text": '{"mentions_topic": true, "subthemes": ["x"]}',
         "output_format": "json", "task": "v1", "error": None},
        {"raw_text": float("nan"), "output_format": "json", "task": "v1",
         "error": "timeout"},
    ])
    out = run.reparse_results(df)
    assert out.loc[0, "mentions_topic"] is True
    assert out.loc[1, "parse_ok"] == False  # noqa: E712 — NaN raw_text -> failed row
    assert out.loc[1, "parse_error"] == "timeout"
    assert "presence_inferred" in out.columns and "subthemes_raw" in out.columns


def test_completion_budget_uncapped_does_not_shrink_reasoning():
    from hansard_llm.config import ModelSpec
    from hansard_llm.prompts import TASK_UNCAPPED

    core = ModelSpec("x/core", family="qwen", max_tokens=512)
    think = ModelSpec("x/think", family="qwen", reasoning=True, max_tokens=4096)
    assert run.completion_budget(core, task=TASK_UNCAPPED) == 1024
    assert run.completion_budget(think, task=TASK_UNCAPPED) == 4096
    assert run.completion_budget(think, task=TASK_UNCAPPED, override=8192) == 8192
    assert run.completion_budget(think, task="v1") == 4096


def test_load_experiment_keeps_latest_duplicate(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    monkeypatch.setattr(config, "RUNS_DIR", runs)
    key = {"speech_id": 1, "prompt_hash": "h", "model_id": "m",
           "temperature": 0.0, "seed": 42, "rep": 0,
           "raw_text": '{"mentions_topic": false}',
           "output_format": "json", "task": "v1", "error": None}
    for rid, text in (("r1", "old"), ("r2", "new")):
        d = runs / "exp" / rid
        d.mkdir(parents=True)
        row = {**key, "raw_text": json.dumps({"mentions_topic": text == "new",
                                              "subthemes": [text]})}
        (d / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    df = run.load_experiment("exp", reparse=False)
    assert len(df) == 1
    assert json.loads(df.iloc[0]["raw_text"])["subthemes"] == ["new"]


def test_uncapped_reparse_not_retruncated():
    subs = [f"t{i}" for i in range(12)]
    raw = json.dumps({"mentions_topic": True, "subthemes": subs})
    df = pd.DataFrame([
        {"raw_text": raw, "output_format": "json", "task": "v1_nocap", "error": None},
        {"raw_text": raw, "output_format": "json", "task": "v1", "error": None},
    ])
    out = run.reparse_results(df)
    assert len(out.loc[0, "subthemes"]) == 12   # uncapped arm keeps all
    assert len(out.loc[1, "subthemes"]) == 5    # capped arm capped at 5
