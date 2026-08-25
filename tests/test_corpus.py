"""Eligible-pool filter, decade×chamber packing, and corpus CLI helpers."""

import json

import pandas as pd

from hansard_llm import config, corpus, run, sample
from hansard_llm.client import CallResult
from hansard_llm.prompts import TASK_UNCAPPED


def test_nonnegative_mod_sql_matches_python():
    import duckdb
    con = duckdb.connect(":memory:")
    sql = corpus.nonnegative_mod_sql("x", "k")
    got = con.execute(
        f"SELECT {sql} FROM (VALUES (-5, 2), (-5, 3), (5, 2), (4, 2)) t(x, k)"
    ).fetchall()
    assert got == [(1,), (1,), (1,), (0,)]
    assert [(-5) % 2, (-5) % 3, 5 % 2, 4 % 2] == [1, 1, 1, 0]


def test_eligible_where_sql_matches_eval_hygiene():
    sql = sample.eligible_where_sql()
    assert "speech_text IS NOT NULL" in sql
    assert "NOT procedural" in sql
    assert f"word_count >= {sample.MIN_WORDS}" in sql
    for tier in sample.LENGTH_TIERS:
        assert f"'{tier}'" in sql
    for chamber in sample.ELIGIBLE_CHAMBERS:
        assert f"'{chamber}'" in sql
    assert "year IS NOT NULL" in sql
    # Custom chambers/min_words still flow through (pilot design).
    custom = sample.eligible_where_sql(min_words=12, chambers=("Commons",))
    assert "word_count >= 12" in custom
    assert "Lords" not in custom
    aliased = sample.eligible_where_sql(table="e")
    assert "e.chamber IN" in aliased
    assert "e.speech_text IS NOT NULL" in aliased


def test_pack_splits_only_cells_above_cap_and_balances():
    # Real eligible-pool cell sizes (Commons/Lords × decade), n=4,274,315.
    cells = pd.DataFrame([
        (1800, "Commons", 4949), (1800, "Lords", 890),
        (1810, "Commons", 6656), (1810, "Lords", 781),
        (1820, "Commons", 9114), (1820, "Lords", 825),
        (1830, "Commons", 32363), (1830, "Lords", 4625),
        (1840, "Commons", 30066), (1840, "Lords", 4353),
        (1850, "Commons", 42182), (1850, "Lords", 6800),
        (1860, "Commons", 44586), (1860, "Lords", 6277),
        (1870, "Commons", 46869), (1870, "Lords", 5815),
        (1880, "Commons", 90392), (1880, "Lords", 7488),
        (1890, "Commons", 100942), (1890, "Lords", 5821),
        (1900, "Commons", 129881), (1900, "Lords", 12116),
        (1910, "Commons", 86364), (1910, "Lords", 13550),
        (1920, "Commons", 52019), (1920, "Lords", 16750),
        (1930, "Commons", 80338), (1930, "Lords", 15096),
        (1940, "Commons", 115962), (1940, "Lords", 24514),
        (1950, "Commons", 154779), (1950, "Lords", 33730),
        (1960, "Commons", 176547), (1960, "Lords", 76449),
        (1970, "Commons", 201140), (1970, "Lords", 105486),
        (1980, "Commons", 200339), (1980, "Lords", 163292),
        (1990, "Commons", 200404), (1990, "Lords", 166315),
        (2000, "Commons", 254416), (2000, "Lords", 221866),
        (2010, "Commons", 631468), (2010, "Lords", 223523),
        (2020, "Commons", 321004), (2020, "Lords", 145173),
    ], columns=["decade_bin", "chamber", "n"])
    pieces, cap, total = corpus.pieces_from_cells(cells, n_shards=8)
    assert total == 4_274_315
    assert cap == 534_290
    split = {(p.decade_bin, p.chamber) for p in pieces if p.k > 1}
    assert split == {(2010, "Commons")}
    assert sum(p.k == 1 for p in pieces) == 45
    assert sum(p.k == 2 for p in pieces) == 2
    mapping = corpus.pack_pieces(pieces, n_shards=8)
    keys = list(mapping)
    assert len(keys) == len(set(keys))
    assert set(mapping.values()) == set(range(8))
    piece_df = corpus.piece_map_frame(pieces, mapping)
    loads = piece_df.groupby("shard_id")["n_est"].sum()
    assert loads.max() / loads.min() <= 1.01
    # Every original cell is covered exactly once (split parts sum to n).
    got = (piece_df.groupby(["decade_bin", "chamber"], as_index=False)["n_est"]
           .sum().rename(columns={"n_est": "n"}))
    merged = cells.merge(got, on=["decade_bin", "chamber"])
    assert (merged["n_x"] == merged["n_y"]).all()


def test_load_shard_filters_and_limit(tmp_path):
    df = pd.DataFrame({
        "speech_id": [10, 11, 12, 13],
        "year": [2011, 2012, 1991, 1992],
        "decade_bin": [2010, 2010, 1990, 1990],
        "chamber": ["Commons"] * 4,
        "speech_type": ["medium"] * 4,
        "word_count": [80, 90, 100, 110],
        "section_title": ["s"] * 4,
        "speech_text": ["aa", "bb", "cc", "dd"],
        "shard_id": [0, 0, 1, 1],
    })
    path = tmp_path / "eligible_pool.parquet"
    df.to_parquet(path, index=False)
    s0 = corpus.load_shard(0, path=path, columns=["speech_id", "shard_id"])
    assert set(s0["speech_id"]) == {10, 11}
    limited = corpus.load_shard(1, path=path, columns=["speech_id"], limit=1)
    assert len(limited) == 1
    assert limited.iloc[0]["speech_id"] == 12  # ORDER BY speech_id


def test_experiment_names_do_not_mix_models():
    assert (corpus.experiment_for_model(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16") == "corpus_nemotron")
    assert corpus.experiment_for_model(
        "Qwen/Qwen3-30B-A3B-Instruct-2507") == "corpus_qwen30"
    assert (corpus.experiment_for_model("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
            != corpus.experiment_for_model("Qwen/Qwen3-30B-A3B-Instruct-2507"))


def test_corpus_plan_is_one_cell_per_speech():
    spec = config.MODELS_BY_ID["Qwen/Qwen3-30B-A3B-Instruct-2507"]
    speeches = pd.DataFrame({"speech_id": [1, 2], "speech_text": ["x", "y"]})
    plan = corpus.corpus_plan(spec, speeches)
    assert plan.pool == "eligible"
    assert len(plan.variants) == 1
    assert plan.variants[0].definition == config.DEFAULT_TOPIC.definition_id
    assert plan.variants[0].task == TASK_UNCAPPED
    assert plan.conditions == (run.CORE,)
    jobs = run._build_jobs(plan, set())
    assert len(jobs) == 2


def test_dry_run_without_speech_text(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    spec = config.MODELS_BY_ID["Qwen/Qwen3-30B-A3B-Instruct-2507"]
    speeches = pd.DataFrame({"speech_id": [7, 8]})  # no speech_text
    plan = corpus.corpus_plan(spec, speeches)
    summary = corpus.dry_run(plan, experiment="corpus_qwen30", shard=0)
    assert summary["n_speeches"] == 2
    assert summary["cells_would_run"] == 2
    assert summary["cells_cached_experiment"] == 0


class _FakeClient:
    def complete(self, *args, **kwargs):
        return CallResult(
            model_id="fake",
            text='{"mentions_topic": false, "subthemes": [], "evidence_quote": ""}',
            reasoning=None,
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            latency_s=0.0,
            temperature=0.0,
            seed=42,
            attempts=1,
        )


def test_execute_bounded_queue_writes_every_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(run, "LLMClient", lambda: _FakeClient())
    spec = config.MODELS_BY_ID["Qwen/Qwen3-30B-A3B-Instruct-2507"]
    speeches = pd.DataFrame({
        "speech_id": list(range(40)),
        "speech_text": ["hello"] * 40,
    })
    plan = corpus.corpus_plan(spec, speeches, max_workers=2)
    n = run.execute(plan, experiment="corpus_qwen30", verbose=False,
                    include_legacy_cache=False, max_inflight=4,
                    cli_args={"shard": 3})
    assert n == 40
    n2 = run.execute(plan, experiment="corpus_qwen30", verbose=False,
                     include_legacy_cache=False, max_inflight=4)
    assert n2 == 0
    logs = list((tmp_path / "runs" / "corpus_qwen30").glob("*/results.jsonl"))
    assert len(logs) == 1
    rows = [json.loads(l) for l in logs[0].read_text(encoding="utf-8").splitlines()
            if l.strip()]
    assert len(rows) == 40
    assert {r["speech_id"] for r in rows} == set(range(40))
    assert all(r["shard"] == 3 for r in rows)
    assert all(r["parse_ok"] for r in rows)
    stats = run.experiment_cell_stats("corpus_qwen30")
    assert stats == {"n": 40, "parse_ok": 40}
    dest = tmp_path / "slim.parquet"
    run.compact_experiment_to_parquet("corpus_qwen30", dest)
    slim = pd.read_parquet(dest)
    assert "raw_text" not in slim.columns
    assert "reasoning" not in slim.columns
    assert len(slim) == 40
