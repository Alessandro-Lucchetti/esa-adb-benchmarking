"""Reproduction package: registry of paper values, comparison logic, Telemanom reader."""
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

import tsadbench.reproduce as reproduce
from tsadbench.io.misc import project_root
from tsadbench.io.telemanom import read_binary_scores
from tsadbench.reproduce import report
from tsadbench.reproduce.report import compare, read_registry

SOURCE_KINDS = {"metric", "recall", "taxonomy", "family", "archetype", "decisions", "seeds"}


def test_registry_is_well_formed():
    R = read_registry()
    assert list(R.columns) == ["item", "row", "column", "paper", "expected", "tol", "reproduced", "note",
                               "source"]
    assert not R.duplicated(["item", "row", "column"]).any()
    assert (R.paper != "").all() and (R.expected != "").all()
    for tol in R.tol[R.tol != ""]:
        assert float(tol) >= 0
    # a documented difference says why
    assert (R.note[R.reproduced != ""] != "").all()

    keys = reproduce.runs()
    for src in R.source:
        kind, *a = src.split(":")
        assert kind in SOURCE_KINDS, src
        if kind in ("metric", "decisions"):
            assert a[0] in keys, src
        if kind == "metric":
            assert a[1] in ("all_events", "anomalies_alone"), src
        if kind == "recall":   # window and event counts are per archetype, not per method
            assert a[0] in reproduce.RECALL_RUNS or (a[0] == "all" and a[2] not in ("hits", "recall")), src


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(reproduce, "project_root", lambda: tmp_path)
    monkeypatch.setattr(report, "project_root", lambda: tmp_path)
    report._json.cache_clear()
    report._csv.cache_clear()
    run = tmp_path / reproduce.runs()["smed_full_5"]
    (run / "labels").mkdir(parents=True)
    cfg = {"metrics_all_events": {"event_wise": {"f0.5": "0.902", "precision": "0.982"},
                                  "alarming_precision": "0.721"}}
    (run / "config.json").write_text(json.dumps(cfg))
    for ch, n in (("channel_41", 3), ("channel_42", 4)):
        pd.DataFrame({"timestamp": pd.date_range("2007-01-01", periods=n, freq="30s"),
                      "label": np.zeros(n, np.int8)}).to_parquet(run / "labels" / f"{ch}.parquet")
    yield tmp_path
    report._json.cache_clear()
    report._csv.cache_clear()


def test_compare_statuses(fake_repo):
    rows = [
        # item, row, column, paper, expected, tol, reproduced, note, source
        ("T", "pass", "", "0.902", "0.902", "0.0005", "", "", "metric:smed_full_5:all_events:event_wise/f0.5"),
        ("T", "known", "", "0.689", "0.689", "0.0005", "0.721", "float32",
         "metric:smed_full_5:all_events:alarming_precision"),
        ("T", "diff", "", "0.990", "0.990", "0.0005", "", "", "metric:smed_full_5:all_events:event_wise/precision"),
        ("T", "unscored", "", "0.5", "0.5", "0.0005", "", "", "metric:smed_full_5:anomalies_alone:event_wise/f0.5"),
        ("T", "no run", "", "0.5", "0.5", "0.0005", "", "", "metric:ool_full_5:all_events:event_wise/f0.5"),
        ("T", "decisions", "", "7", "7", "0", "", "", "decisions:smed_full_5"),
    ]
    R = pd.DataFrame(rows, columns=["item", "row", "column", "paper", "expected", "tol", "reproduced", "note",
                                    "source"])
    status = compare(R).set_index("row").status
    assert status.to_dict() == {"pass": "PASS", "known": "KNOWN", "diff": "DIFF", "unscored": "MISSING",
                                "no run": "MISSING", "decisions": "PASS"}


def test_write_report(fake_repo):
    R = pd.DataFrame([("Table 6", "SMED", "F0.5", "0.902", "0.902", "0.0005", "", "",
                       "metric:smed_full_5:all_events:event_wise/f0.5")],
                     columns=["item", "row", "column", "paper", "expected", "tol", "reproduced", "note", "source"])
    summary = report.write_report(compare(R), fake_repo / "out", timings={"smed": 60.0},
                                  checks={"golden_check": "PASS 88/88"})
    assert summary == "1 PASS, 0 KNOWN, 0 DIFF, 0 MISSING"
    md = (fake_repo / "out" / "report.md").read_text(encoding="utf-8")
    assert "| SMED | F0.5 | 0.902 | 0.902 | PASS |" in md and "PASS 88/88" in md


def _scores(rows: list[list[int]]) -> bytes:
    cell = {0: b"0.000000000000000000e+00", 1: b"1.000000000000000000e+00"}
    return b"".join(b",".join(cell[v] for v in r) + b"\r\n" for r in rows)


def test_read_binary_scores():
    rows = [[0, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0]]
    out = read_binary_scores(io.BytesIO(_scores(rows)), 4, 3)
    assert out.dtype == np.int8 and out.tolist() == rows


@pytest.mark.parametrize("data, n_rows, msg", [
    (_scores([[0, 1]] * 3), 4, "file ends"),
    (_scores([[0, 1]] * 3), 2, "more than"),
    (_scores([[0, 1]]) + b"0.500000000000000000e+00,1.000000000000000000e+00\r\n", 2, "not binary"),
])
def test_read_binary_scores_rejects(data, n_rows, msg):
    with pytest.raises(ValueError, match=msg):
        read_binary_scores(io.BytesIO(data), n_rows, 2)


def test_worker_count(monkeypatch):
    from tsadbench.runner import workers

    monkeypatch.setattr(workers.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(workers, "available_memory", lambda: None)
    monkeypatch.delenv("TSADBENCH_WORKERS", raising=False)
    assert workers.worker_count(58) == 32
    assert workers.worker_count(6) == 6
    assert workers.worker_count(58, cap=30) == 30
    monkeypatch.setenv("TSADBENCH_WORKERS", "4")
    assert workers.worker_count(58) == 4
    monkeypatch.delenv("TSADBENCH_WORKERS")
    monkeypatch.setattr(workers, "available_memory", lambda: 16 * 2**30)
    assert workers.worker_count(58) == 10            # 1.5 GB each
    monkeypatch.setattr(workers, "available_memory", lambda: 2**29)
    assert workers.worker_count(58) == 1


def test_qgpr_archive_holds_every_qgpr_run():
    from tsadbench.reproduce.download import md5sum

    archive = project_root() / reproduce.QGPR_ARCHIVE

    assert md5sum(archive) == reproduce.QGPR_MD5
    names = set(zipfile.ZipFile(archive).namelist())
    qgpr = [rel for rel in reproduce.runs().values() if rel.parts[0] == "runs" and rel.parts[1].startswith("qgpr")]
    assert len(qgpr) == 28
    for rel in qgpr:
        assert f"{rel.as_posix()}/config.json" in names


def test_cgroup_free_counts_the_page_cache(tmp_path):
    from tsadbench.runner.workers import cgroup_free

    gib = 2**30
    (tmp_path / "memory.max").write_text(f"{8 * gib}\n")
    (tmp_path / "memory.current").write_text(f"{int(7.5 * gib)}\n")
    (tmp_path / "memory.stat").write_text(f"anon {2 * gib}\nfile {int(5.5 * gib)}\ninactive_file {gib}\n")
    assert cgroup_free(str(tmp_path)) == 6 * gib    # 8 - 7.5 + 5.5
    (tmp_path / "memory.max").write_text("max\n")
    assert cgroup_free(str(tmp_path)) is None
    assert cgroup_free(str(tmp_path / "missing")) is None
