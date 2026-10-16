"""Comparison of the recomputed numbers with the paper values of ``reproduce/paper_values.csv``.

Each row of the registry gives the paper value (as printed and as a number), a tolerance
and a ``source``, which says where the value is recomputed:

    metric:<run>:<mode>:<block>[/<key>]   metric of a run's config.json (mode all_events or
                                          anomalies_alone; e.g. event_wise/f0.5)
    recall:<method>:<archetype>:<column>  per-archetype recall table (hits, n_windows, n_events)
    taxonomy:<key>[/<key>]                runs/taxonomy/taxonomy.json
    family:<S1..S8>:<column>              runs/taxonomy/sampling_families.csv
    archetype:<A1..A8>:<column>           runs/taxonomy/archetypes.csv
    decisions:<run>                       labels of a run over the test period (rows, all channels)
    seeds:<precision|recall|f0.5>:<stat>  event-wise scores of the 15 Q-GPR seed runs
                                          (mean, std, min, max, mode, mode_count, cv)

``<run>`` is a key of ``tsadbench.reproduce.runs()``. A value matches when it is within
the tolerance of the paper value (PASS); a documented difference matches the
``reproduced`` column instead (KNOWN); a value that cannot be computed because its run is
missing is MISSING.
"""
import json
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tsadbench.io.misc import project_root
from tsadbench.reproduce import QGPR_SEEDS, output_dir, runs

STATUSES = ("PASS", "KNOWN", "DIFF", "MISSING")


def registry_path() -> Path:
    return project_root() / "reproduce" / "paper_values.csv"


def read_registry(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or registry_path(), dtype=str, keep_default_na=False)


class Missing(Exception):
    """The value cannot be computed yet (run or table not on disk)."""


@lru_cache(maxsize=None)
def _json(path: Path) -> dict:
    if not path.exists():
        raise Missing(str(path.relative_to(project_root())))
    return json.loads(path.read_text())


@lru_cache(maxsize=None)
def _csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise Missing(str(path.relative_to(project_root())))
    return pd.read_csv(path)


def _run_dir(key: str) -> Path:
    return project_root() / runs()[key]


def _metric(key: str, mode: str, path: str) -> float:
    cfg = _json(_run_dir(key) / "config.json")
    block = cfg.get(f"metrics_{mode}")
    if block is None:
        raise Missing(f"{runs()[key]} not scored")
    v = block
    for k in path.split("/"):
        v = v[k]
    return float(v)


def _decisions(key: str) -> int:
    labels = sorted((_run_dir(key) / "labels").glob("channel_*.parquet"))
    if not labels:
        raise Missing(str(runs()[key]))
    return sum(pq.ParquetFile(f).metadata.num_rows for f in labels)


def _seeds(metric: str, stat: str) -> float:
    x = np.array([_metric(f"qgpr_seed_{s}", "all_events", f"event_wise/{metric}") for s in QGPR_SEEDS])
    if stat == "mode":
        return Counter(x.tolist()).most_common(1)[0][0]
    if stat == "mode_count":
        return Counter(x.tolist()).most_common(1)[0][1]
    return {"mean": x.mean(), "std": x.std(ddof=1), "min": x.min(), "max": x.max(),
            "cv": x.std(ddof=1) / x.mean()}[stat]


def resolve(source: str):
    kind, *a = source.split(":")
    root = project_root()
    if kind == "metric":
        return _metric(*a)
    if kind == "recall":
        method, arch, col = a
        T = _csv(output_dir() / "archetype_recall.csv").set_index("archetype")
        name = f"{method}_{col}" if col in ("hits", "recall") else col
        if name not in T.columns:
            raise Missing(f"{method} not in the recall table")
        return float(T.loc[arch, name])
    if kind == "taxonomy":
        v = _json(root / "runs" / "taxonomy" / "taxonomy.json")
        for k in a[0].split("/"):
            v = v[k]
        return v[0] if isinstance(v, list) else v
    if kind in ("family", "archetype"):
        name, col = a
        file, index = {"family": ("sampling_families.csv", "family"),
                       "archetype": ("archetypes.csv", "archetype_id")}[kind]
        return _csv(root / "runs" / "taxonomy" / file).set_index(index).loc[name, col]
    if kind == "decisions":
        return _decisions(a[0])
    if kind == "seeds":
        return _seeds(*a)
    raise ValueError(f"unknown source {source}")


def _matches(observed, expected: str, tol: str) -> bool:
    try:
        e = float(expected)
    except ValueError:
        return str(observed) == expected
    return abs(float(observed) - e) <= float(tol or 0) + 1e-9


def compare(registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in registry.itertuples(index=False):
        try:
            obs = resolve(r.source)
        except (Missing, KeyError) as e:
            rows.append({**r._asdict(), "observed": "", "status": "MISSING", "detail": str(e)})
            continue
        if _matches(obs, r.expected, r.tol):
            status = "PASS"
        elif r.reproduced and _matches(obs, r.reproduced, r.tol):
            status = "KNOWN"
        else:
            status = "DIFF"
        if isinstance(obs, (float, np.floating)):
            obs = float(f"{obs:.6g}")
        rows.append({**r._asdict(), "observed": obs, "status": status, "detail": ""})
    return pd.DataFrame(rows)


def _environment() -> str:
    from importlib.metadata import version

    import numba
    try:
        commit = subprocess.run(["git", "-C", str(project_root()), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "n/a"
    return (f"python {platform.python_version()} on {platform.platform()}; numpy {np.__version__}, "
            f"numba {numba.__version__}, rocket-fft {version('rocket-fft')}, pandas {pd.__version__}; "
            f"commit {commit}")


def write_report(result: pd.DataFrame, out: Path, timings: dict | None = None,
                 checks: dict | None = None) -> str:
    """Write ``report.csv`` and ``report.md`` to ``out``; returns the one-line summary."""
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "report.csv", index=False)
    counts = result.status.value_counts()
    summary = ", ".join(f"{counts.get(s, 0)} {s}" for s in STATUSES)
    lines = ["# Reproduction of the paper numbers", "",
             f"{len(result)} values: {summary}.", "",
             f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC; {_environment()}.", ""]
    if timings:
        lines += ["Stage times: " + ", ".join(f"{k} {v / 60:.1f} min" for k, v in timings.items()) + ".", ""]
    titles = {"data_check": "Converted data against reproduce/data_checksums.json",
              "golden_check": "SMED labels against golden_checksums.json",
              "taxonomy_check": "Taxonomy tables against data/ESA-Mission1/taxonomy/"}
    for name, text in (checks or {}).items():
        lines += [f"{titles.get(name, name)}:", "", "```", text, "```", ""]
    for status, title in (("DIFF", "Differences"), ("KNOWN", "Known differences"), ("MISSING", "Not computed")):
        sub = result[result.status == status]
        if len(sub):
            lines += [f"## {title}", ""]
            for r in sub.itertuples():
                what = r.note or r.detail
                lines.append(f"- {r.item}, {r.row}, {r.column}: paper {r.paper}, here "
                             f"{r.observed if r.observed != '' else 'n/a'}" + (f" ({what})" if what else ""))
            lines.append("")
    lines += ["## All values", ""]
    for item, sub in result.groupby("item", sort=False):
        lines += [f"### {item}", "", "| Row | Column | Paper | Here | Status |", "|---|---|---|---|---|"]
        lines += [f"| {r.row} | {r.column} | {r.paper} | {r.observed} | {r.status} |" for r in sub.itertuples()]
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
