"""Recompute the numbers of the paper, from the dataset download to a PASS/DIFF report.

    python scripts/reproduce.py all                   every stage below, in order
    python scripts/reproduce.py data [--esa-zip ZIP]  download ESA-Mission1, convert, verify
    python scripts/reproduce.py smed                  SMED, lightweight Phases 1-5 and full Phase 5
    python scripts/reproduce.py ool                   robust OOL, same runs
    python scripts/reproduce.py znorm                 z-normalized SMED, p sweep, both sets
    python scripts/reproduce.py taxonomy              sampling families, features, archetypes
    python scripts/reproduce.py external [--telemanom-zip ZIP]
                                                      Q-GPR predictions (results/) and
                                                      Telemanom predictions (ESA-ADB), scored
    python scripts/reproduce.py recall                per-archetype recall of the paper runs
    python scripts/reproduce.py report                compare with reproduce/paper_values.csv

Stages skip the runs already on disk (``--force`` recomputes them) and write where the
rest of the repository expects them: ``data/ESA-Mission1/``, ``smed/predictions/``,
``runs/``. The report goes to ``runs/reproduce/report.md`` and ``report.csv``; the exit
code is 1 when a value differs from the paper beyond the documented differences.
``--skip STAGE`` (repeatable) leaves a stage of ``all`` out; ``--workers N`` caps the
parallel processes (by default one per core, as many as the available memory allows).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tsadbench.io.misc import load_json, project_root
from tsadbench.reproduce import (MISSION, QGPR_ARCHIVE, QGPR_MD5, RECALL_RUNS, SMED_PHASES,
                                 TELEMANOM_FULL_SETS, ZNORM_PS, output_dir, runs)

ROOT = project_root()
PY = sys.executable
STAGES = ("data", "smed", "ool", "znorm", "taxonomy", "external", "recall", "report")


def sh(*args: str) -> None:
    print("$", " ".join(Path(a).name if i == 0 else a for i, a in enumerate(args)), flush=True)
    subprocess.run([PY, *args], check=True, cwd=ROOT)


def script(name: str) -> str:
    return str(ROOT / "scripts" / name)


def check(name: str, *args: str) -> None:
    """Run a check script, keep its output for the report, do not stop on failure."""
    r = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout, end="")
    output_dir().mkdir(parents=True, exist_ok=True)
    (output_dir() / f"{name}.txt").write_text(r.stdout)


def scored(key: str) -> bool:
    cfg = ROOT / runs()[key] / "config.json"
    return cfg.exists() and "metrics_all_events" in load_json(cfg)


def stage_data(args) -> None:
    from tsadbench.reproduce.data import prepare, verify

    if args.force or verify():
        prepare(args.esa_zip)
    bad = verify()
    msg = (f"{MISSION}: the 58 target channels and 3 tables match reproduce/data_checksums.json"
           if not bad else f"{MISSION}: differ from reproduce/data_checksums.json: {', '.join(bad)}")
    print(msg)
    (output_dir() / "data_check.txt").write_text(msg + "\n")


def stage_smed(args) -> None:
    todo = [(s, ph) for s, ph in [("light", ph) for ph in SMED_PHASES] + [("full", 5)]
            if args.force or not scored(f"smed_{s}_{ph}")]
    for s, ph in todo:
        sh(script("run.py"), str(ROOT / "configs" / "base" / "smed.yaml"), "--set",
           f"mission={MISSION}", f"channels_mode={s}", f"phase={ph}", "experimental_mode=false")
    check("golden_check", script("golden.py"), "verify", "smed/predictions", "golden_checksums.json")


def stage_ool(args) -> None:
    for s, ph in [("light", ph) for ph in SMED_PHASES] + [("full", 5)]:
        if args.force or not scored(f"ool_{s}_{ph}"):
            sh(script("limits_psweep.py"), s, "--phase", str(ph), "--full-metrics", "0.1")


def stage_znorm(args) -> None:
    for s in ("light", "full"):
        if args.force or not all(scored(f"znorm_{s}_{p}") for p in ZNORM_PS):
            sh(script("damp_psweep.py"), s, *map(str, ZNORM_PS))


def stage_taxonomy(args) -> None:
    out = ROOT / "runs" / "taxonomy"
    if args.force or not (out / "taxonomy.json").exists():
        sh(script("taxonomy.py"), "all")
    check("taxonomy_check", script("taxonomy.py"), "check")


def _qgpr_zip() -> Path:
    from tsadbench.reproduce.download import md5sum

    archive = ROOT / QGPR_ARCHIVE
    if not archive.exists() or md5sum(archive) != QGPR_MD5:
        raise SystemExit(f"{QGPR_ARCHIVE} is missing or differs from the paper's (md5): "
                         f"restore it with `git checkout -- {QGPR_ARCHIVE}`")
    return archive


def stage_external(args) -> None:
    qgpr = [k for k in runs() if k.startswith(("qgpr", "max_only"))]
    if args.force or not all((ROOT / runs()[k] / "labels").exists() for k in qgpr):
        archive = _qgpr_zip()
        print(f"unpacking the Q-GPR predictions from {archive.name}")
        with zipfile.ZipFile(archive) as z:
            z.extractall(ROOT)
    todo = [k for k in qgpr if args.force or not scored(k)]
    if todo:
        sh(script("score_runs.py"), *(str(ROOT / runs()[k]) for k in todo),
           "--out", str(output_dir() / "qgpr_metrics.csv"))

    out = ROOT / runs()["telemanom_full"]
    if args.force or not (out / "config.json").exists():
        if args.telemanom_zip:
            sh(script("ingest_telemanom.py"), str(args.telemanom_zip), str(out))
        else:
            from tsadbench.io.telemanom import FULL_SET_RUN, ingest
            from tsadbench.reproduce.download import _open, opener, remote_zip

            print("reading the Telemanom-ESA-Pruned full-set run from the ESA-ADB archive (about 40 MB)")
            op = opener()
            _open(op, TELEMANOM_FULL_SETS).close()   # session cookie of the SharePoint link
            z = remote_zip(TELEMANOM_FULL_SETS + "&download=1", op)
            hp = json.loads(z.read(FULL_SET_RUN + "hyper_params.json"))
            with z.open(FULL_SET_RUN + "anomaly_scores.ts") as scores:
                ingest(scores, hp, out, MISSION, source=f"ESA-ADB Telemanom-ESA full sets:{FULL_SET_RUN}")


def stage_recall(args) -> None:
    specs = [f"{name}={ROOT / runs()[key]}" for name, key in RECALL_RUNS.items()
             if (ROOT / runs()[key] / "labels").exists()]
    missing = [name for name, key in RECALL_RUNS.items() if not (ROOT / runs()[key] / "labels").exists()]
    if missing:
        print(f"recall without {', '.join(missing)} (runs not on disk)")
    windows = ROOT / "runs" / "taxonomy" / "windows.csv"
    extra = ["--windows", str(windows)] if windows.exists() else []
    sh(script("taxonomy.py"), "--out", str(output_dir()), "recall", *specs, *extra)


def stage_report(args) -> int:
    from tsadbench.reproduce.report import compare, read_registry, write_report

    timings_file = output_dir() / "timings.json"
    timings = json.loads(timings_file.read_text()) if timings_file.exists() else None
    checks = {name: (output_dir() / f"{name}.txt").read_text().strip()
              for name in ("data_check", "golden_check", "taxonomy_check") if (output_dir() / f"{name}.txt").exists()}
    result = compare(read_registry())
    summary = write_report(result, output_dir(), timings, checks)
    print(f"{len(result)} paper values: {summary}")
    print(f"report: {output_dir() / 'report.md'}")
    for r in result[result.status == "DIFF"].itertuples():
        print(f"  DIFF {r.item}, {r.row}, {r.column}: paper {r.paper}, here {r.observed}")
    return int((result.status == "DIFF").any())


FUNCS = {"data": stage_data, "smed": stage_smed, "ool": stage_ool, "znorm": stage_znorm,
         "taxonomy": stage_taxonomy, "external": stage_external, "recall": stage_recall,
         "report": stage_report}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("all", *STAGES))
    ap.add_argument("--force", action="store_true", help="recompute runs already on disk")
    ap.add_argument("--skip", action="append", default=[], choices=STAGES, help="stage left out of 'all'")
    ap.add_argument("--esa-zip", type=Path, help="ESA-Mission1.zip already downloaded from Zenodo")
    ap.add_argument("--telemanom-zip", type=Path, help="Telemanom-ESA full-sets zip already downloaded")
    ap.add_argument("--workers", type=int, help="at most this many parallel processes (default: one per "
                                                "core, as many as the available memory allows)")
    args = ap.parse_args()
    if args.workers:
        os.environ["TSADBENCH_WORKERS"] = str(args.workers)   # read by the stage subprocesses

    stages = [s for s in STAGES if s not in args.skip] if args.stage == "all" else [args.stage]
    output_dir().mkdir(parents=True, exist_ok=True)
    timings_file = output_dir() / "timings.json"
    timings = json.loads(timings_file.read_text()) if timings_file.exists() else {}
    code = 0
    for s in stages:
        print(f"\n=== {s} ===", flush=True)
        t0 = time.time()
        code = FUNCS[s](args) or 0
        if s != "report":
            timings[s] = round(time.time() - t0, 1)
            timings_file.write_text(json.dumps(timings, indent=2) + "\n")
    sys.exit(code)


if __name__ == "__main__":
    main()
