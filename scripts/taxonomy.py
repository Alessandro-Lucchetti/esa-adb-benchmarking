"""Telemetry characterization and archetype taxonomy of ESA-ADB Mission 1.

Reads the raw channels (``data/ESA-Mission1/raw/``) and ``labels.csv``; writes CSV tables
to ``--out`` (default ``runs/taxonomy``). Reference tables of the paper are in
``data/ESA-Mission1/taxonomy/``.

    python scripts/taxonomy.py all                    sampling + features + classify
    python scripts/taxonomy.py sampling               sampling_stats.csv, sampling_families.csv
    python scripts/taxonomy.py features               window_features.csv (needs sampling_stats.csv)
    python scripts/taxonomy.py classify [--features CSV]
                                                      units, windows, archetypes, dendrogram, taxonomy.json
    python scripts/taxonomy.py check [DIR]            compare DIR with the reference tables
    python scripts/taxonomy.py recall NAME=RUN_DIR ... [--channels full|light] [--windows CSV]
                                                      archetype_recall.csv, archetype_recall_windows.csv

``classify`` also writes the summary numbers quoted in the paper text (``summary`` in
``taxonomy.json``, see ``tsadbench.taxonomy.summary``). Without ``--features`` it reads ``window_features.csv`` from ``--out`` if present,
else the reference one, so the taxonomy can be rebuilt without the telemetry.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tsadbench.io.misc import load_metadata, project_root
from tsadbench.taxonomy import TABLES, compare_tables, read_table, reference_dir, write_taxonomy


def cmd_sampling(args) -> None:
    from tsadbench.taxonomy.sampling import family_table, sampling_table

    print("sampling statistics")
    stats = sampling_table(args.mission)
    stats.to_csv(args.out / "sampling_stats.csv", index=False)
    family_table(stats).to_csv(args.out / "sampling_families.csv", index=False)
    print(f"  {len(stats)} channels -> {args.out / 'sampling_stats.csv'}")


def cmd_features(args) -> None:
    from tsadbench.taxonomy.features import feature_table
    from tsadbench.taxonomy.windows import load_windows

    stats_path = args.out / "sampling_stats.csv"
    if not stats_path.exists():
        cmd_sampling(args)
    windows = load_windows(args.mission)
    print(f"window features ({len(windows)} labelled intervals on target channels)")
    F = feature_table(windows, read_table(stats_path), args.mission)
    F.to_csv(args.out / "window_features.csv", index=False)
    print(f"  {len(F)} windows -> {args.out / 'window_features.csv'}")


def cmd_classify(args) -> None:
    from tsadbench.taxonomy.archetypes import build_taxonomy
    from tsadbench.taxonomy.classify import density_clusters, signature, split_clusters
    from tsadbench.taxonomy.summary import summary_numbers
    from tsadbench.taxonomy.windows import load_windows

    path = args.features or next(p for p in (args.out / "window_features.csv",
                                             reference_dir(args.mission) / "window_features.csv")
                                 if p.exists())
    F = read_table(path)
    print(f"classification of {len(F)} windows from {path}")
    Z, info = signature(F)
    clusters, cat = density_clusters(Z, F)
    lab = split_clusters(clusters, F)
    tax = build_taxonomy(F, lab, cat, load_windows(args.mission))
    tax["meta"]["summary"] = summary_numbers(F, tax["windows"], tax["units"], clusters)
    write_taxonomy(tax, info, args.out)
    m = tax["meta"]
    print(f"  signature: {info['n_metrics']} metrics + {len(info['flags'])} flags -> "
          f"{info['n_components']} components")
    print(f"  classes: {m['n_classes']} density + {m['n_rule_based']} rule-based; "
          f"{m['n_assigned']}/{m['n_windows']} windows assigned")
    print(f"  Ward fusions 9->8 at {m['h_9to8']:.2f}, 8->7 at {m['h_8to7']:.2f}; "
          f"rules agree with the K=8 cut on {m['rules_vs_ward']}/{m['n_classes'] + m['n_rule_based']}")
    print(tax["archetypes"][["archetype_id", "archetype", "n_units", "n_windows", "n_events"]]
          .to_string(index=False))


def cmd_all(args) -> None:
    cmd_sampling(args)
    cmd_features(args)
    args.features = args.out / "window_features.csv"
    cmd_classify(args)


def cmd_check(args) -> None:
    ref = reference_dir(args.mission)
    failed = False
    for name in TABLES:
        p = args.dir / f"{name}.csv"
        if not p.exists():
            print(f"  {name}: not in {args.dir}")
            continue
        diffs = compare_tables(read_table(p), read_table(ref / f"{name}.csv"))
        failed |= bool(diffs)
        print(f"  {name}: {'matches the reference' if not diffs else '; '.join(diffs)}")
    sys.exit(1 if failed else 0)


def cmd_recall(args) -> None:
    from tsadbench.taxonomy.recall import archetype_recall

    runs = {}
    for spec in args.runs:
        name, _, path = spec.partition("=")
        if not path:
            sys.exit(f"expected NAME=RUN_DIR, got {spec}")
        runs[name] = Path(path)
    W = read_table(args.windows or reference_dir(args.mission) / "windows.csv")
    if args.channels:
        keep = {int(c.removeprefix("channel_")) for c in load_metadata()[args.mission][args.channels]}
        W = W[W.channel.isin(keep)].reset_index(drop=True)
    per_window, summary = archetype_recall(W, runs, args.mission)
    per_window.to_csv(args.out / "archetype_recall_windows.csv", index=False)
    summary.to_csv(args.out / "archetype_recall.csv", index=False)
    (args.out / "archetype_recall.json").write_text(
        json.dumps({k: str(v) for k, v in runs.items()}, indent=2) + "\n")
    print(summary.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", default="ESA-Mission1")
    ap.add_argument("--out", type=Path, default=project_root() / "runs" / "taxonomy")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("all").set_defaults(func=cmd_all)
    sub.add_parser("sampling").set_defaults(func=cmd_sampling)
    sub.add_parser("features").set_defaults(func=cmd_features)
    p = sub.add_parser("classify")
    p.add_argument("--features", type=Path)
    p.set_defaults(func=cmd_classify)
    p = sub.add_parser("check")
    p.add_argument("dir", type=Path, nargs="?")
    p.set_defaults(func=cmd_check)
    p = sub.add_parser("recall")
    p.add_argument("runs", nargs="+", metavar="NAME=RUN_DIR")
    p.add_argument("--windows", type=Path)
    p.add_argument("--channels", choices=["full", "light"])
    p.set_defaults(func=cmd_recall)
    args = ap.parse_args()
    if args.cmd == "check" and args.dir is None:
        args.dir = args.out
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    args.func(args)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
