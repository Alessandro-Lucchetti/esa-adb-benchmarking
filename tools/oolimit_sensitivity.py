"""Sensitivity of the robust out-of-limit (OOL) baseline to the envelope percentile p.

Collects every `runs/oolimit[_full]/oolimit_*_p<p>/config.json` written by
scripts/limits_psweep.py and renders a 1-D annotated heatmap (rows: lightweight /
full set, columns: p) of the corrected event-wise F0.5 (all events), in the same
style as the SMED / Q-GPR sensitivity figures. Also writes the underlying table.

With --smed the SMED p sweep at the nominal window sizes (scripts/smed_psweep.py,
runs/smed_psweep[_full]/) is added as two more rows, plus a line-plot version of
the same comparison (F0.5 vs p, log-x), so the two methods' p dependence can be
read side by side.

p is shown as a fraction (0.001 = 0.1-th percentile), the convention used for SMED
in the paper; run names carry the percentile value. Only phase-5 runs are used
(the training-phase sweep lives in tools/oolimit_phase_sensitivity.py).

Usage:
    python tools/oolimit_sensitivity.py [--no-tex] [--smed] [--ps 0.1 1 3 5 10 25]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from tsadbench.io.misc import project_root
from tsadbench.viz.heatmap import heatmap, annotate_heatmap

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 9

ROOTS = {"oolimit": {"light": "oolimit", "full": "oolimit_full"},
         "smed": {"light": "smed_psweep", "full": "smed_psweep_full"}}
LABELS = {"oolimit": "OOL", "smed": "SMED"}
DEFAULT_PS = [0.1, 1.0, 3.0, 5.0, 10.0, 25.0]   # the sweep grid of scripts/limits_psweep.py


def collect(ps: list[float] | None = None, k: float = 1.5, algos: tuple[str, ...] = ("oolimit",),
            phase: str = "5") -> pd.DataFrame:
    rows = []
    for algo in algos:
        for set_name, folder in ROOTS[algo].items():
            root = project_root() / "runs" / folder
            for cfg_file in sorted(root.glob("*/config.json")):
                cfg = json.loads(cfg_file.read_text())
                if float(cfg["k"]) != k or str(cfg["phase"]) != phase or "metrics_all_events" not in cfg:
                    continue
                if ps is not None and float(cfg["p"]) not in ps:
                    continue
                ew = cfg["metrics_all_events"]["event_wise"]
                ev_file = cfg_file.parent / "labels" / "event.parquet"
                rows.append({"algo": algo, "set": set_name, "p_percentile": float(cfg["p"]),
                             "p_fraction": float(cfg["p"]) / 100.0,
                             "precision": float(ew["precision"]), "recall": float(ew["recall"]),
                             "f0.5": float(ew["f0.5"]),
                             "alarming_precision": float(cfg["metrics_all_events"]["alarming_precision"]),
                             "n_predicted_events": len(pd.read_parquet(ev_file)) if ev_file.exists() else np.nan})
    df = pd.DataFrame(rows)
    df["algo"] = pd.Categorical(df["algo"], categories=list(algos), ordered=True)
    df = df.sort_values(["algo", "set", "p_percentile"], ascending=[True, False, True])
    return df.reset_index(drop=True)


def plot(df: pd.DataFrame, outfile: Path, fmt: str = ".3f") -> None:
    """Annotated heatmap, one row per (algo, set), one column per p."""
    algos = list(df["algo"].cat.categories) if hasattr(df["algo"], "cat") else list(dict.fromkeys(df["algo"]))
    df = df.assign(row=[f"{LABELS[a]} {'lightweight' if s == 'light' else 'full'}"
                        for a, s in zip(df["algo"], df["set"])])
    order = [f"{LABELS[a]} {s}" for a in algos for s in ("lightweight", "full")]
    pivot = (df.pivot(index="row", columns="p_fraction", values="f0.5")
               .reindex([r for r in order if r in set(df["row"])]).sort_index(axis=1))
    data = np.ma.masked_invalid(pivot.values.astype(float))

    c_orange = np.array([230, 159, 0]) / 255
    cmap = LinearSegmentedColormap.from_list("okabe_ito_sequential", [[1.0, 0.95, 0.85], c_orange], N=256)
    cmap.set_bad(alpha=0.0)

    n_rows, n_cols = data.shape
    fig, ax = plt.subplots(figsize=((1.5 * n_cols + 4) / 2.54, (2.2 + 0.7 * n_rows) / 2.54))
    im, cbar = heatmap(data, row_labels=list(pivot.index),
                       col_labels=[f"{p:g}" for p in pivot.columns],
                       ax=ax, cbarlabel=r"$\mathrm{F}_{0.5}$", cmap=cmap,
                       vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel(r"$p$")
    annotate_heatmap(im, data=data, valfmt="{x:" + fmt + "}", textcolors=("black", "white"), fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{outfile}.pdf")
    fig.savefig(f"{outfile}.png", dpi=300)
    plt.close(fig)


def plot_lines(df: pd.DataFrame, outfile: Path) -> None:
    """F0.5 vs p (log-x), one line per (algo, set); Okabe-Ito colours, set = line style."""
    colours = {"oolimit": np.array([230, 159, 0]) / 255, "smed": np.array([0, 114, 178]) / 255}
    styles = {"light": ("-", "o", "lightweight"), "full": ("--", "s", "full")}
    fig, ax = plt.subplots(figsize=(8.5 / 2.54, 5.5 / 2.54))
    for (algo, set_name), g in df.groupby(["algo", "set"], observed=True, sort=False):
        g = g.sort_values("p_fraction")
        ls, marker, set_label = styles[set_name]
        ax.plot(g["p_fraction"], g["f0.5"], ls, marker=marker, ms=3.5, lw=1.2,
                color=colours[algo], label=f"{LABELS[algo]} {set_label}")
    ax.set_xscale("log")
    ax.set_xticks(sorted(df["p_fraction"].unique()))
    ax.set_xticklabels([f"{p:g}" for p in sorted(df["p_fraction"].unique())],
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$\mathrm{F}_{0.5}$")
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(f"{outfile}.pdf")
    fig.savefig(f"{outfile}.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tex", action="store_true", help="render with mathtext instead of LaTeX")
    ap.add_argument("--smed", action="store_true", help="add the SMED p sweep rows (runs/smed_psweep[_full])")
    ap.add_argument("--ps", nargs="*", type=float, default=DEFAULT_PS,
                    help="p values (percentile) to include; pass --ps alone for every run on disk")
    args = ap.parse_args()
    mpl.rcParams['text.usetex'] = not args.no_tex

    algos = ("oolimit", "smed") if args.smed else ("oolimit",)
    table = collect(args.ps or None, algos=algos)
    stem = "oolimit_vs_smed_sensitivity_p_ESA-Mission1" if args.smed else "oolimit_sensitivity_p_ESA-Mission1"
    out = project_root() / "figures" / stem
    out.parent.mkdir(exist_ok=True)
    table.to_csv(f"{out}.csv", index=False)
    print(table.to_string(index=False))
    plot(table, out)
    print(f"\nwrote {out}.csv / .pdf / .png")
    if args.smed:
        plot_lines(table, out.with_name(stem + "_lines"))
        print(f"wrote {out.with_name(stem + '_lines')}.pdf / .png")
