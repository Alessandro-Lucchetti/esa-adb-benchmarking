import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl

from tsadbench.io.misc import project_root

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 9


# === From Matplotlib's "Annotated heatmap" example ===
from tsadbench.viz.heatmap import heatmap, annotate_heatmap
# === End helpers ===


def plot_sensitivity_heatmap(csv_file,
                             metric_source='metrics_all_events',  # or 'metrics_anomalies_alone'
                             value_key='f0.5',
                             fmt=".3f",
                             figsize=(8, 6),
                             outfile=None,
                             cmap='Greys',
                             slide_length_filter=5):
    """
    Reads a CSV where each row has a 'config' dict column.
    Extracts config[metric_source]['event_wise'][value_key] and plots an annotated heatmap per mission.
    """

    df = pd.read_csv(csv_file)

    # --- Expand config safely (new format) or ew_metrics (legacy) ---
    if 'config' in df.columns:
        cfg = df['config'].apply(ast.literal_eval)
        cfg_df = cfg.apply(pd.Series)

        # IMPORTANT: remove overlapping columns from the original df to avoid duplicates
        overlap = set(df.columns).intersection(set(cfg_df.columns))
        # We'll prefer values coming from config (cfg_df), so drop overlaps from original df
        df = df.drop(columns=list(overlap.union({'config'})), errors='ignore')
        # Recombine without duplicates
        df = pd.concat([df, cfg_df], axis=1)

        # Helper to extract float metric from nested dict (values may be strings)
        def extract_metric(d):
            if not isinstance(d, dict):
                return np.nan
            ew = d.get('event_wise')
            if not isinstance(ew, dict):
                return np.nan
            v = ew.get(value_key)
            try:
                return float(v)
            except (TypeError, ValueError):
                return np.nan

        # Work out which column holds the metric dict
        if metric_source in df.columns:
            metric_col = metric_source
        elif f"metrics_{metric_source}" in df.columns:
            metric_col = f"metrics_{metric_source}"
        else:
            metric_col = None

        if metric_col is None:
            # Legacy fallback if no metric in config
            if 'ew_metrics' in df.columns:
                parsed_legacy = df['ew_metrics'].apply(ast.literal_eval)
                metrics_df = parsed_legacy.apply(pd.Series)
                df = pd.concat([df.drop(columns=['ew_metrics']), metrics_df], axis=1)
                # Try mapping common legacy names
                guess = {'metrics_all_events': 'all_events',
                         'metrics_anomalies_alone': 'score'}.get(metric_source, metric_source)
                df[value_key] = df[guess].apply(
                    lambda d: np.nan if pd.isna(d) or not isinstance(d, dict) else d.get(value_key))
            else:
                raise KeyError(
                    f"Could not find metrics column '{metric_source}' (or 'metrics_{metric_source}') "
                    f"in config, and no legacy 'ew_metrics' present. Available columns: {list(df.columns)}"
                )
        else:
            df[value_key] = df[metric_col].apply(extract_metric)

    else:
        # Pure legacy path (no 'config' column)
        if 'ew_metrics' not in df.columns:
            raise KeyError("Neither 'config' nor 'ew_metrics' column found in the CSV.")
        parsed_legacy = df['ew_metrics'].apply(ast.literal_eval)
        metrics_df = parsed_legacy.apply(pd.Series)
        df = pd.concat([df.drop(columns=['ew_metrics']), metrics_df], axis=1)
        guess = {'metrics_all_events': 'all_events',
                 'metrics_anomalies_alone': 'score'}.get(metric_source, metric_source)
        df[value_key] = df[guess].apply(
            lambda d: np.nan if pd.isna(d) or not isinstance(d, dict) else d.get(value_key))

    # === Ensure expected columns exist and are numeric ===
    expected = ['mission', 'query_length', 'reference_length']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns from expanded config: {missing}")

    df['query_length'] = pd.to_numeric(df['query_length'], errors='coerce')
    df['reference_length'] = pd.to_numeric(df['reference_length'], errors='coerce')
    if 'slide_length' in df.columns:
        df['slide_length'] = pd.to_numeric(df['slide_length'], errors='coerce')

    # Optional filter
    if slide_length_filter is not None and 'slide_length' in df.columns:
        df = df[df['slide_length'] == slide_length_filter].copy()

    # === Plot per mission ===
    for mission, group in df.groupby('mission'):
        pivot = (group
                 .pivot(index='query_length', columns='reference_length', values=value_key)
                 .sort_index()
                 .sort_index(axis=1))

        data = pivot.values.astype(float)
        data_ma = np.ma.masked_invalid(data)

        fig, ax = plt.subplots(figsize=figsize)

        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(alpha=0.0)

        im, cbar = heatmap(
            data_ma,
            row_labels=pivot.index.values,
            col_labels=pivot.columns.values,
            ax=ax,
            cbarlabel=r'$\mathrm{F}_{0.5}$' if value_key.lower() in ['f0.5', 'f05', 'f0_5'] else value_key,
            cmap=cmap_obj,
            vmin=0,
            vmax=1,
            aspect='auto',
            interpolation='nearest'
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlabel(r'$W_\mathrm{R}$')
        ax.set_ylabel(r'$W_\mathrm{Q}$')
        ax.set_aspect('equal')

        annotate_heatmap(im, data=data_ma, valfmt="{x:" + fmt + "}",
                         textcolors=("black", "white"))

        fig.tight_layout()

        if outfile is not None:
            fig.savefig(f"{outfile}_{mission}.pdf")

    plt.show()




if __name__ == "__main__":
    coarse = "exp_2025-10-21_12.26.18.csv"
    fine = "exp_2025-10-21_15.39.14.csv"

    plot_sensitivity_heatmap(
        csv_file=project_root() / "smed" / "experiments" / coarse,
        metric_source="metrics_all_events",      # or "metrics_anomalies_alone"
        value_key="f0.5",
        fmt=".3f",
        figsize=(8/2.54, 8/2.54),
        outfile=(project_root() / "figures" / "smed_sensitivity_light_resampled_coarse")
    )

    plot_sensitivity_heatmap(
        csv_file=project_root() / "smed" / "experiments" / fine,
        metric_source="metrics_all_events",      # or "metrics_anomalies_alone"
        value_key="f0.5",
        fmt=".3f",
        figsize=(8/2.54, 8/2.54),
        outfile=(project_root() / "figures" / "smed_sensitivity_light_resampled_fine")
    )
