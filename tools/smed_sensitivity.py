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


def plot_sensitivity_heatmap(csv_file, metric='score', value_key='f0.5',
                             fmt=".3f", figsize=(8, 6), outfile=None,
                             slide_length_filter=5):
    """
    Reads the sensitivity analysis CSV, extracts ew_metrics[metric][value_key],
    and plots an annotated heatmap for each mission using Matplotlib's
    example helpers (heatmap / annotate_heatmap).

    Parameters:
    - csv_file: path to CSV
    - metric: key in the ew_metrics dict to visualize (e.g., 'score' or 'all_events')
    - value_key: nested key inside the metric dict to plot (e.g., 'f0.5')
    - fmt: string/format specifier for annotations (e.g., '.3f')
    - figsize: figure size per mission (tuple)
    - outfile:
    - slide_length_filter: keep rows with slide_length == this value (set to None to disable)
    """
    from matplotlib.colors import LinearSegmentedColormap

    df = pd.read_csv(csv_file)
    if slide_length_filter is not None and 'slide_length' in df.columns:
        df = df[df['slide_length'] == slide_length_filter].copy()

    # Check if ew_metrics is empty and fall back to config
    if 'ew_metrics' in df.columns and df['ew_metrics'][0] != "{}":
        # Expand ew_metrics safely
        parsed = df['ew_metrics'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else {})
        metrics_df = parsed.apply(pd.Series)
        df = pd.concat([df.drop(columns=['ew_metrics']), metrics_df], axis=1)
    elif 'config' in df.columns:
        # Fall back to config column which contains the metrics
        parsed_configs = df['config'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else {})

        # Extract only the metrics fields we need
        metrics_keys = ['metrics_all_events', 'metrics_anomalies_alone']
        for key in metrics_keys:
            if key in parsed_configs.iloc[0]:
                df[key] = parsed_configs.apply(lambda config: config.get(key, {}).get("event_wise"))
    else:
        raise ValueError("Neither 'ew_metrics' nor 'config' column found in CSV")

    # Extract nested metric value
    df[value_key] = df[metric].apply(
        lambda d: None if pd.isna(d) else (d.get(value_key) if isinstance(d, dict) else None))

    # Define Okabe-Ito palette colors
    c_blue = np.array([0, 114, 178]) / 255
    c_orange = np.array([230, 159, 0]) / 255

    # Create custom colormap: shades of blue (light blue for low, dark blue for high)
    # Option 1: Shades of blue (uncomment to use)
    # colors = [[0.8, 0.9, 1.0], c_blue]  # Light blue -> Dark blue

    # Option 2: Shades of orange (uncomment to use, comment Option 1)
    colors = [[1.0, 0.95, 0.85], c_orange]  # Light orange -> Dark orange

    n_bins = 256
    cmap_custom = LinearSegmentedColormap.from_list('okabe_ito_sequential', colors, N=n_bins)

    # Plot per mission
    for mission, group in df.groupby('mission'):
        pivot = (group
                 .pivot(index='query_length', columns='reference_length', values=value_key)
                 .sort_index()
                 .sort_index(axis=1))

        # Prepare data as masked array to handle NaNs
        data = pivot.values.astype(float)
        data_ma = np.ma.masked_invalid(data)

        fig, ax = plt.subplots(figsize=figsize)

        # Ensure NaNs are transparent
        cmap_obj = cmap_custom.copy()
        cmap_obj.set_bad(alpha=0.0)

        im, cbar = heatmap(
            data_ma,
            row_labels=pivot.index.values,
            col_labels=pivot.columns.values,
            ax=ax,
            cbarlabel=r'$\mathrm{F}_{0.5}$',
            cmap=cmap_obj,
            vmin=0,
            vmax=1,
            aspect='auto',
            interpolation='nearest'
        )

        # Rotate x tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlabel(r'$W_\mathrm{R}$')
        ax.set_ylabel(r'$W_\mathrm{Q}$')
        ax.set_aspect('equal')

        # Annotate cells
        annotate_heatmap(im, data=data_ma, valfmt="{x:" + fmt + "}",
                         textcolors=("black", "white"))

        fig.tight_layout()

        if outfile is not None:
            fig.savefig(f"{outfile}.pdf")

    plt.show()


if __name__ == "__main__":
    smed_sensitivity_full_raw_coarse_ESAMission1 = "exp_2025-09-24_16.51.32.csv"
    smed_sensitivity_full_raw_fine_ESAMission1 = "exp_2025-09-25_12.59.25.csv"

    smed_sensitivity_light_raw_coarse_ESAMission1 = "exp_2025-11-12_12.20.40.csv"
    smed_sensitivity_light_raw_fine_ESAMission1 = "exp_2025-11-12_11.10.04.csv"


    plot_sensitivity_heatmap(
        csv_file=project_root() / "smed" / "experiments" / smed_sensitivity_full_raw_coarse_ESAMission1,
        metric="all_events",  # or "score"
        value_key="f0.5",
        fmt=".3f",
        figsize=(8/2.54, 8/2.54),
        outfile = (project_root() / "figures" / "smed_sensitivity_full_raw_coarse_ESA-Mission1")
    )

    plot_sensitivity_heatmap(
        csv_file=project_root() / "smed" / "experiments" / smed_sensitivity_full_raw_fine_ESAMission1,
        metric="all_events",  # or "score"
        value_key="f0.5",
        fmt=".3f",
        figsize=(8/2.54, 8/2.54),
        outfile = (project_root() / "figures" / "smed_sensitivity_full_raw_fine_ESA-Mission1")
    )

    plot_sensitivity_heatmap(
        csv_file=project_root() / "smed" / "experiments" / smed_sensitivity_light_raw_coarse_ESAMission1,
        metric="metrics_all_events",  # or "score"
        value_key="f0.5",
        fmt=".3f",
        figsize=(8/2.54, 8/2.54),
        outfile = (project_root() / "figures" / "smed_sensitivity_light_raw_coarse_ESA-Mission1")
    )

    plot_sensitivity_heatmap(
        csv_file=project_root() / "smed" / "experiments" / smed_sensitivity_light_raw_fine_ESAMission1,
        metric="metrics_all_events",  # or "score"
        value_key="f0.5",
        fmt=".3f",
        figsize=(8/2.54, 8/2.54),
        outfile = (project_root() / "figures" / "smed_sensitivity_light_raw_fine_ESA-Mission1")
    )



