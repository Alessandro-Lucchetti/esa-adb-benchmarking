import time
import pandas as pd
from tqdm import tqdm
from pandas import Timestamp
from typing import Mapping, Sequence, Optional, Any
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.gridspec as mgs
import matplotlib.patches as m_patches
import matplotlib.axes as m_axes
import matplotlib.dates as m_dates
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from collections import OrderedDict
import tkinter as tk
from tkinter import ttk
from absl import app
from screeninfo import get_monitors

from tsadbench.io.misc import load_gt_events, project_root, select_prediction, load_predictions_dict, load_json, add_event_bands


def matplotlib_results(
    config: Mapping[str, Any],
    predictions_dict: Mapping[str, Sequence[tuple[pd.Timestamp, float]]],
    scores_thresholds_dict: Optional[Mapping[str, Mapping[str, Sequence[float]]]] = None,
    subsystem: str = "Subsystem",
    t0: float | None = None,
    *,
    h_signal: float = 2,
    h_pred:   float = 0.4,
    h_score:  float = 1.2,
    h_gap:    float = 1,
    dpi:      int   = 40,
) -> Figure:

    t0 = t0 or time.time()

    print(f"[{(time.time() - t0):6.2f}s] Generating figure for {subsystem.replace('_',' ').capitalize()}")

    channels = list(predictions_dict)
    n_channels = len(channels)

    per_ch_rows = 3 if scores_thresholds_dict is not None else 2

    mission = config["mission"]
    data_path =project_root() / "data" / mission
    channel_path = data_path / "raw"
    labels_path = data_path / "labels.csv"
    anomaly_path = data_path / "anomaly_types.csv"
    metadata = load_json(project_root() / "data" / "metadata.json")

    start = Timestamp(metadata[mission]["test_period"][0])
    end = Timestamp(metadata[mission]["test_period"][1])

    gt_df  = load_gt_events(labels_path, channels, start, end)
    typ_df = pd.read_csv(anomaly_path)
    anomalies = gt_df[gt_df.ID.isin(typ_df.loc[typ_df.Category == "Anomaly", "ID"])]
    rare = gt_df[gt_df.ID.isin(typ_df.loc[typ_df.Category == "Rare Event", "ID"])]

    hr = []
    for _ in range(n_channels):
        hr.extend([h_signal, h_pred] if per_ch_rows == 2 else [h_signal, h_pred, h_score])
        hr.append(h_gap)
    hr = hr[:-1]

    m = next(iter(get_monitors()))
    for monitor in get_monitors():
        if monitor.is_primary:
            m = monitor
    width = m.width_mm / 25.4

    fig = plt.figure(figsize=(width, sum(hr)), dpi=dpi)

    gs = mgs.GridSpec(
        nrows=len(hr), ncols=1, figure=fig,
        height_ratios=hr, hspace=0.0
    )

    row_idx = 0
    master_x_axis = None
    legend_items = OrderedDict()

    for ch in tqdm(channels,
                   desc=f"[{(time.time() - t0):6.2f}s] Building channels",
                   unit="ch"):
    # Adding signal

        if master_x_axis is None:
            ax_sig = fig.add_subplot(gs[row_idx])
            ax_sig.set_xlim(m_dates.date2num(start), m_dates.date2num(end))
            master_x_axis = ax_sig
        else:
            ax_sig = fig.add_subplot(gs[row_idx], sharex=master_x_axis)

        signal_df = pd.read_parquet(channel_path / f"{ch}.parquet")

        sig_line, = ax_sig.plot(signal_df["timestamp"], signal_df[ch], label="signal")
        add_event_bands(ax_sig, anomalies, ch, "red")
        add_event_bands(ax_sig, rare, ch, "orange")

        ax_sig.spines['bottom'].set_visible(False)
        ax_sig.autoscale(enable=True, axis='y', tight=False)
        ax_sig.set_title(ch)
        ax_sig.get_xaxis().set_visible(False)
        ax_sig.get_yaxis().set_visible(False)
        row_idx += 1

        # Adding predictions

        ax_pred = fig.add_subplot(gs[row_idx], sharex=master_x_axis)

        timestamps = pd.DatetimeIndex([t for t, _ in predictions_dict[ch]])
        predictions = [lbl for _, lbl in predictions_dict[ch]]

        pred_line, = ax_pred.plot(timestamps, predictions, color="black", label="prediction")
        add_event_bands(ax_pred, anomalies, ch, "red")
        add_event_bands(ax_pred, rare, ch, "orange")
        ax_pred.spines['top'].set_visible(False)
        ax_pred.set_yticks([0, 1])
        ax_pred.set_yticklabels(["0", "1"], color="red")
        ax_pred.grid(visible=True, axis='y', linestyle='dotted', color='red')
        ax_sig.autoscale(enable=True, axis='y', tight=False)

        row_idx += 1

        score_line : Line2D | None = None
        thr_line: Line2D | None = None

        # Adding (optional) scores and thresholds

        if scores_thresholds_dict is not None:
            ax_pred.spines['bottom'].set_visible(False)
            ax_pred.get_xaxis().set_visible(False)

            ax_score = fig.add_subplot(gs[row_idx], sharex=master_x_axis)

            timestamps, scores, thresholds = zip(*scores_thresholds_dict[ch])

            score_line, = ax_score.plot(timestamps, scores,  label="Score",     color="tab:green")
            thr_line, = ax_score.plot(timestamps, thresholds, label="Threshold", color="tab:purple")
            add_event_bands(ax_score, anomalies, ch, "red")
            add_event_bands(ax_score, rare, ch, "orange")

            ax_sig.autoscale(enable=True, axis='y', tight=False)
            ax_score.spines['top'].set_visible(False)
            ax_score.get_yaxis().set_visible(False)

            row_idx += 1

        if row_idx < len(hr):
            row_idx += 1

        for handle, label in [(sig_line, "signal"), (pred_line, "prediction")]:
            if label not in legend_items:
                legend_items[label] = handle
        if scores_thresholds_dict is not None:
            for handle, label in [(score_line, "score"), (thr_line,   "threshold")]:
                if label not in legend_items:
                    legend_items[label] = handle

    if "anomaly" not in legend_items:
        legend_items["anomaly"] = m_patches.Patch(facecolor="red",
                                                 edgecolor="darkred",
                                                 alpha=0.30,
                                                 label="anomaly")
    if "rare event" not in legend_items:
        legend_items["rare event"] = m_patches.Patch(facecolor="orange",
                                                    edgecolor="darkorange",
                                                    alpha=0.30,
                                                    label="rare event")

    plt.tight_layout()
    fig_width_in, fig_height_in = fig.get_size_inches()
    fig.set_size_inches(fig_width_in, fig_height_in + 0.5)
    legend_frac = 0.4 / fig_height_in

    fig.subplots_adjust(top=1.0 - legend_frac)

    handles = list(legend_items.values())
    labels = list(legend_items.keys())

    fig.legend(handles, labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 1),
               ncol=len(handles),
               frameon=False,
               borderaxespad=0.0)

    print(f"[{(time.time() - t0):6.2f}s] Figure built for {subsystem.replace('_',' ').capitalize()}")

    return fig


def open_scrollable_window(
        figs: Mapping[str, Figure],
        t0: float | None = None,
        *,
        title: str ="Interactive results viewer"):

    t0 = t0 or time.time()

    root = tk.Tk()

    m = next(iter(get_monitors()))
    for monitor in get_monitors():
        if monitor.is_primary:
            m = monitor

    dpi_monitor = m.width/m.width_mm * 25.4
    reference_fig = next(iter(figs.values()))
    dpi_fig   = reference_fig.get_dpi()
    scale = dpi_monitor / dpi_fig
    root.tk.call('tk', 'scaling', scale*1.3)
    # root.attributes("-topmost", True)
    root.state("zoomed")
    root.title(title)

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)

    for name, fig in figs.items():
        label = name.replace('_', ' ').capitalize()
        print(f"[{time.time() - t0:6.2f}s] Adding tab for {label}")

        tab = ttk.Frame(nb)
        nb.add(tab, text=label)

        toolbar_frame = ttk.Frame(tab)
        toolbar_frame.pack(side="top", fill="x")

        canvas_frame = ttk.Frame(tab)
        canvas_frame.pack(side="top", fill="both", expand=True)

        canvas = tk.Canvas(canvas_frame, borderwidth=0)
        vsb = tk.Scrollbar(canvas_frame, orient="vertical",
                           command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")

        m_canvas = FigureCanvasTkAgg(fig, master=inner)
        m_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(m_canvas, toolbar_frame,
                                       pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="top", fill="x")

        inner.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

        def _wheel(event, c=canvas):
            c.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _wheel, add="+")
        canvas.bind_all("<Button-4>", _wheel, add="+")  # X11
        canvas.bind_all("<Button-5>", _wheel, add="+")

    print(f"[{time.time() - t0:6.2f}s] Opening window")
    root.mainloop()
    print(f"[{time.time() - t0:6.2f}s] Done")


def plot_results(
    config: Mapping[str, Any],
    predictions_dict: Mapping[str, Sequence[tuple[pd.Timestamp, float]]],
    scores_thresholds_dict: Optional[Mapping[str, Mapping[str, Sequence[float]]]] = None,
    t0: float | None = None
):

    t0 = t0 or time.time()
    
    figs = {}
    channels_path = project_root() / "data" / config["mission"] / "channels.csv"
    channel_df = pd.read_csv(channels_path)
    subsystems = channel_df.groupby("Subsystem")

    for name, group in subsystems:
        channels = group.loc[group["Channel"].isin(predictions_dict), "Channel"]
        sub_predictions_dict = {k: predictions_dict[k] for k in channels}
        sub_scores_thresholds_dict = {k: scores_thresholds_dict[k] for k in channels} if scores_thresholds_dict else None

        if sub_predictions_dict:
            fig = matplotlib_results(config=config,
                                       predictions_dict=sub_predictions_dict,
                                       scores_thresholds_dict=sub_scores_thresholds_dict,
                                       subsystem=str(name),
                                       t0=t0)
            figs[name] = fig

    open_scrollable_window(figs, t0)


def main(_):
    """Entry point: ask for a predictions parquet and launch the viewer."""
    pred_path = select_prediction()
    t0 = time.time()
    config = load_json(pred_path / "config")
    predictions_dict = load_predictions_dict(pred_path, t0=t0)

    scores_threshold_folder = pred_path / "scores"
    scores_thresholds_dict = {}

    if scores_threshold_folder.exists():
        print(f"[{(time.time() - t0):6.2f}s] Loading scores info")
        for parquet_file in scores_threshold_folder.glob("*.parquet"):
            ch = parquet_file.stem
            df= pd.read_parquet(parquet_file)
            scores_thresholds_dict[ch] =  list(zip(df["timestamp"], df["score"].values, df["threshold"].values))


    plot_results(config, predictions_dict, scores_thresholds_dict, t0=t0)

if __name__ == "__main__":
    app.run(main)
