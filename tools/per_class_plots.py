import time
from datetime import timedelta
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.gridspec as mgs
import matplotlib.dates as m_dates
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from absl import app
from screeninfo import get_monitors

from tsadbench.io.misc import load_gt_events, project_root, add_event_bands, load_metadata


def matplotlib_results(
        anomaly_class: str,
        mission: str = "ESA-Mission1",
        t0: float | None = None,
        *,
        h_signal: float = 2,
        h_gap: float = 1,
        dpi: int = 40,
) -> Figure:

    t0 = t0 or time.time()

    print(f"[{(time.time() - t0):6.2f}s] Generating figure for {anomaly_class}")

    data_path = project_root() / "data" / mission
    channel_path = data_path / "raw"
    labels_path = data_path / "labels.csv"
    anomaly_path = data_path / "anomaly_types.csv"
    metadata = load_metadata()

    channels = [ch for ch in metadata[mission]["full"]]

    gt_df = load_gt_events(labels_path, channels)
    typ_df = pd.read_csv(anomaly_path)

    class_ids = typ_df[typ_df["Class"] == anomaly_class]["ID"].tolist()
    gt_df = gt_df[gt_df["ID"].isin(class_ids)]

    start = gt_df["StartTime"].min()
    end = gt_df["EndTime"].max()

    win_start = start - timedelta(weeks=1)
    win_end = end + timedelta(weeks=1)

    channels = sorted(gt_df["Channel"].unique().tolist())
    n_channels = len(channels)

    anomalies = gt_df[gt_df.ID.isin(typ_df.loc[typ_df.Category == "Anomaly", "ID"])]
    rare = gt_df[gt_df.ID.isin(typ_df.loc[typ_df.Category == "Rare Event", "ID"])]

    hr = []
    for _ in range(n_channels):
        hr.extend([h_signal])
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

    for ch in tqdm(channels,
                   desc=f"[{(time.time() - t0):6.2f}s] Building channels",
                   unit="ch"):

        if master_x_axis is None:
            ax_sig = fig.add_subplot(gs[row_idx])
            ax_sig.set_xlim(m_dates.date2num(win_start), m_dates.date2num(win_end))
            master_x_axis = ax_sig
        else:
            ax_sig = fig.add_subplot(gs[row_idx], sharex=master_x_axis)

        signal_df = pd.read_parquet(channel_path / f"{ch}.parquet")
        mask = signal_df["timestamp"].between(win_start, win_end, inclusive="both")
        ax_sig.plot(signal_df.loc[mask, "timestamp"], signal_df.loc[mask, ch])
        ax_sig.relim()
        add_event_bands(ax_sig, anomalies, ch, "red")
        add_event_bands(ax_sig, rare, ch, "orange")

        ax_sig.set_title(ch)
        ax_sig.get_yaxis().set_visible(False)
        ax_sig.autoscale_view(scaley=True, tight=False)

        row_idx += 2

    print(f"[{(time.time() - t0):6.2f}s] Figure built for {anomaly_class}")

    return fig


def open_scrollable_window(
        figs: dict[str, Figure],
        t0: float | None = None,
        *,
        title: str = "Interactive results viewer"):
    t0 = t0 or time.time()

    root = tk.Tk()

    m = next(iter(get_monitors()))
    for monitor in get_monitors():
        if monitor.is_primary:
            m = monitor

    dpi_monitor = m.width / m.width_mm * 25.4
    reference_fig = next(iter(figs.values()))
    dpi_fig = reference_fig.get_dpi()
    scale = dpi_monitor / dpi_fig
    root.tk.call('tk', 'scaling', scale * 1.3)
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

def main(_):
    """Entry point: ask for a predictions parquet and launch the viewer."""
    t0 = time.time()

    figs = {}

    # anomaly_types_path = project_root() / "data" / mission / "anomaly_types.csv"
    # anomaly_types_df = pd.read_csv(anomaly_types_path)
    # for anomaly_class in sorted(anomaly_types_df["Class"].unique()):

    anomaly_class = "class_13"

    fig = matplotlib_results(anomaly_class, mission="ESA-Mission1", t0=t0)
    figs[anomaly_class] = fig

    open_scrollable_window(figs, t0)

if __name__ == "__main__":
    app.run(main)
