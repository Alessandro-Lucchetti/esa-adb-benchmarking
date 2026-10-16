"""Per-channel sampling statistics of the raw telemetry and the eight sampling families.

For each target channel, over the full mission span: median and dispersion of the
inter-sample interval dt (whole seconds), the three dominant sampling regimes, data gaps
(dt above ``GAP_FACTOR`` x the median), and a distribution-shift index, the fraction of
adjacent-month pairs whose two-sample Kolmogorov-Smirnov statistic exceeds ``KS_THR``
(months with fewer than 100 samples break the chain; larger months are subsampled to
5000 values with a fixed seed).

The families group channels with near-identical sampling behaviour. Their membership
was set by inspection of the per-channel table and is fixed here; ``family_table``
summarizes the statistics of each family.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from tsadbench.io.misc import load_metadata
from tsadbench.taxonomy.windows import data_dir

KS_THR = 0.2
GAP_FACTOR = 10
SAMPLING_FAMILIES = {
    "S1": (41, 42, 43, 44, 45, 46),
    "S2": (47, 48, 49, 57, 58, 59, 60),
    "S3": (70, 71, 72, 73, 74, 75, 76),
    "S4": (12, 13, 17, 18, 19, 20, 25, 26, 27, 28, 34, 35, 36, 37),
    "S5": (14, 15, 16, 21, 22, 23, 24, 29, 30, 31, 32, 33, 38, 39, 40),
    "S6": (50, 51, 52),
    "S7": (64, 65, 66),
    "S8": (61, 62, 63),
}
FAMILY_OF = {ch: fam for fam, chs in SAMPLING_FAMILIES.items() for ch in chs}


def channel_stats(ts: np.ndarray, values: np.ndarray, ks_thr: float = KS_THR,
                  gap_factor: int = GAP_FACTOR) -> dict | None:
    """Sampling statistics of one channel (sorted, de-duplicated datetime64[ns] samples)."""
    dt = np.diff(ts).astype("timedelta64[s]").astype(float)
    if len(dt) == 0:
        return None
    med = float(np.median(dt))
    if med <= 0:  # sub-second bursts dominate: use the base cadence between them
        med = float(np.median(dt[dt > 0])) if (dt > 0).any() else 1.0
    span = float((ts[-1] - ts[0]).astype("timedelta64[s]").astype(float))
    gaps = dt[dt > gap_factor * med]
    vals, counts = np.unique(np.round(dt), return_counts=True)
    top = np.argsort(counts)[::-1][:3]
    regimes = "; ".join(f"{vals[i]:.0f}s ({100 * counts[i] / len(dt):.1f}%)" for i in top)

    months = ts.astype("datetime64[M]")
    _, starts = np.unique(months, return_index=True)
    bounds = np.append(starts, len(ts))
    n_shift = n_pairs = 0
    rng = np.random.default_rng(0)
    prev = None
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sel = values[lo:hi]
        if len(sel) < 100:
            prev = None
            continue
        if len(sel) > 5000:
            sel = rng.choice(sel, 5000, replace=False)
        if prev is not None:
            n_pairs += 1
            n_shift += ks_2samp(prev, sel).statistic > ks_thr
        prev = sel

    return dict(n_samples=len(values), span_years=span / (365.25 * 86400),
                med_dt_s=med, iqr_dt_s=float(np.subtract(*np.percentile(dt, [75, 25]))),
                cv_dt=float(dt.std() / dt.mean()), p95_over_med=float(np.percentile(dt, 95) / med),
                regimes=regimes, n_gaps=len(gaps), gap_total_days=float(gaps.sum() / 86400),
                gap_pct=float(100 * gaps.sum() / span), max_gap_h=float(dt.max() / 3600),
                shift_pairs=n_pairs, shift_frac=float(n_shift / n_pairs) if n_pairs else np.nan)


def sampling_table(mission: str = "ESA-Mission1", channels: list[int] | None = None,
                   verbose: bool = True) -> pd.DataFrame:
    """One row of sampling statistics per target channel (or per channel of ``channels``),
    with its sampling family and the count of duplicated timestamps that were dropped."""
    if channels is None:
        channels = sorted(int(c.removeprefix("channel_")) for c in load_metadata()[mission]["full"])
    rows = []
    for ch in channels:
        col = f"channel_{ch}"
        df = pd.read_parquet(data_dir(mission) / "raw" / f"{col}.parquet")
        df = df.dropna(subset=[col]).sort_values("timestamp", kind="stable")
        dup = df["timestamp"].duplicated(keep=False)
        n_dup_ts = int(df["timestamp"].duplicated().sum())
        n_dup_conflict = int((df[dup].groupby("timestamp")[col].nunique() > 1).sum()) if n_dup_ts else 0
        df = df.drop_duplicates(subset="timestamp", keep="first")
        s = channel_stats(df["timestamp"].to_numpy("datetime64[ns]"), df[col].to_numpy(np.float64))
        if s is None:
            continue
        rows.append(dict(channel=ch, family=FAMILY_OF.get(ch, ""), n_dup_ts=n_dup_ts,
                         n_dup_conflict=n_dup_conflict, **s))
        if verbose:
            print(f"  {col}: {s['n_samples']:,} samples, median dt {s['med_dt_s']:.0f} s, "
                  f"shift {s['shift_frac']:.2f}", flush=True)
    return pd.DataFrame(rows)


def family_table(stats: pd.DataFrame) -> pd.DataFrame:
    """Ranges of the per-channel statistics within each sampling family."""
    rows = []
    for fam, chs in SAMPLING_FAMILIES.items():
        s = stats[stats["channel"].isin(chs)]
        if s.empty:
            continue
        rows.append(dict(
            family=fam, channels=" ".join(str(c) for c in chs), n_channels=len(s),
            med_dt_s_min=s.med_dt_s.min(), med_dt_s_max=s.med_dt_s.max(),
            cv_dt_min=s.cv_dt.min(), cv_dt_max=s.cv_dt.max(),
            regimes=s.regimes.iloc[0],
            gap_pct_min=s.gap_pct.min(), gap_pct_max=s.gap_pct.max(), max_gap_h=s.max_gap_h.max(),
            shift_frac_min=s.shift_frac.min(), shift_frac_max=s.shift_frac.max()))
    return pd.DataFrame(rows)
