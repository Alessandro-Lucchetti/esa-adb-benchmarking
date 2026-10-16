"""Morphological features of each labelled window.

Values are expressed on a robust ruler taken from the clean pre-event context (samples
inside other labelled intervals removed): median as the level, IQR as the unit (with MAD
and standard-deviation fallbacks). Families of features:

    magnitude     mag_z, peak_z, peak_z99, sgn_z, frac_pos
    persistence   end_shift_z (post level), ret_frac (fraction of the excursion recovered)
    shape         trend_r2, col_frac, settle_z, t_peak_frac, width_frac, imp_ratio, kurt, skew
    variability   var_ratio, jag, dz_frac_ev, dz_frac_pre
    frequency     e1/e4/e16 lag-difference energies at 1, 4, 16 x the context cadence,
                  their ratios to the context, and tilt_ev
    sampling      med_dt_ev, med_dt_pre, rate_ratio, cv_dt_ev, spread_dt_ev, gap_frac
    frozen data   zero_frac, const_frac, mode_frac, mode_z
"""
import numpy as np
import pandas as pd

from tsadbench.taxonomy.windows import EVENT, POST, PRE, load_channel, segment

FEATURES = (
    "n_pre", "n_ev", "n_post", "dur_h", "iqr_pre", "mag_z", "peak_z", "peak_z99", "sgn_z",
    "frac_pos", "end_shift_z", "ret_frac", "trend_r2", "col_frac", "settle_z", "t_peak_frac",
    "width_frac", "imp_ratio", "kurt", "skew", "var_ratio", "jag", "dz_frac_ev", "dz_frac_pre",
    "e1_ev", "e1_ratio", "e4_ev", "e4_ratio", "e16_ev", "e16_ratio", "tilt_ev", "med_dt_ev",
    "med_dt_pre", "rate_ratio", "cv_dt_ev", "spread_dt_ev", "gap_frac", "zero_frac",
    "const_frac", "mode_frac", "mode_z",
)


def _iqr(x: np.ndarray) -> float:
    if len(x) < 4:
        return np.nan
    q75, q25 = np.percentile(x, [75, 25])
    return q75 - q25


def _lin_r2(t: np.ndarray, z: np.ndarray) -> float:
    """R^2 of the linear fit z ~ t."""
    if len(z) < 5:
        return np.nan
    t = t - t.mean()
    denom = np.sum((z - z.mean()) ** 2)
    tt = np.sum(t * t)
    if denom <= 0 or tt <= 0:
        return 0.0
    b = np.sum(t * (z - z.mean())) / tt
    return float(1 - np.sum((z - (b * t + z.mean())) ** 2) / denom)


def _longest_run_frac(v: np.ndarray) -> float:
    """Length of the longest run of identical values over the window length."""
    if len(v) < 2:
        return np.nan
    change = np.flatnonzero(np.diff(v) != 0)
    if len(change) == 0:
        return 1.0
    edges = np.concatenate(([-1], change, [len(v) - 1]))
    return float(np.max(np.diff(edges)) / len(v))


def _diff_energy(ts_ns: np.ndarray, v: np.ndarray, lag_s: float, iqr: float) -> float:
    """P75 of |v(t + lag) - v(t)| over the ruler, pairing each sample with the first one
    at least ``lag_s`` later and keeping pairs whose actual lag is within [0.5, 2] x lag."""
    n = len(v)
    if n < 8:
        return np.nan
    j = np.searchsorted(ts_ns, ts_ns + int(lag_s * 1e9))
    ok = j < n
    i = np.flatnonzero(ok)
    if len(i) < 5:
        return np.nan
    j = j[ok]
    adt = (ts_ns[j] - ts_ns[i]) / 1e9
    good = (adt >= 0.5 * lag_s) & (adt <= 2.0 * lag_s)
    if good.sum() < 5:
        return np.nan
    return float(np.percentile(np.abs(v[j[good]] - v[i[good]]), 75) / iqr)


def _collinear_frac(t: np.ndarray, v: np.ndarray, iqr: float) -> float:
    """Fraction of interior samples lying on the line through their neighbours (linear
    interpolation fill, a sign of missing data)."""
    if len(v) < 5:
        return np.nan
    t = t.astype(float)
    dt = t[2:] - t[:-2]
    ok = dt > 0
    pred = np.where(ok, v[:-2] + (v[2:] - v[:-2]) * (t[1:-1] - t[:-2]) / np.where(ok, dt, 1), v[1:-1])
    return float(np.mean(np.abs(v[1:-1] - pred) <= max(1e-6 * iqr, 1e-12)))


def window_features(seg: pd.DataFrame, gap_dt_s: float) -> dict:
    """Features of one window from its ``segment`` frame. ``gap_dt_s`` is the channel's
    95th-percentile sampling interval: event intervals above 10 x it count as gaps."""
    ts_all = seg["timestamp"].to_numpy("datetime64[ns]").astype(np.int64)
    v_all = seg["value"].to_numpy(float)
    code, oev, dec = seg["seg"].to_numpy(), seg["other_ev"].to_numpy(), seg["dec"].to_numpy()

    m_pre = (code == PRE) & ~oev
    m_post = (code == POST) & ~oev
    v_pre, t_pre, dec_pre = v_all[m_pre], ts_all[m_pre], dec[m_pre]
    v_ev, t_ev = v_all[code == EVENT], ts_all[code == EVENT]
    v_post, t_post = v_all[m_post], ts_all[m_post]

    out = {"n_pre": len(v_pre), "n_ev": len(v_ev), "n_post": len(v_post)}
    if len(v_ev) == 0:
        return out
    dur_s = (t_ev[-1] - t_ev[0]) / 1e9 if len(t_ev) > 1 else 0.0
    out["dur_h"] = dur_s / 3600

    # robust ruler from the clean pre-event context
    if len(v_pre) >= 8:
        med_pre = float(np.median(v_pre))
        iqr = _iqr(v_pre)
        if not np.isfinite(iqr) or iqr <= 0:
            mad = np.median(np.abs(v_pre - med_pre))
            iqr = 1.4826 * mad * 1.349 if mad > 0 else np.nan
        if not np.isfinite(iqr) or iqr <= 0:
            sd = np.std(v_pre)
            iqr = 1.349 * sd if sd > 0 else np.nan
    else:
        med_pre, iqr = float(np.median(v_ev)), np.nan
    if not np.isfinite(iqr) or iqr <= 0:
        sd = np.std(v_ev)  # last resort: the event's own scale
        iqr = 1.349 * sd if sd > 0 else 1.0
    out["iqr_pre"] = iqr
    z = (v_ev - med_pre) / iqr
    az = np.abs(z)

    # magnitude
    out["mag_z"] = float(np.median(az))
    out["peak_z"] = float(np.max(az))
    out["peak_z99"] = float(np.percentile(az, 99))
    out["sgn_z"] = float(np.median(z))
    out["frac_pos"] = float(np.mean(z > 0))

    # persistence: level after the event, over clip(duration, 1 h, 1 d)
    if len(v_post) >= 3:
        mm = t_post <= t_ev[-1] + int(min(max(dur_s, 3600), 86400) * 1e9)
        vp = v_post[mm] if mm.sum() >= 3 else v_post
        out["end_shift_z"] = float(abs(np.median((vp - med_pre) / iqr)))
    else:
        out["end_shift_z"] = np.nan
    if np.isfinite(out["end_shift_z"]):
        out["ret_frac"] = float(np.clip(1 - out["end_shift_z"] / max(out["mag_z"], 0.5), -1, 1))
    else:
        out["ret_frac"] = np.nan

    # shape
    tsec = (t_ev - t_ev[0]) / 1e9
    out["trend_r2"] = _lin_r2(tsec, z)
    out["col_frac"] = _collinear_frac(t_ev, v_ev, iqr)
    if len(z) >= 8:
        q = len(z) // 4
        out["settle_z"] = float(np.median(z[-q:]) - np.median(z[:q]))
    else:
        out["settle_z"] = np.nan
    out["t_peak_frac"] = float(tsec[int(np.argmax(az))] / tsec[-1]) if tsec[-1] > 0 else np.nan
    out["width_frac"] = float(np.mean(az > 0.5 * out["peak_z"])) if out["peak_z"] > 0 else np.nan
    out["imp_ratio"] = float(out["peak_z"] / max(out["mag_z"], 0.1))
    if len(z) >= 8 and np.std(z) > 0:
        zz = (z - z.mean()) / z.std()
        out["kurt"] = float(np.clip(np.mean(zz ** 4) - 3, -3, 50))
        out["skew"] = float(np.clip(np.mean(zz ** 3), -10, 10))
    else:
        out["kurt"] = out["skew"] = np.nan

    # variability
    iqr_ev = _iqr(v_ev)
    out["var_ratio"] = float(iqr_ev / iqr) if np.isfinite(iqr_ev) else np.nan
    vpre_f = v_pre[~dec_pre]  # full-resolution context only
    if len(v_ev) >= 6 and len(vpre_f) >= 6:
        de = np.median(np.abs(np.diff(v_ev)))
        dp = np.median(np.abs(np.diff(vpre_f)))
        out["jag"] = float((de / iqr) / max(dp / iqr, 1e-6))
        out["dz_frac_ev"] = float(np.mean(np.diff(v_ev) == 0))
        out["dz_frac_pre"] = float(np.mean(np.diff(vpre_f) == 0))
    else:
        out["jag"] = out["dz_frac_ev"] = out["dz_frac_pre"] = np.nan

    # frequency content: lag-difference energies at multiples of the context cadence
    dt_ev = np.diff(t_ev) / 1e9
    med_dt_ev = float(np.median(dt_ev)) if len(dt_ev) else np.nan
    t_pre_f = t_pre[~dec_pre]
    dt_pre = np.diff(t_pre_f) / 1e9
    med_dt_pre = float(np.median(dt_pre)) if len(dt_pre) else np.nan
    base = med_dt_pre if np.isfinite(med_dt_pre) and med_dt_pre > 0 else (med_dt_ev or 60.0)
    for k, mult in (("e1", 1), ("e4", 4), ("e16", 16)):
        ev_e = _diff_energy(t_ev, v_ev, base * mult, iqr)
        pr_e = _diff_energy(t_pre_f, vpre_f, base * mult, iqr) if len(vpre_f) > 8 else np.nan
        out[f"{k}_ev"] = ev_e
        out[f"{k}_ratio"] = (float(ev_e / max(pr_e, 0.05))
                             if np.isfinite(ev_e) and np.isfinite(pr_e) else np.nan)
    if np.isfinite(out["e1_ev"]) and np.isfinite(out["e16_ev"]) and out["e16_ev"] > 0:
        out["tilt_ev"] = float(out["e1_ev"] / out["e16_ev"])
    else:
        out["tilt_ev"] = np.nan

    # sampling
    out["med_dt_ev"] = med_dt_ev
    out["med_dt_pre"] = med_dt_pre
    out["rate_ratio"] = (float(med_dt_pre / med_dt_ev)
                         if np.isfinite(med_dt_pre) and med_dt_ev and med_dt_ev > 0 else np.nan)
    if len(dt_ev) >= 4:
        out["cv_dt_ev"] = float(np.std(dt_ev) / max(np.mean(dt_ev), 1e-9))
        p90, p10 = np.percentile(dt_ev, [90, 10])
        out["spread_dt_ev"] = float(p90 / max(p10, 1e-9))
    else:
        out["cv_dt_ev"] = out["spread_dt_ev"] = np.nan
    if np.isfinite(gap_dt_s) and dur_s > 0 and len(dt_ev):
        out["gap_frac"] = float(dt_ev[dt_ev > 10 * gap_dt_s].sum() / max(dur_s, 1e-9))
    else:
        out["gap_frac"] = np.nan

    # frozen data
    out["zero_frac"] = float(np.mean(v_ev == 0))
    out["const_frac"] = _longest_run_frac(v_ev)
    vals, counts = np.unique(v_ev, return_counts=True)
    im = int(np.argmax(counts))
    out["mode_frac"] = float(counts[im] / len(v_ev))
    out["mode_z"] = float(abs((vals[im] - med_pre) / iqr))
    return out


def feature_table(windows: pd.DataFrame, sampling: pd.DataFrame, mission: str = "ESA-Mission1",
                  verbose: bool = True) -> pd.DataFrame:
    """Features of every window that has at least one sample in its window or context.

    ``windows`` comes from ``load_windows``, ``sampling`` from ``sampling_table`` (for the
    channel gap thresholds). Returns the window keys followed by ``FEATURES``.
    """
    gap_dt = (sampling.set_index("channel").med_dt_s * sampling.set_index("channel").p95_over_med).to_dict()
    rows = []
    for ch, w in windows.groupby("channel", sort=True):
        ts, v = load_channel(mission, ch)
        intervals = list(zip(w.t0, w.t1))
        n0 = len(rows)
        for r in w.itertuples(index=False):
            others = [(a, b) for a, b in intervals if not (a == r.t0 and b == r.t1)]
            seg = segment(ts, v, r.t0, r.t1, others)
            if seg is None:
                if verbose:
                    print(f"  {r.event_id} channel_{ch}: no samples in the window or its context, skipped")
                continue
            rows.append(dict(event_id=r.event_id, channel=ch, interval=r.interval, group=r.group,
                             **window_features(seg, gap_dt.get(ch, np.nan))))
        if verbose:
            print(f"  channel_{ch}: {len(rows) - n0} windows", flush=True)
    keys = ["event_id", "channel", "interval", "group"]
    return pd.DataFrame(rows).reindex(columns=keys + list(FEATURES))
