"""From window features to 117 classes: the common signature and the class assignment.

Signature: the 37 morphological metrics of ``METRICS`` (heavy-tailed ones
log-compressed) plus one applicability flag per metric that is missing on some windows
(36). Metrics are standardized robustly and balanced by channel group (centre and scale
are averages of the per-group median and IQR, so no group sets the units of the others),
clipped to +/-5, with missing values at 0; the flag block carries ``FLAG_WEIGHT`` of the
variance. PCA to 90% of the variance gives 15 dimensions.

Classes:

1. rule-based categories for windows without shape: empty (no sample) and single-point
   windows (1-2 samples) going up, down or staying within ``SIGN_DEADBAND`` IQR;
2. HDBSCAN on the signature within each channel group (min_cluster_size = max(3, 2% of
   the group), min_samples = 1, cluster_selection_epsilon = 0.30, leaf selection);
3. density classes whose durations (in samples) span more than ``SPLIT_DECADES`` decades
   between their 10th and 90th percentiles are split recursively at the largest gap of
   the log-duration, as long as both parts keep at least ``SPLIT_MIN_CHILD`` windows.

Windows left as HDBSCAN noise stay unassigned (label -1, category 0).
"""
import numpy as np
import pandas as pd
import hdbscan
from sklearn.decomposition import PCA

METRICS = (
    "n_ev", "dur_h", "mag_z", "peak_z", "peak_z99", "sgn_z", "frac_pos", "end_shift_z",
    "ret_frac", "trend_r2", "col_frac", "settle_z", "t_peak_frac", "width_frac", "imp_ratio",
    "kurt", "skew", "var_ratio", "jag", "dz_frac_ev", "dz_frac_pre", "e1_ev", "e1_ratio",
    "e4_ev", "e4_ratio", "e16_ev", "e16_ratio", "tilt_ev", "med_dt_ev", "rate_ratio",
    "cv_dt_ev", "spread_dt_ev", "gap_frac", "zero_frac", "const_frac", "mode_frac", "mode_z",
)
LOG_PREFIXES = ("mag_z", "peak_z", "peak_z99", "imp_ratio", "var_ratio", "jag", "e1_", "e4_",
                "e16_", "tilt_", "spread_dt", "rate_ratio", "med_dt", "mode_z", "end_shift",
                "n_ev", "dur_h", "gap_frac", "cv_dt")
FLAG_WEIGHT = 0.10
PCA_VARIANCE = 0.90
SEED = 42

MIN_CLUSTER_FRAC = 0.020
MIN_SAMPLES = 1
CLUSTER_EPS = 0.30
MIN_GROUP_SIZE = 6
SPLIT_DECADES = 2.5
SPLIT_MIN_SIZE = 8
SPLIT_MIN_CHILD = 4
SIGN_DEADBAND = 1.0

EMPTY, POINT_UP, POINT_DOWN, POINT_NOISE = 1, 2, 3, 4
CATEGORY_NAMES = {EMPTY: "empty", POINT_UP: "single point up", POINT_DOWN: "single point down",
                  POINT_NOISE: "single point in noise"}


def signature(features: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Common signature of every window, reduced by PCA. Returns (Z, info)."""
    group = features["group"].to_numpy()
    R = features[list(METRICS)].replace([np.inf, -np.inf], np.nan).astype(float)
    for c in METRICS:
        if c.startswith(LOG_PREFIXES):
            R[c] = np.log10(np.clip(R[c], 1e-3, None) + 1e-3)
    missing = R.isna()
    flags = [c for c in missing.columns if missing[c].std() > 0]

    groups = sorted(set(group))
    med = np.nanmean(np.vstack([R[group == g].median().values for g in groups]), 0)
    iqr = np.nanmean(np.vstack([(R[group == g].quantile(.75) - R[group == g].quantile(.25)).values
                                for g in groups]), 0)
    X = np.clip((R.values - med) / np.where(iqr > 0, iqr, 1), -5, 5)
    X[missing.values] = 0.0
    X = X / np.sqrt(np.nanvar(X, 0).sum())
    F = missing[flags].astype(float).values
    F = F / np.sqrt(F.var(0).sum())
    XX = np.hstack([X, F * np.sqrt(FLAG_WEIGHT)])

    pca = PCA(n_components=PCA_VARIANCE, random_state=SEED).fit(XX)
    Z = pca.transform(XX)
    info = dict(n_windows=len(Z), n_groups=len(groups), n_metrics=len(METRICS), flags=flags,
                n_components=int(Z.shape[1]), explained_variance=float(pca.explained_variance_ratio_.sum()))
    return Z, info


def rule_categories(features: pd.DataFrame) -> np.ndarray:
    """0 for windows that go to the density clustering, else the rule-based category."""
    n_ev = features["n_ev"].to_numpy(float)
    sgn_z = features["sgn_z"].to_numpy(float)
    cat = np.zeros(len(features), dtype=int)
    cat[n_ev == 0] = EMPTY
    point = (n_ev >= 1) & (n_ev <= 2)
    cat[point & (sgn_z > SIGN_DEADBAND)] = POINT_UP
    cat[point & (sgn_z < -SIGN_DEADBAND)] = POINT_DOWN
    cat[point & (np.abs(sgn_z) <= SIGN_DEADBAND)] = POINT_NOISE
    return cat


def _split_by_duration(idx: np.ndarray, n_ev: np.ndarray) -> list[np.ndarray]:
    d = np.log10(n_ev[idx])
    lo, hi = np.percentile(d, [10, 90])
    if len(idx) < SPLIT_MIN_SIZE or hi - lo < SPLIT_DECADES:
        return [idx]
    o = np.argsort(d)
    gaps = np.diff(d[o])
    pos = np.arange(len(gaps))
    allowed = (pos + 1 >= SPLIT_MIN_CHILD) & (len(idx) - pos - 1 >= SPLIT_MIN_CHILD)
    if not allowed.any():
        return [idx]
    cut = pos[allowed][np.argmax(gaps[allowed])]
    return _split_by_duration(idx[o[:cut + 1]], n_ev) + _split_by_duration(idx[o[cut + 1:]], n_ev)


def density_clusters(Z: np.ndarray, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """HDBSCAN cluster of each window within its channel group (-1 when none), before the
    duration split, and its rule-based category (0 when none)."""
    group = features["group"].to_numpy()
    cat = rule_categories(features)

    lab = np.full(len(Z), -1, dtype=int)
    off = 0
    for g in sorted(set(group)):
        i = np.flatnonzero((group == g) & (cat == 0))
        if len(i) < MIN_GROUP_SIZE:
            continue
        mcs = max(3, int(round(MIN_CLUSTER_FRAC * len(i))))
        l = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=MIN_SAMPLES, cluster_selection_epsilon=CLUSTER_EPS,
                            cluster_selection_method="leaf").fit(Z[i]).labels_.copy()
        l[l >= 0] += off
        lab[i] = l
        off = lab.max() + 1
    return lab, cat


def split_clusters(clusters: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    """Classes from the density clusters: clusters spanning too many decades of duration are
    split at their largest log-duration gaps; new classes are numbered after the clusters."""
    n_ev = features["n_ev"].to_numpy(float)
    lab = clusters.copy()
    off = lab.max() + 1
    for k in sorted(set(lab[lab >= 0])):
        parts = _split_by_duration(np.flatnonzero(lab == k), n_ev)
        for p in parts[1:]:
            lab[p] = off
            off += 1
    return lab


def assign_classes(Z: np.ndarray, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Density class of each window (-1 when none) and its rule-based category (0 when none)."""
    clusters, cat = density_clusters(Z, features)
    return split_clusters(clusters, features), cat
