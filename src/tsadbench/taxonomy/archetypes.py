"""From 117 classes to 8 archetypes: traits, Ward hierarchy and threshold rules.

Each unit (113 density classes + 4 rule-based categories) is described by 12 traits,
robust medians over its windows (``TRAITS``). Ward linkage on the robustly standardized
traits (every unit weighs the same) shows a jump between 8 and 7 clusters, and the
archetypes are defined by 8 threshold rules applied in order of precedence (``archetype``);
their agreement with the K = 8 Ward cut (units carrying the majority archetype of their
Ward cluster) is reported, not enforced. Archetypes are numbered
A1..A8 by decreasing window count and units are coded C1..C117 by archetype block, then by
leaf order of the dendrogram.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage

from tsadbench.taxonomy.classify import CATEGORY_NAMES, EMPTY

TRAITS = ("dur_samples_log10", "dur_hours_log10", "amplitude_log10", "direction", "recovery",
          "impulsiveness", "rel_width", "agitation_log10", "rate_log10", "gap_frac", "absence",
          "frozen")
K = 8
UNASSIGNED = "U0"


def _median(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def unit_traits(F: pd.DataFrame, m: np.ndarray) -> dict:
    """The 12 traits of the windows selected by mask ``m`` of the feature table ``F``."""
    col = lambda c: F[c].to_numpy(float)[m]
    n_ev = col("n_ev")
    s = col("sgn_z")
    s = s[np.isfinite(s) & (np.abs(s) > 1)]
    return {
        "dur_samples_log10": _median(np.log10(n_ev + 1)),
        "dur_hours_log10": _median(np.log10(col("dur_h") + 1e-3)),
        "amplitude_log10": _median(np.log10(col("mag_z") + .1)),
        # sign balance of the windows that leave the +/-1 IQR band
        "direction": float((s > 0).mean() * 2 - 1) if len(s) >= 3 else 0.0,
        "recovery": _median(col("ret_frac")),
        "impulsiveness": _median(np.log10(col("imp_ratio") + .1)),
        "rel_width": _median(col("width_frac")),
        "agitation_log10": _median(np.log10(col("var_ratio") + 1e-3)),
        "rate_log10": _median(np.log10(col("rate_ratio") + 1e-3)),
        "gap_frac": _median(col("gap_frac")),
        "absence": float((n_ev == 0).mean()),
        "frozen": _median(col("const_frac")),
    }


def trait_table(features: pd.DataFrame, lab: np.ndarray, cat: np.ndarray) -> pd.DataFrame:
    """One row per unit (density classes in label order, then the rule-based categories),
    missing traits imputed: median, or minimum for amplitude and impulsiveness, and the
    empty category set to no amplitude, direction, texture or agitation."""
    group = features["group"].to_numpy()
    rows = []
    for k in sorted(set(lab[lab >= 0])):
        m = lab == k
        rows.append(dict(unit=f"class {k}", unit_type="class", group=int(group[np.flatnonzero(m)[0]]),
                         n_windows=int(m.sum()), **unit_traits(features, m)))
    for c, name in CATEGORY_NAMES.items():
        m = cat == c
        rows.append(dict(unit=name, unit_type="rule-based", group=-1, n_windows=int(m.sum()),
                         **unit_traits(features, m)))
    T = pd.DataFrame(rows)
    for c in TRAITS:
        T[c] = T[c].astype(float)
        T[c] = T[c].fillna(T[c].min() if c in ("amplitude_log10", "impulsiveness") else T[c].median())
    empty = T.unit == CATEGORY_NAMES[EMPTY]
    T.loc[empty, ["amplitude_log10", "direction", "impulsiveness", "rel_width", "agitation_log10"]] = \
        [T.amplitude_log10.min(), 0, T.impulsiveness.min(), 0, 0]
    return T


def standardized_traits(T: pd.DataFrame) -> np.ndarray:
    X = T[list(TRAITS)].values.astype(float)
    med = np.median(X, 0)
    iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
    scale = np.maximum(iqr, .75 * X.std(0)) + 1e-9
    return np.clip((X - med) / scale, -4, 4)


def archetype(r) -> str:
    """Archetype of a unit from its traits; the first rule that holds wins."""
    if r.absence >= .5:
        return "silence"
    if r.dur_samples_log10 <= 0.9:  # about 7 samples or fewer
        return "point-like"
    if r.amplitude_log10 >= 1.0 and r.direction <= -0.5 and r.frozen >= .4:
        return "frozen collapse"
    if r.rate_log10 >= 0.7:
        # the sampling rate multiplies: alone if the value stays in the noise and recovers
        return "pure rate change" if (r.amplitude_log10 <= -0.5 and r.recovery >= 0.5) else "regime switch"
    if r.dur_hours_log10 >= 1.0 and r.amplitude_log10 >= -0.2:
        return "sustained excursion"
    if r.amplitude_log10 <= -0.5 and r.recovery >= .8:
        return "quiet interval"
    return "short deviation"


def build_taxonomy(features: pd.DataFrame, lab: np.ndarray, cat: np.ndarray,
                   windows: pd.DataFrame) -> dict:
    """Archetypes, codes and the tables of the taxonomy.

    ``features`` and ``lab``/``cat`` are row-aligned (``classify.assign_classes``);
    ``windows`` (``windows.load_windows``) provides times and ESA categories. Returns a
    dict with ``units``, ``windows``, ``archetypes``, ``linkage`` (scipy format) and
    ``meta`` (fusion heights, cut, agreement between rules and the Ward cut).
    """
    T = trait_table(features, lab, cat)
    Xs = standardized_traits(T)
    L = linkage(Xs, method="ward")
    h = np.sort(L[:, 2])
    leaf = leaves_list(L)
    pos = np.empty(len(T), int)
    pos[leaf] = np.arange(len(T))
    T["ward_cluster"] = fcluster(L, t=K, criterion="maxclust")
    T["archetype"] = T.apply(archetype, axis=1)

    order = T.groupby("archetype").n_windows.sum().sort_values(ascending=False, kind="stable")
    aid = {a: f"A{i + 1}" for i, a in enumerate(order.index)}
    T["archetype_id"] = T.archetype.map(aid)
    code_order = np.lexsort((pos, T.archetype_id.str[1:].astype(int).to_numpy()))
    code = np.empty(len(T), object)
    code[code_order] = [f"C{r + 1}" for r in range(len(T))]
    T["class_code"] = code
    T["leaf_pos"] = pos + 1

    # per-window assignment, aligned with features
    unit_of = np.where(lab >= 0, [f"class {k}" for k in lab],
                       [CATEGORY_NAMES.get(c, "unassigned") for c in cat])
    by_unit = T.set_index("unit")
    keys = ["event_id", "channel", "interval"]
    W = features[keys + ["group"]].merge(windows[keys + ["t0", "t1", "esa_category"]], on=keys,
                                         how="left", validate="one_to_one")
    W["unit"] = unit_of
    assigned = W.unit != "unassigned"
    W["class_code"] = np.where(assigned, W.unit.map(by_unit.class_code), UNASSIGNED)
    W["archetype_id"] = np.where(assigned, W.unit.map(by_unit.archetype_id), UNASSIGNED)
    W["archetype"] = np.where(assigned, W.unit.map(by_unit.archetype), "unassigned")
    T["n_events"] = [W.event_id[W.unit == u].nunique() for u in T.unit]

    rows = []
    for a in order.index:
        S = T[T.archetype == a]
        w = W[W.archetype == a]
        codes = sorted(S.class_code, key=lambda s: int(s[1:]))
        n_lab = w.esa_category.isin(["Anomaly", "Rare Event", "Communication Gap"]).sum()
        pct = lambda c: round((w.esa_category == c).sum() / max(n_lab, 1) * 100, 1)
        rows.append(dict(
            archetype_id=aid[a], archetype=a, code_first=codes[0], code_last=codes[-1],
            n_units=len(S), n_classes=int((S.unit_type == "class").sum()),
            n_rule_based=int((S.unit_type == "rule-based").sum()), n_windows=int(S.n_windows.sum()),
            n_events=w.event_id.nunique(),
            n_groups=S.group[S.group >= 0].nunique() + int((S.unit_type == "rule-based").any()),
            pct_anomaly=pct("Anomaly"), pct_rare=pct("Rare Event"), pct_gap=pct("Communication Gap")))
    A = pd.DataFrame(rows)

    # agreement of the rules with the Ward cut: units carrying the majority archetype of
    # their Ward cluster
    ct = pd.crosstab(T.ward_cluster, T.archetype)
    meta = dict(n_windows=len(W), n_assigned=int(assigned.sum()),
                n_classes=int((T.unit_type == "class").sum()),
                n_rule_based=int((T.unit_type == "rule-based").sum()),
                h_9to8=float(h[-K]), h_8to7=float(h[-K + 1]), cut=float((h[-K] + h[-K + 1]) / 2),
                max_pairwise=float(np.max(np.linalg.norm(Xs[:, None] - Xs[None], axis=2))),
                rules_vs_ward=int(ct.max(axis=1).sum()))

    units = T[["class_code", "unit", "unit_type", "group", "n_windows", "n_events", "archetype_id",
               "archetype", "ward_cluster", "leaf_pos", *TRAITS]]
    W = W[["event_id", "channel", "interval", "t0", "t1", "group", "esa_category", "unit",
           "class_code", "archetype_id", "archetype"]]
    return dict(units=units, windows=W, archetypes=A, linkage=L, meta=meta)
