"""Summary numbers of the taxonomy quoted in the paper text, computed from its tables.

``summary_numbers`` takes the window features, the per-window assignment and the unit
table (``archetypes.build_taxonomy``) and returns a flat dict that ``scripts/taxonomy.py``
stores under ``summary`` in ``taxonomy.json``:

- rule-based categories: empty windows (and how many have a zero-duration label),
  single-point windows above, below and within the noise band;
- density stage: windows clustered, HDBSCAN clusters before the duration split;
- neighbourhood purity of the archetypes in trait space (paper Eq. 1) and its random
  baseline;
- disagreements between the rules and the K = 8 Ward cut, by kind;
- cadence multipliers of the regime switch (A5) and pure rate change (A7) windows, and the
  event contributing most of them;
- quiet interval (A6) classes at the variability floor;
- windows and events of the archetypes without a value-based waveform excursion (A3-A7).
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from tsadbench.taxonomy.archetypes import standardized_traits
from tsadbench.taxonomy.classify import EMPTY, POINT_DOWN, POINT_NOISE, POINT_UP, rule_categories

KEYS = ["event_id", "channel", "interval"]
# archetypes whose windows carry no value-based waveform excursion
LOW_EXCURSION = ("A3", "A4", "A5", "A6", "A7")
# archetypes along the continuous duration-amplitude boundary of the Ward partition
DURATION_AMPLITUDE = {"A1", "A2", "A6"}


def neighbourhood_purity(units: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Purity of each unit and its random baseline (paper Eq. 1).

    For a unit of an archetype with n units, the share of its n - 1 nearest units in
    standardized trait space (itself excluded) that carry the same archetype; the
    baseline is (n - 1) / (N - 1). Both are 0 for an archetype with a single unit.
    """
    X = standardized_traits(units)
    a = units["archetype_id"].to_numpy()
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    N = len(units)
    purity, baseline = np.zeros(N), np.zeros(N)
    for i in range(N):
        n = int((a == a[i]).sum())
        if n == 1:
            continue
        nearest = np.argsort(D[i], kind="stable")[:n - 1]
        purity[i] = (a[nearest] == a[i]).mean()
        baseline[i] = (n - 1) / (N - 1)
    return purity, baseline


def summary_numbers(features: pd.DataFrame, windows: pd.DataFrame, units: pd.DataFrame,
                    clusters: np.ndarray | None = None) -> dict:
    """``features`` and ``windows`` are row-aligned; ``clusters`` are the HDBSCAN labels
    before the duration split (``classify.density_clusters``), when available."""
    cat = rule_categories(features)
    W = windows.assign(n_ev=features["n_ev"].to_numpy(), rate_ratio=features["rate_ratio"].to_numpy())
    empty = W[cat == EMPTY]
    zero = empty.t0 == empty.t1
    out = dict(
        n_empty=int(len(empty)),
        n_empty_zero_duration=int(zero.sum()),
        empty_outage_hours=[round((r.t1 - r.t0).total_seconds() / 3600, 3) for r in empty[~zero].itertuples()],
        n_point_up=int((cat == POINT_UP).sum()),
        n_point_down=int((cat == POINT_DOWN).sum()),
        n_point_noise=int((cat == POINT_NOISE).sum()),
        n_density_windows=int((cat == 0).sum()),
        n_class_windows=int(W.unit.str.startswith("class ").sum()),
        n_unassigned=int((W.archetype_id == "U0").sum()),
        assigned_share=float((W.archetype_id != "U0").mean()),
    )
    if clusters is not None:
        out["n_density_clusters"] = int(len(set(clusters[clusters >= 0])))

    purity, baseline = neighbourhood_purity(units)
    single = units.groupby("archetype_id").archetype_id.transform("size").to_numpy() == 1
    out.update(purity_mean=float(purity.mean()), purity_baseline_mean=float(baseline.mean()),
               n_purity_above_baseline=int((purity > baseline).sum()),
               purity_mean_multi_unit=float(purity[~single].mean()))

    # each Ward cluster is mapped to the archetype holding most of its units
    ct = pd.crosstab(units.ward_cluster, units.archetype_id)
    ward = units.ward_cluster.map(ct.idxmax(axis=1))
    off = units[ward != units.archetype_id].assign(ward=ward)
    boundary = off.archetype_id.isin(DURATION_AMPLITUDE) & off.ward.isin(DURATION_AMPLITUDE)
    single_off = off.archetype_id.map(units.archetype_id.value_counts()) == 1
    out.update(n_rules_vs_ward_disagree=int(len(off)), n_disagree_duration_amplitude=int(boundary.sum()),
               n_disagree_single_unit=int(single_off.sum()),
               n_disagree_other=int((~boundary & ~single_off).sum()))

    for aid in ("A5", "A7"):
        r = W.loc[W.archetype_id == aid, "rate_ratio"]
        out.update({f"{aid}_n_windows": int(len(r)), f"{aid}_rate_median": float(r.median()),
                    f"{aid}_rate_max": float(r.max()), f"{aid}_n_rate_max": int((r == r.max()).sum())})
    by_event = W[W.archetype_id.isin(["A5", "A7"])].groupby(["event_id", "archetype_id"]).size() \
        .unstack(fill_value=0)
    top = by_event.sort_values(["A5", "A7"], ascending=False, kind="stable").iloc[0]
    out.update(rate_top_event=str(top.name), rate_top_event_A5=int(top["A5"]), rate_top_event_A7=int(top["A7"]))

    a6 = units[units.archetype_id == "A6"]
    out["A6_n_units_variability_floor"] = int((a6.agitation_log10 <= units.agitation_log10.min()).sum())

    low = W.archetype_id.isin(LOW_EXCURSION)
    assigned = W[W.archetype_id != "U0"]
    all_low = assigned.archetype_id.isin(LOW_EXCURSION).groupby(assigned.event_id).all()
    out.update(n_low_excursion_windows=int(low.sum()), low_excursion_share=float(low.mean()),
               n_events=int(W.event_id.nunique()), n_events_all_low_excursion=int(all_low.sum()))
    return out
