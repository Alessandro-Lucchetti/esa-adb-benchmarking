"""Event taxonomy: rebuild of the reference tables and the per-archetype hit rule."""
import numpy as np
import pandas as pd
import pytest

from tsadbench.taxonomy import compare_tables, read_table, reference_dir
from tsadbench.taxonomy.recall import window_hits

REF = reference_dir()


def _ref(name: str) -> pd.DataFrame:
    path = REF / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"reference table not found: {path}")
    return read_table(path)


def test_classification_reproduces_reference():
    """Signature, classes, traits, Ward and rules rebuilt from the reference window features
    give the reference tables and the numbers quoted in the paper."""
    pytest.importorskip("hdbscan")
    from tsadbench.taxonomy.archetypes import build_taxonomy
    from tsadbench.taxonomy.classify import density_clusters, signature, split_clusters
    from tsadbench.taxonomy.summary import summary_numbers

    F = _ref("window_features")
    Z, info = signature(F)
    clusters, cat = density_clusters(Z, F)
    lab = split_clusters(clusters, F)
    tax = build_taxonomy(F, lab, cat, _ref("windows"))

    assert info["n_components"] == 15 and len(info["flags"]) == 36
    m = tax["meta"]
    assert (m["n_windows"], m["n_assigned"], m["n_classes"], m["n_rule_based"]) == (3586, 3313, 113, 4)
    assert m["rules_vs_ward"] == 97
    assert round(m["h_9to8"], 2) == 7.53 and round(m["h_8to7"], 2) == 9.30
    assert tax["archetypes"].n_windows.tolist() == [1303, 939, 535, 187, 144, 102, 58, 45]

    for name in ("units", "windows", "archetypes"):
        assert compare_tables(tax[name], _ref(name)) == [], name
    L = tax["linkage"]
    D = _ref("dendrogram")
    assert np.array_equal(L[:, :2].astype(int), D[["left", "right"]].to_numpy())
    np.testing.assert_allclose(L[:, 2], D.height, rtol=1e-12)

    s = summary_numbers(F, tax["windows"], tax["units"], clusters)
    assert (s["n_empty"], s["n_empty_zero_duration"], s["n_point_up"], s["n_point_down"],
            s["n_point_noise"], s["n_density_windows"], s["n_density_clusters"]) == (187, 186, 142, 27, 206, 3024, 108)
    assert round(s["purity_mean"], 2) == 0.62 and round(s["purity_baseline_mean"], 2) == 0.25
    assert s["n_purity_above_baseline"] == 113
    assert (s["n_rules_vs_ward_disagree"], s["n_disagree_duration_amplitude"], s["n_disagree_other"],
            s["n_disagree_single_unit"]) == (20, 14, 5, 1)
    assert (s["rate_top_event_A5"], s["rate_top_event_A7"], s["A6_n_units_variability_floor"]) == (50, 29, 9)
    assert (s["n_low_excursion_windows"], s["n_events_all_low_excursion"]) == (1026, 16)


def _run(tmp_path, ts, labels):
    (tmp_path / "labels").mkdir()
    pd.DataFrame({"timestamp": pd.to_datetime(ts), "label": np.array(labels, dtype=np.int8)}) \
        .to_parquet(tmp_path / "labels" / "channel_1.parquet")
    return tmp_path


def test_window_hits_sample_and_hold(tmp_path):
    run = _run(tmp_path, ["2010-01-01 00:00", "2010-01-01 01:00", "2010-01-01 02:00", "2010-01-01 03:00"],
               [0, 1, 0, 0])
    W = pd.DataFrame({
        "channel": 1,
        "t0": pd.to_datetime(["2010-01-01 00:30", "2010-01-01 01:30", "2010-01-01 01:30",
                              "2010-01-01 02:30", "2010-01-01 01:00"]),
        "t1": pd.to_datetime(["2010-01-01 01:00", "2010-01-01 01:30", "2010-01-01 01:45",
                              "2010-01-01 03:00", "2010-01-01 01:00"]),
    })
    # alarm sample inside; zero-duration label while the alarm is held; interval while held;
    # only quiet samples; zero-duration label on the alarm sample itself
    assert window_hits(W, run).tolist() == [1.0, 1.0, 1.0, 0.0, 1.0]


def test_window_hits_needs_every_channel(tmp_path):
    run = _run(tmp_path, ["2010-01-01"], [0])
    W = pd.DataFrame({"channel": [1, 2], "t0": pd.to_datetime(["2010-01-01"] * 2),
                      "t1": pd.to_datetime(["2010-01-02"] * 2)})
    with pytest.raises(FileNotFoundError, match=r"\[2\]"):
        window_hits(W, run)


def test_features_and_sampling_match_reference_on_channel_41(has_mission1_raw):
    """Raw telemetry to sampling statistics and window features, on one channel."""
    if not has_mission1_raw:
        pytest.skip("data/ESA-Mission1/raw not available")
    from tsadbench.taxonomy.features import feature_table
    from tsadbench.taxonomy.sampling import sampling_table
    from tsadbench.taxonomy.windows import load_windows

    stats = sampling_table(channels=[41], verbose=False)
    ref_stats = _ref("sampling_stats")
    assert compare_tables(stats, ref_stats[ref_stats.channel == 41].reset_index(drop=True)) == []

    W = load_windows()
    F = feature_table(W[W.channel == 41], stats, verbose=False)
    ref_F = _ref("window_features")
    assert compare_tables(F, ref_F[ref_F.channel == 41].reset_index(drop=True)) == []
