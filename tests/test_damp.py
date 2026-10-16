"""DAMP detector: z-norm kernel correctness + the controlled SMED-vs-DAMP comparison."""
import numpy as np
import pytest

from tsadbench.detectors import REGISTRY, get_detector
from tsadbench.detectors.damp.distance import best_match_znorm
from tsadbench.detectors.smed.distance import best_match
from tsadbench.data.splits import Split
from tsadbench.runner.config import RunConfig


def _brute_znorm(x, y, m):
    xn = (x - x.mean()) / (x.std() if x.std() > 1e-6 else 1e-8)
    best = np.inf
    for k in range(len(y) - m + 1):
        w = y[k:k + m].astype(np.float64)
        s = w.std()
        d = np.sqrt(2 * m) if s <= 1e-6 else np.sqrt(((xn - (w - w.mean()) / s) ** 2).sum())
        best = min(best, d)
    return best


def test_znorm_kernel_matches_brute_force():
    rng = np.random.default_rng(7)
    for _ in range(4):
        m = 80
        y = rng.standard_normal(1000).astype(np.float32)
        x = rng.standard_normal(m).astype(np.float32)
        kern, _ = best_match_znorm(x, y)
        assert abs(kern - _brute_znorm(x.astype(np.float64), y.astype(np.float64), m)) < 1e-2


def test_znorm_handles_constant_windows():
    # constant-vs-constant -> 0; varying query vs flat region -> finite, > 0
    d0, _ = best_match_znorm(np.ones(50, np.float32), np.ones(200, np.float32))
    assert d0 == 0.0
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(200, np.float32), rng.standard_normal(200).astype(np.float32)])
    d1, _ = best_match_znorm(rng.standard_normal(50).astype(np.float32), y)
    assert np.isfinite(d1) and d1 > 0.0


def test_damp_registered_and_validates():
    assert "damp" in REGISTRY
    assert get_detector("damp").euclidean == "znorm"
    with pytest.raises(ValueError):
        get_detector("damp", euclidean="cosine")


@pytest.mark.slow
def test_damp_standard_reproduces_smed(has_mission1_raw):
    """euclidean='standard' must give labels identical to SMED (shared pipeline check)."""
    if not has_mission1_raw:
        pytest.skip("ESA-Mission1/raw not present")
    sp = Split(RunConfig(mission="ESA-Mission1", channels_mode="light", phase="5", signal_mode="raw"))
    smed = get_detector("smed").score(sp)
    damp_std = get_detector("damp", euclidean="standard").score(sp)
    for ch in sp.channels:
        assert np.array_equal(smed.labels[ch], damp_std.labels[ch]), f"DAMP-standard != SMED for {ch}"


@pytest.mark.slow
def test_damp_znorm_differs_from_standard(has_mission1_raw):
    if not has_mission1_raw:
        pytest.skip("ESA-Mission1/raw not present")
    sp = Split(RunConfig(mission="ESA-Mission1", channels_mode="light", phase="5", signal_mode="raw"))
    std = get_detector("damp", euclidean="standard").score(sp)
    zn = get_detector("damp", euclidean="znorm").score(sp)
    differ = any(not np.array_equal(std.labels[ch], zn.labels[ch]) for ch in sp.channels)
    assert differ, "z-normalized DAMP produced identical labels to standard - distance not applied?"
