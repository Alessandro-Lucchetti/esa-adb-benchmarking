"""Guards the SMED distance against the float32 catastrophic-cancellation regression.

An earlier version of mass() accumulated in float32 instead of float64. Because the
distance profile is assembled as ``sum_x2 + sum_y2 - 2*dot`` (a small difference of large,
near-equal values near zero-distance), float32 makes near-zero distances collapse or become
platform-dependent (channel_66 gave 226 on rocket_fft 0.3.1 vs 399 on 0.2.5, true value 449).
These tests need no dataset and fail fast if mass() is ever downgraded again.
"""
import numpy as np

from tsadbench.detectors.smed.distance import mass, best_match


def test_mass_accumulates_in_float64():
    x = np.full(100, 1.0, dtype=np.float32)
    y = np.full(750, 1.0, dtype=np.float32)
    assert mass(x, y).dtype == np.float64, "mass() must accumulate in float64 (do not revert to float32)"


def test_best_match_survives_catastrophic_cancellation():
    # Large amplitude (sum_x2 ~ 1e10) with a single 0.5 offset => true best-match distance = 0.5.
    # In float32 the sum_x2 + sum_y2 - 2*dot cancellation collapses this to 0.0; float64 keeps it.
    C, m, n = 1e4, 100, 750
    x = np.full(m, C, dtype=np.float32)
    x[0] = np.float32(C + 0.5)
    y = np.full(n, C, dtype=np.float32)

    d, _ = best_match(x, y)
    assert abs(d - 0.5) < 1e-2, f"expected ~0.5, got {d} (float32 cancellation would give ~0)"
