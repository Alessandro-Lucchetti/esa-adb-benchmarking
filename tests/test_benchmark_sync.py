"""Guard against drift between the standalone timing benchmark and the src library.

smed-benchmark/ is a dependency-light, version-pinned copy of the SMED numeric core that
runs standalone in Docker for timing. Its distance kernel accumulates in float32 while
src accumulates in float64, so the two agree to float32 precision rather than bit for
bit; on real telemetry a few near-threshold windows can flip, as they can across
hardware and OS. This test catches algorithmic drift, not rounding.
"""
import sys

import numpy as np
import pytest

from tsadbench.io.misc import project_root

ROOT = project_root()
BENCH = ROOT / "smed-benchmark"


@pytest.mark.skipif(not (BENCH / "smed_core" / "algorithm.py").exists(), reason="benchmark absent")
def test_run_algorithm_matches_src():
    from tsadbench.detectors.smed.core import run_algorithm as src_run

    sys.path.insert(0, str(BENCH))
    import smed_core.algorithm as bench  # noqa: E402

    rng = np.random.default_rng(0)
    x = rng.standard_normal(6000).astype(np.float32)
    for thr in (np.inf, 12.0):
        l_s, q_s, d_s = src_run(x, slide_length=5, query_length=100, reference_length=750, threshold=thr)
        l_b, q_b, d_b = bench.run_algorithm(x, slide_length=5, query_length=100, reference_length=750, threshold=thr)
        assert np.array_equal(q_s, q_b), "query_starts differ"
        assert np.array_equal(l_s, l_b), "labels differ between src and benchmark"
        np.testing.assert_allclose(d_b, d_s, rtol=1e-5, err_msg="distances differ beyond float32 precision")
