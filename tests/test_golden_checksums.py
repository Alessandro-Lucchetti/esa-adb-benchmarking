"""Regression gate: SMED must produce bit-exact identical labels after any refactor.

Re-runs SMED on the ESA-Mission1 raw splits and compares per-channel int8 label
SHA-256 hashes against the committed golden baseline (golden_checksums.json).

  pytest tests/test_golden_checksums.py            # light phase 5 (~1 min)
  pytest tests/test_golden_checksums.py -m slow    # all light phases (~4 min)

Requires the ESA-Mission1/raw dataset locally (skipped otherwise).
"""
import hashlib
import shutil

import numpy as np
import pandas as pd
import pytest

from tsadbench.detectors.smed.core import run_smed
from tsadbench.io.misc import project_root

ROOT = project_root()

BASE_CFG = dict(
    algo_name="smed", mission="ESA-Mission1", channels_mode="light",
    query_length=100, reference_length=750, slide_length=5,
    warm_up_mode="forward", k=1.5, p=0.1, signal_mode="raw",
    plot=False, experimental_mode=True,
)


def _hash(parquet) -> str:
    return hashlib.sha256(pd.read_parquet(parquet)["label"].to_numpy(np.int8).tobytes()).hexdigest()


def _run_and_check(phase: str, golden: dict):
    exp_id = f"_pytest_golden_p{phase}"
    out_root = ROOT / "smed" / "experiments" / exp_id
    if out_root.exists():
        shutil.rmtree(out_root)
    cfg = {**BASE_CFG, "phase": phase, "exp_id": exp_id, "run_id": f"{exp_id}_run"}
    try:
        run_smed(cfg)
        run_dir = next(out_root.glob("smed_*"))
        run_name = run_dir.name
        labels = sorted((run_dir / "labels").glob("channel_*.parquet"))
        assert labels, f"no label parquet produced for phase {phase}"
        mismatches = []
        for f in labels:
            key = f"{run_name}/{f.stem}"
            assert key in golden, f"golden has no entry for {key}"
            if _hash(f) != golden[key]["sha256"]:
                mismatches.append(f.stem)
        assert not mismatches, f"phase {phase}: labels changed for {mismatches}"
    finally:
        if out_root.exists():
            shutil.rmtree(out_root)


def test_smed_light_phase5_bitexact(golden, has_mission1_raw):
    if not has_mission1_raw: pytest.skip("ESA-Mission1/raw not present")
    _run_and_check("5", golden)


@pytest.mark.slow
@pytest.mark.parametrize("phase", ["1", "2", "3", "4", "5"])
def test_smed_light_all_phases_bitexact(phase, golden, has_mission1_raw):
    if not has_mission1_raw: pytest.skip("ESA-Mission1/raw not present")
    _run_and_check(phase, golden)
