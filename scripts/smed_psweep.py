"""SMED percentile sweep at the nominal window sizes (q100 / r750 / s5, k=1.5).

Companion of the OOL p sensitivity (scripts/limits_psweep.py): same p grid, same
event-wise scoring, so the two curves can be compared directly. Reuses the
distance-agnostic sweep of damp_psweep.py with SMED's raw Euclidean kernel: warm-up
and test distances are computed once per channel and re-thresholded for every p,
which is orders of magnitude cheaper than one full SMED run per p.

p=0.1 is the paper's canonical configuration and must reproduce the published SMED
numbers (light 0.999 / 0.462 / 0.811, full 0.982 / 0.681 / 0.902).

Usage:
    python scripts/smed_psweep.py <channels_mode> [--phase N] [p1 p2 ...]
        channels_mode: light | full
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsadbench.detectors.smed.distance import best_match
from damp_psweep import sweep, cli, DEFAULT_PS

ALGO = "smed_psweep"


if __name__ == "__main__":
    args = cli(__doc__)
    sweep(args.channels_mode, args.ps or DEFAULT_PS, phase=args.phase,
          dist_fn=best_match, algo=ALGO, euclidean="raw")
