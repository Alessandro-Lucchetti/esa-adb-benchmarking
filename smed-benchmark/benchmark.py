import os

# Must be set before importing numpy/numba
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from smed_core.algorithm import eval_channel
from smed_core.distance import best_match
from smed_core.helpers import parse_channels_mode, ROOT

OUTPUT_DIR = Path("/output")
REF_FILE = ROOT / "reference_checksums.json"

CONFIG_BASE = {
    "mission": "ESA-Mission1",
    "phase": "5",
    "query_length": 100,
    "reference_length": 750,
    "slide_length": 5,
    "k": 1.5,
    "p": 0.1,
    "warm_up_mode": "forward",
    "signal_mode": "raw",
}


def jit_warmup() -> float:
    print("Triggering numba JIT compilation...", flush=True)
    t0 = time.time()
    dummy_q = np.zeros(CONFIG_BASE["query_length"], dtype=np.float32)
    dummy_r = np.zeros(CONFIG_BASE["reference_length"], dtype=np.float32)
    best_match(dummy_q, dummy_r)
    elapsed = time.time() - t0
    print(f"JIT warmup complete in {elapsed:.2f}s\n", flush=True)
    return elapsed


def sha256_labels(labels: np.ndarray) -> str:
    return hashlib.sha256(labels.tobytes()).hexdigest()


def run_mode(mode: str) -> tuple[dict, dict]:
    """
    Run benchmark for one channel mode (light or full).
    Returns (timing, checksums) where timing has keys training/evaluation/other
    and checksums maps channel -> sha256 hex string.
    """
    config = {**CONFIG_BASE, "channels_mode": mode}
    channels = sorted(parse_channels_mode(config))

    print(f"\n{'='*60}", flush=True)
    print(f"Mode: {mode}  ({len(channels)} channels)", flush=True)
    print(f"{'='*60}", flush=True)

    total_warmup = 0.0
    total_eval = 0.0
    checksums: dict[str, str] = {}

    wall_start = time.time()

    for ch in tqdm(channels, desc=f"{mode}", unit="ch"):
        ch_name, _, labels, _, _, warmup_time, eval_time = eval_channel(ch, config)
        total_warmup += warmup_time
        total_eval += eval_time
        checksums[ch_name] = sha256_labels(labels)

    total_wall = time.time() - wall_start
    other = total_wall - total_warmup - total_eval

    timing = {
        "training": total_warmup,
        "evaluation": total_eval,
        "other": other,
    }

    print(f"\n  Training  (warmup)    : {total_warmup:9.2f}s", flush=True)
    print(f"  Evaluation            : {total_eval:9.2f}s", flush=True)
    print(f"  Other (I/O, etc.)     : {other:9.2f}s", flush=True)
    print(f"  Total wall clock      : {total_wall:9.2f}s", flush=True)

    return timing, checksums


def verify_checksums(computed: dict[str, dict[str, str]]) -> bool:
    if not REF_FILE.exists():
        print("\nNo reference_checksums.json found -- skipping verification.", flush=True)
        return True

    with open(REF_FILE) as f:
        ref = json.load(f)

    ref_versions = ref.get("versions", {})
    ref_checksums = ref.get("checksums", {})

    import numpy, numba, rocket_fft
    cur_versions = {
        "numpy": numpy.__version__,
        "numba": numba.__version__,
        "rocket_fft": rocket_fft.__version__,
    }

    print("\n--- Library versions ---", flush=True)
    version_ok = True
    for lib, ver in cur_versions.items():
        ref_ver = ref_versions.get(lib, "unknown")
        match = "OK" if ver == ref_ver else "WARN"
        if ver != ref_ver:
            version_ok = False
        print(f"  {lib:12s}: {ver:10s}  (reference: {ref_ver})  [{match}]", flush=True)
    if not version_ok:
        print("  WARNING: version mismatch -- label checksums may differ", flush=True)

    print("\n--- Checksum verification ---", flush=True)
    all_pass = True
    for mode in ["light", "full"]:
        mode_checksums = computed.get(mode, {})
        for ch, expected in ref_checksums.items():
            actual = mode_checksums.get(ch)
            if actual is None:
                continue  # channel not in this mode
            status = "PASS" if actual == expected else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  [{status}] {mode}/{ch}", flush=True)

    if all_pass:
        print("\nAll checksums match -- algorithm output is correct.", flush=True)
    else:
        print("\nSome checksums FAILED -- output differs from reference.", flush=True)

    return all_pass


def write_csv(timings: dict[str, dict]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "benchmark_results.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "light", "full"])
        for row in ["training", "evaluation", "other"]:
            writer.writerow([row, timings["light"][row], timings["full"][row]])
    return out


def main():
    t_start = time.time()
    print("SMED Benchmark", flush=True)
    print(f"Config: phase={CONFIG_BASE['phase']}, q={CONFIG_BASE['query_length']}, "
          f"r={CONFIG_BASE['reference_length']}, s={CONFIG_BASE['slide_length']}, "
          f"k={CONFIG_BASE['k']}, p={CONFIG_BASE['p']}", flush=True)

    jit_warmup()

    timings: dict[str, dict] = {}
    all_checksums: dict[str, dict] = {}

    for mode in ["light", "full"]:
        timing, checksums = run_mode(mode)
        timings[mode] = timing
        all_checksums[mode] = checksums

    verify_checksums(all_checksums)

    csv_path = write_csv(timings)
    print(f"\nResults written to {csv_path}", flush=True)
    print(f"Total elapsed: {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
