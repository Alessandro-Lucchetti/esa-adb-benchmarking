"""Golden regression tool for SMED labels (Phase 0 seed -> future repo tool).

Usage:
  python golden.py snapshot <predictions_root> <out.json>      # hash all */labels/channel_*.parquet
  python golden.py verify   <predictions_root> <golden.json>   # compare, exit 1 on any mismatch
  python golden.py compare  <dirA> <dirB>                       # ad-hoc: do two run dirs match?

Key   = "<run_name>/<channel>"
Value = sha256( read_parquet["label"].to_numpy(int8).tobytes() )   # matches generate_reference.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def hash_labels(parquet: Path) -> tuple[str, int, int]:
    arr = pd.read_parquet(parquet)["label"].to_numpy(dtype=np.int8)
    return hashlib.sha256(arr.tobytes()).hexdigest(), int(arr.sum()), int(arr.size)


def snapshot_root(root: Path) -> dict:
    out: dict[str, dict] = {}
    for f in sorted(root.glob("*/labels/channel_*.parquet")):
        run = f.parent.parent.name
        ch = f.stem
        digest, n_anom, n = hash_labels(f)
        out[f"{run}/{ch}"] = {"sha256": digest, "anomalies": n_anom, "length": n}
    return out


def env_versions() -> dict:
    import numpy
    v = {"numpy": numpy.__version__}
    for mod in ("numba", "rocket_fft", "pandas", "pyarrow"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = None
    return v


def cmd_snapshot(root: str, out: str):
    data = snapshot_root(Path(root).resolve())
    payload = {"versions": env_versions(), "source": str(Path(root).resolve()), "checksums": data}
    Path(out).write_text(json.dumps(payload, indent=2))
    print(f"snapshot: {len(data)} channel-files -> {out}")
    for k, v in list(data.items())[:6]:
        print(f"  {k}: {v['anomalies']} anom / {v['length']}  {v['sha256'][:16]}")
    if len(data) > 6:
        print(f"  ... (+{len(data)-6} more)")


def cmd_verify(root: str, golden: str):
    gold = json.loads(Path(golden).read_text())["checksums"]
    cur = snapshot_root(Path(root).resolve())
    miss, mism, ok = [], [], 0
    for k, g in gold.items():
        if k not in cur:
            miss.append(k)
        elif cur[k]["sha256"] != g["sha256"]:
            mism.append(k)
        else:
            ok += 1
    extra = [k for k in cur if k not in gold]
    print(f"PASS {ok}/{len(gold)}  | mismatch {len(mism)} | missing {len(miss)} | extra {len(extra)}")
    for k in mism[:20]:
        print(f"  MISMATCH {k}: gold {gold[k]['sha256'][:12]} ({gold[k]['anomalies']} anom) vs cur {cur[k]['sha256'][:12]} ({cur[k]['anomalies']} anom)")
    for k in miss[:20]:
        print(f"  MISSING  {k}")
    sys.exit(1 if (mism or miss) else 0)


def cmd_compare(a: str, b: str):
    ha, hb = snapshot_root(Path(a).resolve()), snapshot_root(Path(b).resolve())
    # strip run-name prefix so dirs with different run names but same channels compare by channel
    def by_ch(h):
        return {k.split("/")[-1]: v for k, v in h.items()}
    ca, cb = by_ch(ha), by_ch(hb)
    keys = sorted(set(ca) | set(cb))
    ok = mism = 0
    for k in keys:
        if k in ca and k in cb:
            if ca[k]["sha256"] == cb[k]["sha256"]:
                ok += 1
            else:
                mism += 1
                print(f"  MISMATCH {k}: A {ca[k]['sha256'][:12]} ({ca[k]['anomalies']}) vs B {cb[k]['sha256'][:12]} ({cb[k]['anomalies']})")
        else:
            print(f"  ONLY in {'A' if k in ca else 'B'}: {k}")
    print(f"compare: {ok} identical, {mism} differ ({len(keys)} channels)")
    sys.exit(1 if mism else 0)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "snapshot":
        cmd_snapshot(sys.argv[2], sys.argv[3])
    elif cmd == "verify":
        cmd_verify(sys.argv[2], sys.argv[3])
    elif cmd == "compare":
        cmd_compare(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(1)
