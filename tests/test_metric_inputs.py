"""The reduced metric inputs (label changes only) give the same events as the full series."""
import numpy as np
import pandas as pd
import pytest

from tsadbench.eval.metrics.utils import convert_time_series_to_events, label_changes
from tsadbench.io.misc import build_union_labels


def _series(rng, n, p_switch, start="2007-01-01", jitter=True):
    step = rng.integers(20, 100, n) if jitter else np.full(n, 30)
    t = pd.Timestamp(start) + pd.to_timedelta(np.cumsum(step), unit="s")
    lab = (np.cumsum(rng.random(n) < p_switch) % 2).astype(np.int8)
    return t, lab


@pytest.mark.parametrize("seed", range(20))
def test_label_changes_keep_the_events(seed):
    rng = np.random.default_rng(seed)
    t, lab = _series(rng, int(rng.integers(1, 400)), rng.choice([0.01, 0.2, 0.7]))
    full = list(zip(t, lab))
    reduced = label_changes(t, lab)
    assert len(reduced) <= len(full)
    assert reduced[0] == full[0] and reduced[-1] == full[-1]
    assert convert_time_series_to_events(reduced) == convert_time_series_to_events(full)


def _union_reference(frames):
    """The union on the whole timestamp grid (the former implementation)."""
    series = [df.set_index("timestamp")["label"].astype("uint8") for df in frames]
    idx = series[0].index
    for s in series[1:]:
        idx = idx.union(s.index)
    idx = idx.sort_values()
    aligned = [s.reindex(idx).ffill().fillna(0).astype("uint8") for s in series]
    any_active = pd.concat(aligned, axis=1).max(axis=1).astype("uint8")
    return list(zip(any_active.index, any_active.values))


@pytest.mark.parametrize("seed", range(10))
def test_union_labels_keep_the_events(tmp_path, seed):
    rng = np.random.default_rng(seed)
    frames = []
    (tmp_path / "labels").mkdir()
    for c in range(int(rng.integers(1, 6))):
        start = pd.Timestamp("2007-01-01") + pd.Timedelta(minutes=int(rng.integers(0, 600)))
        t, lab = _series(rng, int(rng.integers(1, 300)), rng.choice([0.02, 0.3]), start=start)
        df = pd.DataFrame({"timestamp": t, "label": lab})
        df.to_parquet(tmp_path / "labels" / f"channel_{c + 1}.parquet")
        frames.append(df)
    got = build_union_labels(tmp_path)
    assert convert_time_series_to_events(got) == convert_time_series_to_events(_union_reference(frames))
