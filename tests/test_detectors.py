"""Detector registry, Split resolution, and SMED-adapter bit-exactness."""
import hashlib
import json

import numpy as np
import pytest

from tsadbench.detectors import REGISTRY, get_detector
from tsadbench.detectors.base import Detector, DetectorResult
from tsadbench.data.splits import Split
from tsadbench.runner.config import RunConfig
from tsadbench.runner.naming import build_run_name
from tsadbench.io.misc import project_root

ROOT = project_root()


def test_smed_registered():
    assert "smed" in REGISTRY
    det = get_detector("smed")
    assert det.name == "smed"
    assert isinstance(det, Detector)  # satisfies the runtime-checkable protocol


def test_unknown_detector_raises():
    with pytest.raises(KeyError):
        get_detector("does-not-exist")


def test_split_resolves_light_channels():
    sp = Split(RunConfig(mission="ESA-Mission1", channels_mode="light"))
    assert sp.channels == [f"channel_{i}" for i in range(41, 47)]


def test_split_paths():
    sp = Split(RunConfig(mission="ESA-Mission1", channels_mode="light", signal_mode="raw"))
    assert sp.channel_path("channel_41").as_posix().endswith(
        "data/ESA-Mission1/raw/channel_41.parquet"
    )


@pytest.mark.slow
def test_smed_adapter_bitexact(has_mission1_raw):
    """The Detector adapter must yield the same labels as the golden baseline."""
    if not has_mission1_raw:
        pytest.skip("ESA-Mission1/raw not present")
    golden = json.loads((ROOT / "golden_checksums.json").read_text())["checksums"]
    sp = Split(RunConfig(mission="ESA-Mission1", channels_mode="light", phase="5",
                         signal_mode="raw", experimental_mode=True))
    result = get_detector("smed").score(sp)
    run = build_run_name(sp.run_config_dict, sp.channels)
    assert set(result.labels) == set(sp.channels)
    for ch, lab in result.labels.items():
        h = hashlib.sha256(lab.astype(np.int8).tobytes()).hexdigest()
        assert h == golden[f"{run}/{ch}"]["sha256"], f"SMED adapter labels drifted for {ch}"
