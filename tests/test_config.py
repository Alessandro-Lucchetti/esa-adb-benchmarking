"""Unit tests for the config system and run-name encode/decode (no data needed)."""
import warnings

import pytest

from tsadbench.runner.config import RunConfig, load_config
from tsadbench.runner.naming import build_run_name, parse_run_name
from tsadbench.io.misc import project_root

CONFIGS = project_root() / "configs"


def test_extends_merge():
    cfg = load_config(CONFIGS / "experiments" / "mission1_light_phase5.yaml")
    # inherited from base/smed.yaml
    assert cfg.query_length == 100 and cfg.reference_length == 750 and cfg.k == 1.5
    # overridden by the child
    assert cfg.mission == "ESA-Mission1" and cfg.channels_mode == "light"
    assert cfg.phase == "5" and cfg.experimental_mode is True


def test_query_must_not_exceed_reference():
    with pytest.raises(ValueError):
        RunConfig(query_length=100, reference_length=50)


def test_extra_field_forbidden():
    with pytest.raises(Exception):
        RunConfig(bogus=1)


def test_phase_coerced_to_str():
    assert RunConfig(phase=3).phase == "3"


def test_mission2_phase5_downgrades_to_4():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = RunConfig(mission="ESA-Mission2", phase="5")
    assert cfg.phase == "4"
    assert any("Phase 5" in str(x.message) for x in w)


def test_to_run_dict_has_engine_keys():
    keys = RunConfig().to_run_dict().keys()
    for k in ("algo_name", "mission", "channels_mode", "signal_mode", "phase",
              "warm_up_mode", "query_length", "reference_length", "slide_length",
              "k", "p", "plot", "experimental_mode"):
        assert k in keys


def test_run_name_roundtrip():
    cfg = RunConfig(mission="ESA-Mission1", channels_mode="light", phase="5").to_run_dict()
    name = build_run_name(cfg, ["channel_41", "channel_42"])
    assert name == "smed_ESA-Mission1_phase_5_raw_light_q100_r750_s5_k1.5_p0.1"
    parsed = parse_run_name(name)
    assert parsed["mission"] == "ESA-Mission1" and parsed["query_length"] == 100
    assert parsed["k"] == 1.5 and parsed["channels_label"] == "light"


def test_run_name_manual_label():
    cfg = RunConfig(channels_mode="manual").to_run_dict()
    name = build_run_name(cfg, ["channel_18"])
    assert "_channel_18_q" in name
