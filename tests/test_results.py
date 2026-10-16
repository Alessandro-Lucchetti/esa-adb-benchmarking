"""Tests for the typed metrics reader against real on-disk config.json files."""
import pytest

from tsadbench.eval.results import RunMetrics, load_run_metrics
from tsadbench.io.misc import project_root

ROOT = project_root()
FULL_RUN = ROOT / "smed" / "predictions" / "smed_ESA-Mission1_phase_5_raw_full_q100_r750_s5_k1.5_p0.1"


def test_string_values_parse_to_floats():
    rm = load_run_metrics({
        "metrics_all_events": {
            "event_wise": {"precision": "0.982", "recall": "0.681", "f0.5": "0.902"},
            "alarming_precision": "0.689",
            "adtqc": {"after_rate": "0.177", "total_score": "0.540"},
        }
    })
    assert rm.all_events.event_wise.f0_5 == pytest.approx(0.902)
    assert rm.all_events.event_wise.precision == pytest.approx(0.982)
    assert rm.all_events.alarming_precision == pytest.approx(0.689)
    assert rm.all_events.adtqc.after_rate == pytest.approx(0.177)
    assert rm.anomalies_alone is None  # absent -> None, no crash


def test_legacy_missing_aspects_are_none():
    # older runs only carry event_wise + alarming_precision
    rm = load_run_metrics({
        "metrics_all_events": {
            "event_wise": {"precision": "0.911", "recall": "0.477", "f0.5": "0.771"},
            "alarming_precision": "0.574",
        }
    })
    assert rm.all_events.event_wise.recall == pytest.approx(0.477)
    assert rm.all_events.subsystem_aware is None
    assert rm.all_events.adtqc is None


def test_flat_dict():
    rm = load_run_metrics({
        "metrics_all_events": {"event_wise": {"precision": "0.9", "recall": "0.6", "f0.5": "0.8"}},
    })
    flat = rm.flat()
    assert flat["all_events.event_wise.f0.5"] == pytest.approx(0.8)


@pytest.mark.skipif(not FULL_RUN.exists(), reason="paper full-set predictions not present")
def test_read_real_full_run():
    rm = load_run_metrics(FULL_RUN)
    assert rm.all_events.event_wise.f0_5 == pytest.approx(0.902)
    assert rm.all_events.channel_aware.f0_5 == pytest.approx(0.569)
    assert rm.all_events.affiliation_based is not None
    assert isinstance(rm.flat(), dict) and rm.flat()
