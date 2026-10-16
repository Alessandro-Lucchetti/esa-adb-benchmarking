"""Typed reader for the metrics stored in a run's config.json.

The engine persists metrics as 3-decimal *strings* under `metrics_all_events`
and `metrics_anomalies_alone` (older runs carry fewer sub-keys). This module
parses them into typed floats so downstream analysis (leaderboards, stats,
per-class tables) never juggles strings or missing keys.

    rm = load_run_metrics("smed/predictions/<run_name>")     # or .../config.json, or a config dict
    rm.all_events.event_wise.f0_5        # -> 0.902 (float) or None
    rm.flat()                            # -> {"all_events.event_wise.f0.5": 0.902, ...}
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_LENIENT = ConfigDict(extra="ignore", populate_by_name=True)


class PRF(BaseModel):
    """precision / recall / F0.5 triple."""
    model_config = _LENIENT
    precision: float | None = None
    recall: float | None = None
    f0_5: float | None = Field(default=None, alias="f0.5")


class ADTQC(BaseModel):
    model_config = _LENIENT
    after_rate: float | None = None
    total_score: float | None = None


class MetricSet(BaseModel):
    model_config = _LENIENT
    event_wise: PRF | None = None
    subsystem_aware: PRF | None = None
    channel_aware: PRF | None = None
    affiliation_based: PRF | None = None
    alarming_precision: float | None = None
    adtqc: ADTQC | None = None


class RunMetrics(BaseModel):
    model_config = _LENIENT
    all_events: MetricSet | None = None
    anomalies_alone: MetricSet | None = None

    def flat(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for scope in ("all_events", "anomalies_alone"):
            ms = getattr(self, scope)
            if ms is None:
                continue
            for aspect in ("event_wise", "subsystem_aware", "channel_aware", "affiliation_based"):
                prf = getattr(ms, aspect)
                if prf is None:
                    continue
                for field, key in (("precision", "precision"), ("recall", "recall"), ("f0_5", "f0.5")):
                    val = getattr(prf, field)
                    if val is not None:
                        out[f"{scope}.{aspect}.{key}"] = val
            if ms.alarming_precision is not None:
                out[f"{scope}.alarming_precision"] = ms.alarming_precision
            if ms.adtqc is not None:
                for field in ("after_rate", "total_score"):
                    val = getattr(ms.adtqc, field)
                    if val is not None:
                        out[f"{scope}.adtqc.{field}"] = val
        return out


def load_run_metrics(config_or_path: str | Path | dict) -> RunMetrics:
    if isinstance(config_or_path, dict):
        data = config_or_path
    else:
        p = Path(config_or_path)
        if p.is_dir():
            p = p / "config.json"
        data = json.loads(p.read_text())
    return RunMetrics(
        all_events=data.get("metrics_all_events"),
        anomalies_alone=data.get("metrics_anomalies_alone"),
    )
