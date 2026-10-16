"""Typed, validated experiment configuration (pydantic v2 + YAML).

A `RunConfig` is the single front door for an experiment. YAML files may inherit
from a base via a top-level `_extends: <relative-path>` key (shallow merge, child
overrides parent). Use `load_config(path)` to read+validate.

`RunConfig.to_run_dict()` yields the plain dict consumed by the SMED engine
(`tsadbench.detectors.smed.core.run_smed`), keeping the numeric path unchanged.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algo_name: str = "smed"
    mission: Literal["ESA-Mission1", "ESA-Mission2"] = "ESA-Mission1"
    channels_mode: Literal["light", "full", "manual", "dev"] = "full"
    signal_mode: Literal["raw", "resampled"] = "raw"
    phase: Literal["1", "2", "3", "4", "5"] = "5"
    warm_up_mode: Literal["forward", "backward"] = "forward"

    query_length: int = 100
    reference_length: int = 750
    slide_length: int = 5
    k: float = 1.5
    p: float = 0.1

    plot: bool = False
    experimental_mode: bool = False
    exp_id: str | None = None
    run_id: str | None = None

    @field_validator("phase", mode="before")
    @classmethod
    def _phase_to_str(cls, v):
        return str(v)

    @field_validator("query_length", "reference_length", "slide_length")
    @classmethod
    def _positive(cls, v, info):
        if v <= 0:
            raise ValueError(f"{info.field_name} must be > 0, got {v}")
        return v

    @model_validator(mode="after")
    def _check(self):
        if self.query_length > self.reference_length:
            raise ValueError(
                f"query_length ({self.query_length}) must be <= reference_length "
                f"({self.reference_length})"
            )
        # Phase 5 exists only for Mission 1; make the downgrade explicit at config time.
        if self.mission == "ESA-Mission2" and self.phase == "5":
            warnings.warn("Phase 5 is Mission-1 only; downgrading to Phase 4 for ESA-Mission2.")
            object.__setattr__(self, "phase", "4")
        return self

    def to_run_dict(self) -> dict:
        return self.model_dump()


def _load_yaml_with_extends(path: Path, _seen: set[Path] | None = None) -> dict:
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular _extends involving {path}")
    _seen.add(path)
    data = yaml.safe_load(path.read_text()) or {}
    base = data.pop("_extends", None)
    if base is None:
        return data
    base_data = _load_yaml_with_extends((path.parent / base), _seen)
    base_data.update(data)
    return base_data


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a RunConfig from a YAML file (supports `_extends`)."""
    return RunConfig(**_load_yaml_with_extends(Path(path)))
