"""Run a single experiment from a typed RunConfig.

Thin orchestration over the SMED engine. For experimental_mode runs without an
explicit exp_id, a timestamped one is generated so outputs land under
`<algo>/experiments/<exp_id>/`; otherwise they go to `<algo>/predictions/`.
"""
from __future__ import annotations

from datetime import datetime

from tsadbench.runner.config import RunConfig
from tsadbench.detectors.smed.core import run_smed


def run(config: RunConfig) -> dict:
    cfg = config.to_run_dict()
    if cfg["experimental_mode"] and not cfg.get("exp_id"):
        cfg["exp_id"] = f"exp_{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}"
    return run_smed(cfg)
