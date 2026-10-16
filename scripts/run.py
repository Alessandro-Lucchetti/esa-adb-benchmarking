#!/usr/bin/env python
"""Run a single SMED experiment from a YAML config.

  python scripts/run.py configs/experiments/mission1_full_phase5.yaml
  python scripts/run.py configs/base/smed.yaml --set mission=ESA-Mission1 channels_mode=light phase=3 experimental_mode=true
"""
import argparse

from tsadbench.runner.config import RunConfig, load_config
from tsadbench.runner.run import run


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config", help="path to a YAML run config")
    ap.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="override config fields (values are coerced to the field type)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.set:
        data = cfg.model_dump()
        for kv in args.set:
            key, _, val = kv.partition("=")
            data[key] = val
        cfg = RunConfig(**data)
    run(cfg)


if __name__ == "__main__":
    main()
