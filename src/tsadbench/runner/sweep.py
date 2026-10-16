import itertools
import time
from datetime import datetime
import multiprocessing
import pandas as pd
from tsadbench.detectors.smed.core import run_smed
from tsadbench.io.misc import project_root, save_json


def main():
    # param_grid = {
    #     "mission": ["ESA-Mission1", "ESA-Mission2"],
    #     "channels_mode": ["lightweight"],
    #     "query_length": [10, 25, 50, 75, 100],
    #     "reference_length": [100, 250, 500, 750, 1000],
    #     "slide_length": [30],
    #     "phase": ["5"],
    #     "warm_up_mode": ["forward"],
    #     "k": [1.5],
    #     "p": [0.1],
    #     "signal_mode": ["resampled"],
    #     "plot": [False],
    #     "experimental_mode": [True],
    #     "algo_name": ["smed"],
    # }

    param_grid = {
        "mission": ["ESA-Mission1"],
        "channels_mode": ["light"],
        "query_length": [100],
        "reference_length": [750],
        "slide_length": [5],
        "phase": ["1", "2", "3", "4", "5"],
        "warm_up_mode": ["forward"],
        "k": [1.5],
        "p": [0.1],
        "signal_mode": ["raw"],
        "plot": [False],
        "experimental_mode": [True],
        "algo_name": ["smed"],
    }

    # param_grid = {
    #     "mission": ["ESA-Mission1"],
    #     "channels_mode": ["light"],
    #     "query_length": [10, 100, 1000],
    #     "reference_length": [100, 1000, 10000],
    #     "slide_length": [5],
    #     "phase": ["5"],
    #     "warm_up_mode": ["forward"],
    #     "k": [1.5],
    #     "p": [0.1],
    #     "signal_mode": ["raw"],
    #     "plot": [False],
    #     "experimental_mode": [True],
    #     "algo_name": ["smed"],
    # }

    # param_grid = {
    #     "mission": ["ESA-Mission1"],
    #     "channels_mode": ["all"],
    #     "query_length": [10, 25, 50, 75, 100],
    #     "reference_length": [100, 250, 500, 750, 1000],
    #     "slide_length": [5],
    #     "phase": ["5"],
    #     "warm_up_mode": ["forward"],
    #     "k": [1.5],
    #     "p": [0.1],
    #     "dynamic_threshold": [False],
    #     "signal_mode": ["raw"],
    #     "algo_mode": ["supervised"],
    #     "algo_name": ["smed"],
    #     "plot": [False],
    #     "ew_only": [True],
    #     "experimental_mode": [True]
    # }

    log_dir = project_root() / "smed" / "experiments"
    log_dir.mkdir(exist_ok=True)

    keys, values = zip(*param_grid.items())
    combinations = list(itertools.product(*values))

    i_q = keys.index("query_length")
    i_ref = keys.index("reference_length")
    i_slide = keys.index("slide_length")

    combinations.sort(key=lambda c: (-c[i_ref], c[i_q], c[i_slide]))

    time_id = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    exp_id = f"exp_{time_id}"

    print(f"Experiment id: {exp_id}")
    print(f"Running {len(combinations)} combination")

    df = pd.DataFrame(columns=["run_id", "mission", "query_length", "reference_length", "slide_length", "ew_metrics", "config"])
    t0 = time.time()
    for i, combo in enumerate(combinations):
        config = dict(zip(keys, combo))
        if config["query_length"] > config["reference_length"]:
            print(f"Skipping invalid combination: query_length {config['query_length']} > reference_length {config['reference_length']}")
            continue
        config["exp_id"] = exp_id
        run_id = f"{exp_id}_run_{i:03d}"
        config["run_id"] = run_id

        config = run_smed(config, t0)

        row = {
            "run_id": run_id,
            "mission": config["mission"],
            "query_length": config["query_length"],
            "reference_length": config["reference_length"],
            "slide_length": config["slide_length"],
            "ew_metrics": config.get("ew_metrics", {}),
            "config": config
        }

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(log_dir / f"{exp_id}.csv", index=False)

    meta = {
        "exp_id": exp_id,
        "timestamp": time_id,
        "n_combinations": len(combinations),
        "param_grid": param_grid
    }
    save_json(meta, log_dir / f"{exp_id}")

    print(f"All runs finished")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
