# OOL Benchmark
This folder contains the code to run the robust out-of-limit (OOL) baseline on ESA-Mission1, packaged exactly like the SMED benchmark. Copy the inputs (ESA-Mission1 raw parquets, labels, metadata) from the repository `data/` folder, then run:
```bash
chmod +x prepare_data.sh run.sh
./prepare_data.sh        # or ./prepare_data.sh <data_dir>
./run.sh
```
`data/` must hold the converted ESA-ADB dataset (see the main README, `setup/convert_zips_to_parquet.py`).
This script should work on any linux machine with Docker installed. It takes a few minutes in total; the algorithm itself is very cheap, most of the wall clock is reading the parquet files.

That's a Bash script that creates the docker container for the algorithm, then runs the training/evaluation for both the light and full set, in the same way as the SMED benchmark.
The algorithm (same protocol as SMED, Phase 5, p = 0.1, k = 1.5, window grid q100 / r750 / s5):
- training: for each channel, a two-sided Tukey fence `[Q_p - k*IPR, Q_{100-p} + k*IPR]` on the nominal samples of the warm-up window (first 84 months of the mission, samples inside labelled events removed);
- evaluation: every query window of the test period is flagged when its max exceeds the upper limit or its min falls below the lower limit.

All timings are measured with a monotonic clock (`time.perf_counter`), so they are not affected by adjustments of the system clock (NTP steps, VM time synchronisation); the script prints a warning at the end if the system clock moved during the run.

As for SMED, we didn't include the evaluation metrics (heavy on time and dependencies); instead, the script checks the flagged windows against precomputed checksums to make sure the algorithm produces the same output as on the machine used to generate the paper results.
After the run, you should see a summary of results in the terminal and a .csv file with detailed timing results (in seconds) in the `output` folder. The image is automatically deleted after the run.
