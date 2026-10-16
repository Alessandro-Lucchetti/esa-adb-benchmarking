# SMED Benchmark
This folder contains the code to run the SMED algorithm on ESA-Mission1. Copy the inputs (ESA-Mission1 raw parquets, labels, metadata) from the repository `data/` folder, then run:
```bash
chmod +x prepare_data.sh run.sh
./prepare_data.sh        # or ./prepare_data.sh <data_dir>
./run.sh
```
`data/` must hold the converted ESA-ADB dataset (see the main README, `setup/convert_zips_to_parquet.py`).
This script should work on any linux machine with Docker installed. It took about 30 minutes to run it on an i9 14900.

That's a Bash script that create the docker container for the algorithm, then runs the training/evaluation for both the light and full set. 
We didn't include the evalutaion metrics (heavy on time and dependencies); instead, the script checks the flagged anomalies against a precomputed checksum to make sure the algorithm is working as in the machine used to generate the paper results.
The checksums come from that reference machine: the kernel here works in float32, whose FFT results depend on CPU, OS and library build, so on other hardware a few channels can report `FAIL` because a handful of near-threshold windows flip. That is expected and does not affect the timings.
After the run, you should see a summary of results in the terminal and a .csv file with detailed timing results (in seconds) in the `output` folder. The image is automatically deleted after the run.
