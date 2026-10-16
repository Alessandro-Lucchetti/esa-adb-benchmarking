# esa-adb-benchmarking: SMED on the ESA Anomaly Detection Benchmark

Code for the paper *Evaluating scalable similarity search and gaussian process regression
for satellite telemetry anomaly detection on the ESA-ADB benchmark* (L. Brancato,
A. Lucchetti, Y. Kimura, N. Omata, M. Giglio, S. Tsutsumi, F. Cadini), Aerospace Science
and Technology 180 (2027) 113957, <https://doi.org/10.1016/j.ast.2026.113957>.

**SMED** (Sliding Minimum Euclidean Distance) is a lightweight, model-free detector for
satellite telemetry. For each short *query* window of a channel it computes the minimum
Euclidean distance (MASS / Matrix Profile, FFT-accelerated) to every equal-length window
of the immediately preceding *reference* window; a window with no similar recent
precedent is flagged. A warm-up on anomaly-free training data calibrates a per-channel
inter-percentile-range (Tukey fence) threshold. Distances are raw, not z-normalized, so
level and amplitude shifts count as anomalies.

The Python package is called `tsadbench`. The repository contains:

- the SMED implementation (`src/tsadbench/detectors/smed/`);
- the ESA-ADB evaluation suite: corrected event-wise, subsystem- and channel-aware,
  alarming precision, ADTQC and affiliation-based metrics (`src/tsadbench/eval/`);
- the quantitative characterization of the telemetry (sampling statistics and families)
  and the label-free archetype taxonomy of the annotated windows, with per-archetype
  recall scoring and reference tables (`src/tsadbench/taxonomy/`,
  [`docs/taxonomy.md`](docs/taxonomy.md));
- the baselines and ablations of the paper: robust out-of-limit (OOL) checking, the
  z-normalized DAMP distance ablation, and ingestion of the published
  Telemanom-ESA-Pruned predictions;
- standalone, Dockerized single-core timing packages for SMED and OOL.

The enhanced Q-GPR implementation is not included (JAXA security policy). The ESA-ADB
dataset is not redistributed here.

## Install

Python 3.10 or later. With conda:

```bash
make env                 # creates the `tsadbench` env and installs the package (editable)
conda activate tsadbench
```

or in an existing environment:

```bash
pip install -e ".[dev,taxonomy]"     # add ",viz" for the interactive plotting tools
```

The numeric stack (numpy, numba, rocket-fft, pandas, pyarrow) is pinned in
`pyproject.toml` to the versions behind the regression checksums.

## Data

Download `ESA-Mission1.zip` from the ESA Anomalies Dataset record
(<https://doi.org/10.5281/zenodo.12528696>, CC BY 3.0 IGO) and unzip it. Then:

```bash
mkdir -p data/ESA-Mission1
cp <unzipped>/ESA-Mission1/{labels.csv,channels.csv,anomaly_types.csv} data/ESA-Mission1/
python setup/convert_zips_to_parquet.py --mission ESA-Mission1 \
    --zip-dir <unzipped>/ESA-Mission1/channels --channels full
```

The converter writes `data/ESA-Mission1/raw/channel_NN.parquet` (float32, raw irregular
sampling). `data/metadata.json` (tracked) holds the lightweight / full channel lists, the
training durations of the five phases and the test period.

## Run SMED

```bash
# Mission 1, full target-channel set (58 channels), Phase 5, complete metric suite
python scripts/run.py configs/experiments/mission1_full_phase5.yaml --set experimental_mode=false

# lightweight subset (channels 41-46)
python scripts/run.py configs/experiments/mission1_light_phase5.yaml --set experimental_mode=false
```

With `experimental_mode=false` a run is written to `smed/predictions/<run_name>/`:

```
config.json                  parameters, channels, runtimes and all metric blocks
labels/channel_NN.parquet    timestamp, label (0/1)
labels/event.parquet         predicted events (union over channels)
scores/channel_NN.parquet    timestamp, score (distance), threshold
```

With `experimental_mode=true` (the default in those configs, used for sweeps) only the
event-wise metrics are computed and the run goes to `smed/experiments/<exp_id>/`. Run
names encode the configuration:
`smed_<mission>_phase_<phase>_<signal>_<channels>_q<q>_r<r>_s<s>_k<k>_p<p>`. Metric values
in `config.json` are stored as 3-decimal strings.

## Reproducing the paper

One command recomputes the numbers of the paper, from the download of the dataset to a
report that compares each of them with the published value:

```bash
reproduce/run.sh                   # Docker, pinned environment
python scripts/reproduce.py all    # or directly, in the tsadbench environment
```

It downloads ESA-Mission1 from Zenodo, runs SMED, OOL and the z-normalized ablation,
builds the taxonomy, scores the Q-GPR predictions (stored in the repository) and the
Telemanom-ESA-Pruned predictions (ESA-ADB), computes the per-archetype recall and checks
434 values of Tables 2, 5-8 and 10-14 and of the text against
`reproduce/paper_values.csv`. The report is written to `runs/reproduce/report.md`; stages,
options, requirements (8 GB of memory for Docker, 25 GB of disk; Windows set-up), run
time (about an hour with 8 cores and 8 GB) and the documented
differences are in [`docs/reproduce.md`](docs/reproduce.md).
The window-size sensitivity grids of Fig. 9 are left out (commands below).

The commands for each element of the paper:

The paper writes the threshold percentile `p` as a fraction, the code and run names in
percent: paper `p = 0.001` is `p0.1` here, `0.0025` is `p0.25`, `0.01` is `p1.0`, `0.25`
is `p25.0`. All results are on ESA-Mission1; "Phase 5" is the main setting (84 months of
training data).

| Paper | Content | How |
|---|---|---|
| Table 2 | sampling families | `python scripts/taxonomy.py sampling` |
| Tables 3-5 | 117 classes, traits, rules, 8 archetypes | `python scripts/taxonomy.py all`, then `python scripts/taxonomy.py check` against `data/ESA-Mission1/taxonomy/`; see [`docs/taxonomy.md`](docs/taxonomy.md) |
| Tables 6-7, SMED | Phase 5, lightweight and full sets | `scripts/run.py` as above (event-wise P / R / F0.5: light 0.999 / 0.462 / 0.811, full 0.982 / 0.681 / 0.902); see [numerical precision](#numerical-precision-and-differences-from-the-published-values) |
| Tables 6-7, robust OOL | Phase 5, p = 0.001 | `python scripts/limits_psweep.py light --full-metrics 0.1` and `... full ...`; see [`docs/oolimit_baseline.md`](docs/oolimit_baseline.md) |
| Tables 6-7, Telemanom-ESA-Pruned | quoted from the ESA-ADB paper | the full-set predictions published with ESA-ADB (`results/README.md` of <https://github.com/kplabs-pl/ESA-ADB>) are ingested for Table 8 with `python scripts/ingest_telemanom.py <full-sets zip or run dir> <out_dir>` |
| Table 8 | per-archetype recall, full set, Phase 5 | `python scripts/taxonomy.py recall smed=<run_dir> qgpr=<run_dir> telemanom=<run_dir> ool=<run_dir>` (A1-A8 hits out of 1,547: 470, 458, 1,163, 230) |
| Table 10 | raw vs z-normalized distance | `python scripts/damp_psweep.py light` and `... full` (default p list 0.1 1 3 5 10 25) |
| Table 11 | per-archetype recall of the z-normalized variant at p = 0.01 | `recall` on the `damp_znorm_..._full_..._p1.0` run (11 hits out of 1,547) |
| Table 13 | training length (Phases 1-5), lightweight set | SMED: `scripts/run.py configs/experiments/mission1_light_phase5.yaml --set phase=N experimental_mode=false`; OOL: `python scripts/limits_psweep.py light --phase N --full-metrics 0.1` |
| Table 14 | computational cost | `smed-benchmark/` and `ool-benchmark/` (see below) |
| Fig. 9(a), (b) | window-size grids | edit `param_grid` in `src/tsadbench/runner/sweep.py`, run `python -m tsadbench.runner.sweep`, then `python tools/smed_sensitivity.py` (needs LaTeX) |
| Fig. 9(c) | F0.5 against p, SMED and OOL | `python scripts/smed_psweep.py <set> ...` and `python scripts/limits_psweep.py <set> ...`, then `python tools/oolimit_sensitivity.py --smed` (commands in [`docs/oolimit_baseline.md`](docs/oolimit_baseline.md)) |

Q-GPR (Tables 6-8, 12, 13, Fig. 9(d) and the run-to-run variability of Section 4.3.1)
cannot be re-run from this repository: its implementation is not released. Its
predictions are in `results/qgpr_predictions.zip` (below), unpacked and scored by
`python scripts/reproduce.py external`, or by hand with
`python scripts/score_runs.py <run_dir>` and `scripts/taxonomy.py recall` like any other
run.

### Numerical precision and differences from the published values

The SMED results in the paper were computed with a float32 distance kernel. In float32 the
MASS distance profile suffers from cancellation near zero distance, so the labels depend on
the platform and on the FFT library build. The kernel in this repository accumulates in
float64: it is exact (it matches a direct, non-FFT computation) and gives bit-identical
labels across operating systems and library versions (`golden_checksums.json`), which makes
it the reliable choice for reproducibility. Small differences from the published values are
therefore expected. On Mission 1 the only one is the alarming precision of the lightweight
set in Table 6, 0.577 instead of 0.566; every other SMED value of Tables 6-8, 10, 11 and 13
is reproduced exactly.

## Q-GPR predictions

`results/qgpr_predictions.zip` (1 MB) holds the Q-GPR predictions of the paper and
unpacks at the repository root:

- `runs/qgpr/<config>_<set>/`: the four post-processing configurations (`qgpr_only`,
  `max_only`, `qgpr_max`, `qgpr_max_cusum`, the last one reported in the comparison tables)
  on the lightweight and full sets;
- `runs/qgpr_phase/phase<N>_light/`: the reported configuration in Phases 1-5;
- `runs/qgpr_seeds/seed_<S>_light/`: the 15 seeds of the run-to-run variability
  (Section 4.3.1).

Each run has `labels/channel_NN.parquet` (timestamp, label) on the Q-GPR prediction grid
and a `config.json` with the channels, the prediction period and the positives per
channel, without metrics. Every other result of the paper is recomputed by
`scripts/reproduce.py`.

## Tests and regression gate

```bash
make test                                   # fast tests; dataset-dependent ones skip without data/
pytest tests/test_golden_checksums.py       # re-run SMED and compare labels bit for bit
python scripts/golden.py verify smed/predictions golden_checksums.json   # runs not generated show as missing
```

`golden_checksums.json` stores the SHA-256 of every channel's int8 label array for the
paper runs (lightweight set Phases 1-5, full set Phase 5). SMED's distance profile is accumulated in float64, so labels are identical
across operating systems and library builds on the pinned stack.

## Timing benchmarks

`smed-benchmark/` and `ool-benchmark/` are dependency-light, Dockerized copies of the
detectors that time training and evaluation on a single core:

```bash
cd smed-benchmark
./prepare_data.sh        # copies metadata, labels and raw parquets from ../data
./run.sh                 # builds the image, runs light + full, writes output/benchmark_results.csv
```

Both packages check their labels against reference checksums from the machine used for
the paper. The SMED package keeps the original float32 kernel, so on other hardware a few
channels can report a mismatch; the timings are unaffected.

## Repository layout

```
src/tsadbench/        installable package
  detectors/smed/     SMED core (run_algorithm, warm_up, eval_channel) and distance kernel
  detectors/damp/     z-normalized distance variant for the ablation
  data/  eval/        dataset and split loaders; ESA-ADB metric suite and orchestrator
  runner/             typed YAML configs, run-name encoding, single run and grid sweep
  analysis/           per-class recall by ESA anomaly type
  taxonomy/           sampling statistics, window features, archetype taxonomy, per-archetype recall
  io/  viz/  postproc/
configs/              base and experiment YAMLs
scripts/              command-line entry points (runs, sweeps, scoring, ingestion, golden)
tools/                figure and exploration scripts (some need a display or LaTeX)
tests/                pytest suite, including the golden regression gate
reproduce/            Docker image, paper values and data checksums of scripts/reproduce.py
setup/                environment files and the ESA-ADB zip-to-parquet converter
smed-benchmark/       Dockerized timing package for SMED
ool-benchmark/        Dockerized timing package for the OOL baseline
figures/              sensitivity figures and their data
data/metadata.json    channel subsets, phase durations, test period
data/ESA-Mission1/    taxonomy reference tables (not part of ESA-ADB)
```

## License and citation

Code: MIT, see [LICENSE](LICENSE). Adapted third-party code (ESA-ADB / TimeEval metrics,
affiliation metrics, a Matplotlib example) is listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). If you use this code, please cite the
paper (see [CITATION.cff](CITATION.cff)) and the ESA-ADB benchmark:

> K. Kotowski et al., *European Space Agency Benchmark for Anomaly Detection in Satellite
> Telemetry*, arXiv:2406.17826, 2024.
