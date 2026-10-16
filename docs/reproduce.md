# Automatic reproduction of the paper numbers

`scripts/reproduce.py` recomputes the numbers of the paper, from the download of the
ESA-ADB dataset to a report that compares each of them with the published value. The
Docker image in `reproduce/` gives it a pinned environment:

```bash
reproduce/run.sh                 # every stage; Linux, macOS or Git Bash on Windows
reproduce/run.sh report          # a single stage (any argument of scripts/reproduce.py)
python scripts/reproduce.py all  # the same without Docker, in the tsadbench environment
```

The repository is mounted at `/work`, so every run is written where the rest of the code
expects it and can be inspected or re-scored afterwards. Stages skip the runs already on
disk; `--force` recomputes them and `--skip STAGE` leaves a stage out of `all`. The
conversion of the dataset unpickles the channel files of the ESA archive, one more reason
to run it in the container.

## Stages

| Stage | What it does | Writes |
|---|---|---|
| `data` | downloads `ESA-Mission1.zip` from Zenodo (record 12528696, 3.8 GB, resumable, md5 checked), converts the 58 target channels to parquet and checks their content against `reproduce/data_checksums.json`; `--esa-zip` uses a copy already downloaded | `data/ESA-Mission1/` |
| `smed` | SMED on the lightweight set in Phases 1-5 and on the full set in Phase 5, complete metric suite, labels checked against `golden_checksums.json` | `smed/predictions/` |
| `ool` | robust out-of-limit baseline, same runs, p = 0.001 | `runs/oolimit*/` |
| `znorm` | z-normalized distance ablation, Phase 5, p = 0.001 to 0.25, both sets | `runs/damp_znorm*/` |
| `taxonomy` | sampling families, window features, classes and archetypes, checked against the reference tables of `data/ESA-Mission1/taxonomy/` | `runs/taxonomy/` |
| `external` | the Q-GPR predictions of the paper, unpacked from `results/qgpr_predictions.zip`, and the Telemanom-ESA-Pruned full-set predictions published by ESA-ADB, read from its archive without downloading the 5.9 GB file (`--telemanom-zip` for a local copy); both scored with the metric suite of this repository | `runs/qgpr*/`, `runs/telemanom/` |
| `recall` | per-archetype recall of SMED, Q-GPR, Telemanom, OOL and the z-normalized variant at p = 0.01 | `runs/reproduce/archetype_recall.csv` |
| `report` | compares every value of `reproduce/paper_values.csv` with the recomputed one | `runs/reproduce/report.md`, `report.csv` |

The window-size sensitivity grids of Fig. 9 are not part of it (hundreds of SMED runs);
the commands are in the README.

## Requirements

- Docker with 8 GB of memory. With that limit the heaviest stage, SMED on the full set,
  runs 4 processes and peaks at 5 GB; with more memory and cores the stages run more
  processes and finish sooner. `--workers N` caps the number of processes.
- About 25 GB of free disk in the folder of the repository.

### Windows

1. Install Docker Desktop with the WSL 2 engine (the default) and start it. Docker gets
   half of the RAM by default, so a PC with 16 GB is enough as it is: in Git Bash,
   `docker info | grep "Total Memory"` should show about 8 GiB or more. With less, create
   `C:\Users\<user>\.wslconfig` with

   ```ini
   [wsl2]
   memory=8GB
   ```

   then run `wsl --shutdown` in PowerShell and start Docker Desktop again.
2. Install Git for Windows and run everything from Git Bash, in a folder outside OneDrive
   (the run writes about 25 GB):

   ```bash
   mkdir -p /c/work && cd /c/work
   git clone https://github.com/Alessandro-Lucchetti/esa-adb-benchmarking.git
   cd esa-adb-benchmarking
   ./reproduce/run.sh
   ```

   `run.sh` mounts the folder in the container with the Windows path that Docker Desktop
   expects. If the run stops, the same command resumes it: completed runs and the
   downloaded dataset are kept.

## Run time and disk

A complete run from a fresh clone, Docker image build included, took 65 minutes on a
fresh Ubuntu 24.04 with the container limited to 8 cores and 8 GB of memory (Docker
Desktop on a 16 GB PC; 4 processes) and a 3 MB/s connection to Zenodo:

| Stage | Minutes |
|---|---|
| `data` (mostly the 3.8 GB download) | 28 |
| `smed` | 13 |
| `ool` | 2 |
| `znorm` | 9 |
| `taxonomy` | 6 |
| `external` | 2 |
| `recall`, `report` | 1 |

Fewer cores mean longer `smed` and `znorm` stages.
Plan for about 25 GB of disk: 8 GB for the dataset (the downloaded zip is kept in
`data/downloads/` so that `--force` does not download it again) and 16 GB for the runs,
most of it the distance scores of the z-normalized sweep.

## What is compared

`reproduce/paper_values.csv` lists 434 values of the paper: Tables 2, 5, 6, 7, 8, 10, 11,
12, 13 and 14, the taxonomy numbers of Section 2.2.2 (empty and single-point windows,
clusters, assignment share, neighbourhood purity, rules against the Ward cut, cadence of
the rate archetypes, low-excursion windows and events) and the run-to-run variability of
Q-GPR in Section 4.3.1. Each row has the value as printed, its number, a tolerance (half
a unit of the last printed digit, zero for counts) and the source it is recomputed from. A value is

- `PASS` when it is within the tolerance of the paper,
- `KNOWN` when it differs in a documented way (below) and matches the documented value,
- `DIFF` otherwise, and the exit code is 1,
- `MISSING` when its run is not on disk (a stage was skipped).

The Telemanom-ESA-Pruned rows of Tables 6, 7 and 13 are quoted from the ESA-ADB paper and
are not recomputed; its per-archetype recall in Table 8 is, from the published
predictions. Table 14 is checked on the number of decisions; the run times depend on the
machine and are measured by `smed-benchmark/` and `ool-benchmark/`.

## Documented differences

- Table 6, SMED, alarming precision of the lightweight set: 0.577 instead of 0.566. The
  published runs used a float32 distance kernel whose labels depend on the platform and
  the FFT build; with the float64 kernel of this repository one label sample of channel 43
  changes. Every other SMED value is reproduced exactly.
- Section 2.2.2, cadence of the regime-switch archetype (A5): the text says the sampling
  cadence is multiplied by 240 in all 144 A5 windows. It is 240 in 78 of them and 30 in
  the other 66; 240 is the median.
- Section 2.2.2, neighbourhood purity without the single-point archetype (A4): the text
  says the mean stays at 0.62. The mean over the 116 classes outside A4 is 0.629.
