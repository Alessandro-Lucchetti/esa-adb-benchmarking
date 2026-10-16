# Telemetry characterization and archetype taxonomy

Quantitative description of the ESA-ADB Mission 1 telemetry and a label-free taxonomy of
its annotated windows, used to stratify detection performance by observable pattern
rather than by ESA semantic label. Three products:

- **sampling statistics** of the 58 target channels and their eight **sampling families**
  (S1-S8);
- the **taxonomy**: 3,586 labelled windows, 117 classes (113 by density + 4 by rule),
  12 traits per class, 8 rule-defined **archetypes** (A1-A8);
- **per-archetype recall** of detector runs.

Code: `src/tsadbench/taxonomy/`; entry point: `scripts/taxonomy.py`. Reference tables:
`data/ESA-Mission1/taxonomy/`. The class assignment needs the `taxonomy` extra
(`pip install -e ".[taxonomy]"`, i.e. the `hdbscan` package).

## Commands

```bash
python scripts/taxonomy.py all              # raw telemetry -> every table, in runs/taxonomy/
python scripts/taxonomy.py check            # compare runs/taxonomy/ with the reference tables
python scripts/taxonomy.py classify         # taxonomy only, from the reference window features
python scripts/taxonomy.py recall \
    smed=smed/predictions/smed_ESA-Mission1_phase_5_raw_full_q100_r750_s5_k1.5_p0.1 \
    ool=runs/oolimit_full/oolimit_ESA-Mission1_phase_5_raw_full_q100_r750_s5_k1.5_p0.1
```

`all` reads `data/ESA-Mission1/raw/` and `labels.csv` (see the README for the data setup)
and takes about 8 minutes on one core. `classify` without telemetry rebuilds the taxonomy
from the reference `window_features.csv` in a few seconds. `--out DIR` changes the output
folder.

## 1. Sampling statistics and families

Per channel, over the full 14-year span of the raw (non-resampled) stream, after removing
NaN and duplicated timestamps (`sampling.py`):

- inter-sample interval dt in whole seconds: median (`med_dt_s`; when sub-second bursts
  make it 0, the median of dt > 0, i.e. the base cadence), IQR, coefficient of variation,
  P95 over the median, and the three most frequent rounded values (`regimes`);
- gaps: dt above 10 x the median; count, total time, share of the span, longest one;
- monthly shift: fraction of adjacent-month pairs whose two-sample KS statistic exceeds
  0.2 (months with fewer than 100 samples break the chain, larger months are subsampled
  to 5,000 values with seed 0).

The 58 channels collapse into eight families of near-identical sampling behaviour. The
membership was set by inspection of the per-channel table and is fixed in
`SAMPLING_FAMILIES`; `sampling_families.csv` gives the range of each statistic within each
family (paper table of the sampling families).

| Family | Channels | Median dt |
|---|---|---|
| S1 | 41-46 | 30 s |
| S2 | 47-49, 57-60 | 30 s |
| S3 | 70-76 | 30 s, with 3-6 s bursts |
| S4 | 12-13, 17-20, 25-28, 34-37 | 90 s, 63.6% of samples in sub-second bursts |
| S5 | 14-16, 21-24, 29-33, 38-40 | 90 s |
| S6 | 50-52 | 90 s |
| S7 | 64-66 | 157 s |
| S8 | 61-63 | 900 s |

## 2. Windows and features

A window is one row of `labels.csv` on a target channel; pairs with several disjoint
intervals give one window per interval (`interval`, in time order). Of the 3,589 such
rows, 3 have no sample at all in the window or its context (`id_34` on channels 61-63,
900 s channels in an outage) and are dropped: 3,586 windows on 200 events and 13
channel groups.

Each window is read with a context of `clip(5 x duration, 1 d, 30 d)` before and
`clip(duration, 1 d, 7 d)` after. Event samples are always at full resolution; a context
side longer than 100,000 samples is min-max decimated on uniform time bins. Context
samples inside another labelled interval of the channel are excluded from the reference
statistics.

`features.py` computes 41 columns per window on a robust ruler (level: median of the
clean pre-context; unit: its IQR, with MAD and standard-deviation fallbacks): counts,
magnitude, persistence after the event, shape, variability, lag-difference energies at
1, 4 and 16 x the context cadence, sampling (rate ratio, dispersion, gaps) and frozen-data
indicators. The module docstring lists them by family.

## 3. Classes

`classify.py`:

1. **Signature.** 37 metrics (counts of context samples, the ruler and the context
   cadence are left out), heavy-tailed ones log-compressed, plus one applicability flag
   for each of the 36 metrics that are missing on some windows: 73 components. Metrics
   are standardized with centre and scale averaged over the per-group medians and IQRs
   (each channel group weighs the same), clipped to +/-5, missing values set to 0; the flag
   block carries 10% of the variance. PCA to 90% of the variance: 15 dimensions.
2. **Rule-based categories** for windows without shape: empty (no sample, 187 windows),
   single point (1-2 samples) up, down or within 1 IQR (375 windows).
3. **Density classes**: HDBSCAN within each channel group on the remaining windows
   (`min_cluster_size = max(3, 2%)`, `min_samples = 1`, `cluster_selection_epsilon = 0.30`,
   leaf selection), then classes whose durations in samples span more than 2.5 decades
   between P10 and P90 are split recursively at the largest log-duration gap (children of
   at least 4 windows). 113 classes, 2,751 windows.

Coverage: 3,313 of 3,586 windows (92.4%); the 273 HDBSCAN noise windows stay unassigned
(`U0`).

## 4. Traits, hierarchy and archetypes

`archetypes.py`. Each of the 117 units gets 12 traits, robust medians over its windows:
duration in samples and in hours (log), amplitude (log), direction (sign balance of the
windows leaving the 1 IQR band), recovery, impulsiveness (log), relative width, agitation
(IQR ratio, log), sampling-rate change (log), gap fraction, absence (share of empty
windows) and frozen fraction. Missing traits are imputed (median; minimum for amplitude
and impulsiveness), and the empty category has no amplitude, direction or texture.

Ward linkage on the robustly standardized traits (every unit weighs the same): fusion
heights grow smoothly down to 8 clusters (9 -> 8 at 7.53) and jump at 8 -> 7 (9.30), so
K = 8 (cut at 8.42). The archetypes are defined by threshold rules applied in order of
precedence, first match wins:

| Rule | Archetype |
|---|---|
| absence >= 0.5 | silence |
| duration <= 0.9 (log10 samples, about 7 samples) | point-like |
| amplitude >= 1.0, direction <= -0.5, frozen >= 0.4 | frozen collapse |
| rate change >= 0.7 and amplitude <= -0.5 and recovery >= 0.5 | pure rate change |
| rate change >= 0.7 otherwise | regime switch |
| duration >= 1.0 (log10 hours) and amplitude >= -0.2 | sustained excursion |
| amplitude <= -0.5 and recovery >= 0.8 | quiet interval |
| anything else | short deviation |

The rules agree with the K = 8 Ward cut on 97 of 117 units (each Ward cluster mapped to
the archetype holding most of its units). Archetypes are numbered by decreasing window
count and units are coded C1..C117 by archetype block, then by dendrogram leaf order:

| Id | Archetype | Codes | Windows | Events |
|---|---|---|---|---|
| A1 | sustained excursion | C1-C48 | 1,303 | 124 |
| A2 | short deviation | C49-C78 | 939 | 111 |
| A3 | point-like | C79-C83 | 535 | 66 |
| A4 | silence | C84 | 187 | 38 |
| A5 | regime switch | C85-C97 | 144 | 7 |
| A6 | quiet interval | C98-C108 | 102 | 14 |
| A7 | pure rate change | C109-C113 | 58 | 5 |
| A8 | frozen collapse | C114-C117 | 45 | 34 |

## 5. Per-archetype recall

`recall.py`. A window counts as detected when the run's binary output, held constant
between its samples, is 1 at any instant of the labelled interval; the rule also covers
zero-duration labels. Only windows starting in the test period (2007-2013) are scored,
on the channels of the run. Runs are read in the standard layout
(`labels/channel_NN.parquet`), so any detector in this repository or ingested with
`scripts/ingest_*.py` can be scored. The A1-A8 row excludes unassigned windows.

Full target-channel set, Phase 5 runs, hits over test windows (paper Tables 8 and 11; no
A7 window falls in the test period). Q-GPR is the reported configuration (quantile
detector, max threshold and CUSUM retraining); the z-normalized variant is the
`damp_znorm` run at p = 1.0 (paper p = 0.01), its best full-set operating point.

| Archetype | Windows | Events | SMED | Q-GPR | Telemanom-ESA-Pruned | Robust OOL | z-norm, p = 1.0 |
|---|---|---|---|---|---|---|---|
| A1 sustained excursion | 583 | 64 | 241 | 166 | 492 | 53 | 4 |
| A2 short deviation | 424 | 60 | 62 | 88 | 294 | 22 | 0 |
| A3 point-like | 328 | 41 | 96 | 136 | 228 | 86 | 1 |
| A4 silence | 109 | 23 | 43 | 40 | 83 | 42 | 0 |
| A5 regime switch | 14 | 1 | 14 | 14 | 14 | 14 | 0 |
| A6 quiet interval | 76 | 10 | 1 | 1 | 39 | 0 | 0 |
| A8 frozen collapse | 13 | 2 | 13 | 13 | 13 | 13 | 6 |
| unassigned (U0) | 161 | 34 | 58 | 42 | 123 | 48 | 0 |
| A1-A8 | 1,547 | 90 | 470 (0.304) | 458 (0.296) | 1,163 (0.752) | 230 (0.149) | 11 (0.007) |

## Reference tables

`data/ESA-Mission1/taxonomy/`, all produced by `scripts/taxonomy.py`:

| File | Rows | Content |
|---|---|---|
| `sampling_stats.csv` | 58 | per-channel sampling statistics and family |
| `sampling_families.csv` | 8 | range of each statistic within each family |
| `window_features.csv` | 3,586 | window keys, channel group and the 41 features |
| `units.csv` | 117 | code, unit, group, windows, events, archetype, Ward cluster, leaf position, 12 traits |
| `windows.csv` | 3,586 | window keys, labelled interval, group, ESA category, unit, code, archetype |
| `archetypes.csv` | 8 | code range, units, windows, events, groups, ESA category shares |
| `dendrogram.csv` | 116 | Ward linkage in scipy format (left, right, height, size) |
| `taxonomy.json` | | signature size, coverage, fusion heights, cut, rules vs Ward agreement |

`windows.csv` is the input of `recall`. Unit rows in `units.csv` follow the linkage order
(density classes by label, then the four rule-based categories).

## Provenance

Port of the analysis scripts written by L. Brancato for the paper revision, restructured
as a package with identical numerics: on the same raw channels the port reproduces the
original sampling table exactly and the window features to the last digits (relative
difference below 1e-11), and the taxonomy built from either feature table is the same
(classes, codes and archetypes of every window; traits and linkage heights within 1e-12).
One window, `id_134` on channel 64, is constant over its whole interval: its kurtosis and
skewness are undefined (NaN) here, while the original run produced round-off values.
Not ported: the conversion from the ESA-ADB resampled
multivariate files (the channel files used turned out to be the raw ones), the per-window
segment files (windows are cut in memory), an intermediate merge of classes into
families that does not enter the archetypes, and the UMAP projection and MATLAB scripts
of the figures.
