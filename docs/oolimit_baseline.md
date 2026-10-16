# Robust out-of-limit (OOL) baseline

Operational status-quo detector used as a baseline for SMED on ESA-ADB. It is plain
limit checking with robust, data-derived limits: a per-channel envelope is
estimated on nominal training values and a window is flagged as soon as any of
its samples leaves that envelope. There is no model and no similarity search,
so the comparison with SMED isolates what the sliding minimum-distance search
adds on top of the thresholding machinery the two methods share.

Entry point: `scripts/limits_psweep.py` (run name prefix `oolimit`).

## Algorithm

Everything below is per channel and deterministic.

### 1. Training data (same window as SMED's warm-up)

Take the channel's samples from the start of the mission up to
`train_durations[phase-1]` months later (phase 5: 84 months), never going past
the start of the test period. This is exactly the `forward` warm-up window used
by SMED (`warm_up` in `src/tsadbench/detectors/smed/core.py:71`).

Samples that fall inside a ground-truth event (from `data/<mission>/labels.csv`)
are removed. SMED masks whole warm-up windows that overlap an event; here the
masking is per sample because the envelope is estimated on values, not on
windowed distances.

### 2. Envelope (two-sided Tukey fence)

On the surviving nominal values compute the percentiles `Q_p` and `Q_{100-p}`
and the inter-percentile range `IPR = Q_{100-p} - Q_p`. The limits are

```
lo = Q_p       - k * IPR
hi = Q_{100-p} + k * IPR
```

with `k = 1.5` and `p` the envelope percentile. This is the same fence SMED
uses on its warm-up distances (`thr = Q_{100-p} + k * IPR`), applied two-sided
on raw sample values instead of one-sided on distances. `p = 0.1` (i.e.
`p = 0.001` in the fraction convention used for SMED in the paper) is SMED's
own default and the operating point reported in the tables; the sweep below
covers `p` from 0.01 to 25.

### 3. Test windows (same grid as SMED)

Test samples (from the start of the test period onward) are windowed with
SMED's grid, `q = 100`, `r = 750`, `s = 5`: window `j` starts at sample
`i = r + j * s` and covers `values[i : i + q]`; the number of windows is
`(n - q - r - 1) // s + 1` (see `run_algorithm`,
`src/tsadbench/detectors/smed/core.py:24`). The label of window `j` is stamped
at the timestamp of sample `i`, as SMED does. The first `r` samples of each
channel are therefore never labelled by either method.

For each window only the max and min are kept (`sliding_window_view` over the
channel, then indexed on the grid). These extrema do not depend on `p`, so
they are computed once and reused for every `p`.

### 4. Decision rule and score

```
label  = 1  if  max(window) > hi  or  min(window) < lo,  else 0
score  = max(max(window) - hi,  lo - min(window))        # > 0 iff out of limits
```

`scores/channel_NN.parquet` stores `score` with a constant `threshold = 0`, so
the score file is consistent with the label rule and can be re-thresholded
later if needed.

### 5. Events and metrics (shared pipeline)

Per-channel labels are written to `labels/channel_NN.parquet` in the same
format as SMED / Q-GPR / Telemanom. The evaluation orchestrator
(`metrics_light`, `src/tsadbench/eval/orchestrator.py:13`) takes the union of
the per-channel labels over time, converts it to a list of predicted events
(`build_predictions_event` in `src/tsadbench/io/misc.py`, saved as
`labels/event.parquet`) and scores it with the ESA-ADB suite in both
`all_events` and `anomalies_alone` modes:

- corrected event-wise precision / recall / F0.5 (always);
- with `--full-metrics` (the code path used for the SMED table runs, per-channel
  labels reloaded from disk): subsystem-aware and channel-aware precision /
  recall / F0.5, alarming precision, ADTQC after-ratio and score,
  affiliation-based precision / recall / F0.5.

The metric blocks are written back into the run's `config.json` together with
the fitted envelope of every channel (`"envelope": {channel: [lo, hi]}`) and the
detector timings (`detector_runtime`, shared p-independent pass, and
`threshold_runtime`, per p).

### Parameters

| name | value | note |
|---|---|---|
| `mission` | ESA-Mission1 | |
| `phase` | 5 | 84-month training window |
| `signal_mode` | raw | |
| `channels_mode` | light / full | 6 / 58 target channels |
| `warm_up_mode` | forward | training window from mission start |
| `query_length` q | 100 | window length (samples) |
| `reference_length` r | 750 | grid offset, kept only to match SMED's grid |
| `slide_length` s | 5 | grid stride |
| `k` | 1.5 | Tukey fence multiplier |
| `p` | 0.1 (tables); 0.01 to 25 (sweep) | envelope percentile |

The reference window is not used by the detector; `r` only fixes where the
first window starts so that OOL and SMED label the same timestamps.

## What the comparison with SMED isolates

Same training data, same event masking policy, same window grid, same fence
(`k`, `p`), same event-building and metrics. The only difference is the
quantity being thresholded:

- OOL: the raw sample values of the window (two-sided fence);
- SMED: the minimum Euclidean distance between the window and every
  equal-length window of the preceding reference window (one-sided fence).

So the gap between the two is attributable to the similarity search itself.

## Results: full ESA-ADB suite at p = 0.1 (all events, phase 5)

Rows follow the paper's Mission 1 result tables (all events, lightweight and full
sets). SMED / Q-GPR / Telemanom-ESA-Pruned columns are the published values,
reported here only for reading convenience.

### Lightweight set (channels 41 to 46)

| aspect | metric | OOL (p=0.1) | SMED | Q-GPR | Telemanom-ESA-Pruned |
|---|---|---|---|---|---|
| Event-wise | Precision | 0.995 | 0.999 | 0.976 | 0.999 |
| | Recall | 0.400 | 0.462 | 0.677 | 0.424 |
| | F0.5 | 0.767 | 0.811 | 0.897 | 0.786 |
| Channel-aware | Precision | 0.400 | 0.462 | 0.677 | 0.424 |
| | Recall | 0.400 | 0.449 | 0.505 | 0.275 |
| | F0.5 | 0.400 | 0.457 | 0.594 | 0.362 |
| Alarming precision | | 0.722 | 0.566 | 0.557 | 0.875 |
| ADTQC | After ratio | 0.000 | 0.133 | 0.136 | 0.143 |
| | Score | 0.129 | 0.187 | 0.286 | 0.197 |
| Affiliation-based | Precision | 0.685 | 0.716 | 0.821 | 0.711 |
| | Recall | 0.394 | 0.436 | 0.685 | 0.423 |
| | F0.5 | 0.597 | 0.635 | 0.789 | 0.626 |

Light-set best case (p = 1, F0.5 0.778): event-wise 0.995 / 0.415 / 0.778,
channel-aware 0.415 / 0.403 / 0.408, alarming precision 0.730, ADTQC 0.000 /
0.124, affiliation 0.693 / 0.409 / 0.609.

### Full set (58 target channels)

| aspect | metric | OOL (p=0.1) | SMED | Q-GPR | Telemanom-ESA-Pruned |
|---|---|---|---|---|---|
| Event-wise | Precision | 0.991 | 0.982 | 0.952 | 0.050 |
| | Recall | 0.385 | 0.681 | 0.670 | 0.870 |
| | F0.5 | 0.754 | 0.902 | 0.878 | 0.061 |
| Subsystem-aware | Precision | 0.374 | 0.681 | 0.645 | 0.395 |
| | Recall | 0.236 | 0.477 | 0.501 | 0.861 |
| | F0.5 | 0.326 | 0.613 | 0.594 | 0.436 |
| Channel-aware | Precision | 0.366 | 0.681 | 0.625 | 0.267 |
| | Recall | 0.214 | 0.416 | 0.297 | 0.725 |
| | F0.5 | 0.290 | 0.569 | 0.472 | 0.291 |
| Alarming precision | | 0.897 | 0.721 | 0.570 | 0.206 |
| ADTQC | After ratio | 0.118 | 0.177 | 0.286 | 0.463 |
| | Score | 0.321 | 0.538 | 0.546 | 0.684 |
| Affiliation-based | Precision | 0.687 | 0.864 | 0.842 | 0.649 |
| | Recall | 0.390 | 0.688 | 0.662 | 0.484 |
| | F0.5 | 0.596 | 0.822 | 0.798 | 0.607 |

Anomalies-alone blocks (not in the paper tables) are in every run's
`config.json`: light p=0.1 event-wise 0.995 / 0.862 / 0.966, full p=0.1
0.865 / 0.545 / 0.774.

Reading:

- Light set: the status quo is close to SMED (same precision, recall 0.400 vs
  0.462). Its detections are confined to the sample that crosses the limit,
  hence a high alarming precision (0.722 on light, second to Telemanom's 0.875;
  0.897 on full, the highest of the four methods).
- Full set: at equal precision (0.991 vs 0.982) SMED's recall is 0.681 vs
  0.385, and the gap widens on the diagnostic aspects (subsystem-aware F0.5
  0.326 vs 0.613, channel-aware 0.290 vs 0.569). The anomalies that limit
  checking misses stay inside the envelope and are only visible as a change of
  shape, which is what the similarity search picks up.
- ADTQC after-ratio is 0.000 on the light set: every OOL detection starts at or
  before the annotated onset. Both OOL and SMED stamp a window's label at its
  first sample, so a limit crossing anywhere in the 100-sample window is
  reported up to 99 samples early; with no model there is nothing that delays a
  detection past the onset.
- Cost: the detector pass (parquet loading included) takes 15 s (light) and
  64 s (full); re-thresholding for one `p` takes about 1 s (light) and 10 s
  (full). SMED: 4.0 / 26.5 min.

## Sensitivity to the envelope percentile p

`tools/oolimit_sensitivity.py` collects all `oolimit` runs and writes
`figures/oolimit_sensitivity_p_ESA-Mission1.{csv,pdf,png}`: a 1-D annotated
heatmap of the corrected event-wise F0.5 (all events), rows lightweight / full,
columns `p`, same style as the SMED / Q-GPR sensitivity figures. `p` is shown
as a fraction (0.001 = 0.1-th percentile), the paper's convention.

Corrected event-wise metrics, all events (P / R / F0.5, predicted events):

| p (percentile) | p (fraction) | light | full |
|---|---|---|---|
| **0.1** | **0.001** | 0.995 / 0.400 / 0.767 (36) | **0.991 / 0.385 / 0.754** (35) |
| 1 | 0.01 | **0.995 / 0.415 / 0.778** (36) | 0.009 / 0.923 / 0.011 (4040) |
| 3 | 0.03 | 0.995 / 0.415 / 0.778 (37) | 0.009 / 0.923 / 0.012 (3959) |
| 5 | 0.05 | 0.995 / 0.415 / 0.778 (34) | 0.009 / 0.923 / 0.011 (3926) |
| 10 | 0.1 | 0.824 / 0.446 / 0.705 (45) | 0.008 / 0.923 / 0.010 (3647) |
| 25 | 0.25 | 0.009 / 0.600 / 0.011 (4083) | 0.002 / 0.978 / 0.003 (2048) |

Reading:

- Light set: a flat plateau, F0.5 in [0.767, 0.778] for `p` from 0.1 to 5,
  then degradation and collapse at 25.
- Full set: only the SMED default `p = 0.1` works. For every `p >= 1` two
  quantized channels (`channel_64`, `channel_65`: `Q_p = Q_{100-p}` on their
  nominal training values, so the envelope has zero width) alarm on scattered
  bursts and produce about 4,000 spurious events. This fragility is part of
  the finding and was left as is (no per-channel minimum width or hand tuning),
  since the point of the baseline is naive limit checking.

A finer grid (0.01, 0.05, 0.25, 0.5, 0.75, 2, 7.5, 15, 20; `--prune` runs,
event-wise only, see How to run) does not change the picture (full-set collapse starts at `p = 0.25`, when
`channel_64` alone degenerates). Pass `--ps` to the plotting tool to include it.

### Same sweep on SMED (nominal q100 / r750 / s5, k = 1.5)

The identical grid was run on SMED at its nominal window sizes. `scripts/smed_psweep.py` computes the
warm-up and test distances once per channel (the p-independent part) and
re-thresholds them for every p, exactly as `damp_psweep.py` does for DAMP;
p = 0.1 reproduces the published SMED numbers bit for bit (light
0.999 / 0.462 / 0.811, full 0.982 / 0.681 / 0.902). Output:
`figures/oolimit_vs_smed_sensitivity_p_ESA-Mission1.{csv,pdf,png}` (heatmap,
rows OOL light / OOL full / SMED light / SMED full) and `..._lines.{pdf,png}`
(F0.5 vs p, log-x), from `python tools/oolimit_sensitivity.py --smed`.

Corrected event-wise metrics, all events (P / R / F0.5, predicted events):

| p (percentile) | p (fraction) | SMED light | SMED full | OOL light | OOL full |
|---|---|---|---|---|---|
| **0.1** | **0.001** | **0.999 / 0.462 / 0.811** (54) | **0.982 / 0.681 / 0.902** (84) | 0.767 | **0.754** |
| 1 | 0.01 | 0.768 / 0.615 / 0.732 (101) | 0.295 / 0.802 / 0.338 (303) | **0.778** | 0.011 |
| 3 | 0.03 | 0.191 / 0.677 / 0.223 (284) | 0.117 / 0.890 / 0.142 (792) | 0.778 | 0.012 |
| 5 | 0.05 | 0.144 / 0.754 / 0.172 (396) | 0.069 / 0.923 / 0.084 (1371) | 0.778 | 0.011 |
| 10 | 0.1 | 0.056 / 0.846 / 0.069 (1041) | 0.020 / 0.945 / 0.025 (4673) | 0.705 | 0.010 |
| 25 | 0.25 | 0.004 / 0.969 / 0.005 (14441) | 0.001 / 1.000 / 0.001 (38017) | 0.011 | 0.003 |

Reading:

- SMED is far more sensitive to p than OOL on the light set: F0.5 drops from
  0.811 to 0.732 at p = 1 and collapses (0.223) at p = 3; on the full set the
  drop is immediate (0.902 -> 0.338 at p = 1). The mechanism is the one the
  paper's SMED paragraph anticipates: the threshold
  `Q_{100-p} + k (Q_{100-p} - Q_p)` is meant to sit above the few benign outliers
  of the warm-up distance distribution; as p grows, `Q_{100-p}` moves from the
  tail into the bulk and the IPR shrinks at the same time, so the threshold falls
  into the nominal distance range and the false-alarm count explodes (54 events
  at p = 0.1, 14,441 at p = 25 on light; 84 -> 38,017 on full). Precision goes
  0.999 -> 0.768 -> 0.191 while recall only creeps up (0.462 -> 0.677).
- OOL on the light set is flat for p from 0.1 to 5 because the envelope shrinks
  only slowly with p (`channel_41`: [0.767, 0.855] at p = 0.1, [0.782, 0.841] at
  p = 5, [0.792, 0.830] at p = 25): the nominal values of these six channels
  sit well inside it and the anomalies are gross excursions far outside, so the
  fence only starts to bite on nominal data around p = 10 and collapses at 25.
- Consequence: p is a design constant for SMED ("very small, not zero"), and
  the sweep confirms the choice is not a tuned optimum but a requirement
  (anything above about 1 is unusable). For OOL, p = 0.1 is also the only value
  that works on the full set. Both methods are therefore reported at the same
  p = 0.001 (fraction) without per-method tuning.

## Sensitivity to the training length (mission phase)

Same protocol as the paper's training-length table (`tab:phase_stats_m1`):
p = 0.1 fixed, the envelope estimated on the first 3 / 10 / 21 / 42 / 84 months
of the mission (ESA-ADB Phases 1 to 5, which is what `--phase` selects; the
envelope window equals training + validation length of the benchmark), test
period fixed (2007 to 2014). `tools/oolimit_phase_sensitivity.py` collects the
runs and writes `figures/oolimit_sensitivity_phase_ESA-Mission1.csv` (paper
layout, published SMED / Q-GPR / Telemanom values carried along for reference)
and `..._runs.csv` (P / R / F0.5 and event counts of every OOL run).

Corrected event-wise F0.5, all events:

| Phase | Training | Validation | SMED | Q-GPR | Telemanom-ESA-Pruned | OOL light | OOL full |
|---|---|---|---|---|---|---|---|
| 1 | 9 weeks | 3 weeks | 0.724 | 0.302 | 0.227 | 0.778 | 0.000 |
| 2 | 8 months | 2 months | 0.786 | 0.767 | 0.311 | 0.778 | 0.019 |
| 3 | 18 months | 3 months | 0.786 | 0.856 | 0.776 | 0.767 | 0.036 |
| 4 | 39 months | 3 months | 0.803 | 0.883 | 0.776 | 0.767 | 0.565 |
| 5 | 81 months | 3 months | 0.811 | 0.898 | 0.786 | 0.767 | 0.754 |

(SMED, Q-GPR and Telemanom columns are the paper's, lightweight set; the OOL
full column is an extra, not in the paper's table.)

Reading:

- Lightweight set: OOL is essentially insensitive to the training length,
  0.778 (Phases 1 and 2) vs 0.767 (Phases 3 to 5). The six channels are
  stationary enough that three months already span their nominal range; the
  envelopes move by a few thousandths between phases (e.g. `channel_41`
  [0.774, 0.852] at Phase 1 vs [0.767, 0.855] at Phase 5), and the longer
  history only widens them slightly, costing one detected event (recall
  0.415 -> 0.400) at unchanged precision 0.995. So on this set OOL beats SMED at
  Phase 1 (0.778 vs 0.724) and is below it from Phase 2 on.
- Full set: OOL depends dramatically on the training length, 0.000 / 0.019 /
  0.036 / 0.565 / 0.754. Many of the 58 channels drift or change level over the
  mission, so a short envelope does not contain the later nominal range: at
  Phase 1 `channel_34` has envelope [0.169, 0.224] vs [-0.125, 0.398] at Phase 5
  and 84 % of its test windows are out of limits; twelve channels flag more than
  40 % of their windows and the union is a single event spanning the whole test
  period (P = 0, R = 1). At Phases 2 and 3 the survivors are channels 57 to 60
  and 47 (2,328 and 1,225 events); at Phase 4 a handful of channels (22, 31, 23,
  15, 39, 30) still flag 2 % of their windows (53 events, P = 0.633). Only the
  full 84-month envelope covers the nominal range of every channel.
- Contrast with SMED: SMED's warm-up only sets a threshold on distances, and
  the nominal reference is the immediately preceding 750 samples, so slow
  drifts and level changes are absorbed by the sliding reference rather than
  compared against a fixed historical range. This is why SMED is stable across
  phases (0.724 to 0.811 on light) while a limit checker needs to have seen the
  whole mission envelope. It is the strongest argument in this study for
  similarity search over static limits on a heterogeneous channel set.

## How to run

```bash
conda activate tsadbench          # or: conda run -n tsadbench --no-capture-output ...

# table operating points, complete ESA-ADB suite (light ~3 min, full ~27 min)
python scripts/limits_psweep.py light --full-metrics 0.1 1 3 5 10 25
python scripts/limits_psweep.py full  --full-metrics 0.1 1 3 5 10 25

# sensitivity grid, event-wise only, per-channel labels deleted after scoring
python scripts/limits_psweep.py light --prune 0.01 0.05 0.25 0.5 0.75 2 7.5 15 20
python scripts/limits_psweep.py full  --prune 0.01 0.05 0.25 0.5 0.75 2 7.5 15 20

# training-length sweep at p = 0.1 (phase 5 is the run above)
for ph in 1 2 3 4; do python scripts/limits_psweep.py light --phase $ph 0.1; done
for ph in 1 2 3 4; do python scripts/limits_psweep.py full  --phase $ph 0.1; done

# SMED on the same p grid, nominal windows (light ~7 min, full ~1 h; distances once, then re-thresholded)
python scripts/smed_psweep.py light 0.1 1 3 5 10 25
python scripts/smed_psweep.py full  0.1 1 3 5 10 25

python tools/oolimit_sensitivity.py           # OOL p sweep figure + csv under figures/
python tools/oolimit_sensitivity.py --smed    # OOL vs SMED p sweep (heatmap + line plot)
python tools/oolimit_phase_sensitivity.py     # training-length table
```

Outputs (gitignored) go to `runs/oolimit/` (light) and `runs/oolimit_full/`
(full), one folder per (phase, `p`); the SMED sweep writes the same layout to
`runs/smed_psweep[_full]/`:

```
runs/oolimit_full/oolimit_ESA-Mission1_phase_5_raw_full_q100_r750_s5_k1.5_p0.1/
  config.json               parameters, envelope per channel, timings, metrics (both modes)
  labels/channel_NN.parquet timestamp, label (int8)          [absent for --prune runs]
  labels/event.parquet      predicted events (StartTime, EndTime), union over channels
  scores/channel_NN.parquet timestamp, score, threshold (= 0) [absent for --prune runs]
```

The dataset must be in place under `data/ESA-Mission1/` (see the top-level
README).

## Where the code is

| what | where |
|---|---|
| OOL baseline: training window, event masking, envelope, windowing, labels, sweep, metrics | `scripts/limits_psweep.py` (`nominal_train_values`, `sweep`) |
| p sensitivity table and figure (OOL alone, or OOL vs SMED with `--smed`) | `tools/oolimit_sensitivity.py` |
| training-length (phase) table | `tools/oolimit_phase_sensitivity.py` |
| SMED p sweep at nominal windows (distances once, re-thresholded per p) | `scripts/smed_psweep.py` (thin wrapper around `damp_psweep.sweep` with `best_match`) |
| Dockerized timing package for the reference machine (mirror of `smed-benchmark/`) | `ool-benchmark/` (`ool_core/algorithm.py` frozen copy of the detector, `benchmark.py`, `prepare_data.sh` copies the inputs, `run.sh` runs it) |
| SMED warm-up window and grid the baseline mirrors | `src/tsadbench/detectors/smed/core.py` (`run_algorithm` l.24, `warm_up` l.71) |
| Run configuration (pydantic) and run-name encoding | `src/tsadbench/runner/config.py` (`RunConfig` l.20), `src/tsadbench/runner/naming.py` (`build_run_name` l.29) |
| Channel / split loading | `src/tsadbench/data/splits.py` (`Split` l.24) |
| Ground-truth events, metadata, event file | `src/tsadbench/io/misc.py` (`load_gt_events`, `load_metadata`, `load_predictions_dict`, `load_predictions_event`, `build_predictions_event`) |
| Metrics orchestrator (event-wise, channel/subsystem-aware, alarming precision, ADTQC, affiliation) | `src/tsadbench/eval/orchestrator.py` (`metrics_light` l.13) |
| ESA-ADB metric implementations | `src/tsadbench/eval/metrics/` |
| Heatmap helpers shared with the SMED / Q-GPR sensitivity figures | `src/tsadbench/viz/heatmap.py` |
| Sibling sweep the script is modelled on (DAMP z-norm ablation) | `scripts/damp_psweep.py` |
| Dataset metadata (train durations, test period) | `data/metadata.json` |
