# Third-party notices

This repository includes code adapted from the projects below. Their licenses apply
to those files in addition to the repository [LICENSE](LICENSE).

## ESA-ADB / TimeEval evaluation metrics

- Source: https://github.com/kplabs-pl/ESA-ADB (`timeeval/metrics/`), itself built on
  TimeEval (https://github.com/HPI-Information-Systems/TimeEval)
- Files: `src/tsadbench/eval/metrics/ESA_ADB_metrics.py`, `latency_metrics.py`,
  `ranking_metrics.py`, `metric.py`, `utils.py` (adapted)
- License: MIT

```
Copyright (c) 2024 Airbus
Copyright (c) 2024 KP Labs
Copyright (c) 2020-2021 Phillip Wenig and Sebastian Schmidl
```

## Affiliation-based metrics

- Source: https://github.com/ahstat/affiliation-metrics-py
- Files: `src/tsadbench/eval/metrics/affiliation_based_metrics_repo/` (vendored, with its
  own `LICENSE`)
- License: MIT

```
Copyright (c) 2022 Alexis Huet and others
```

## Matplotlib annotated heatmap example

- Source: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
- Files: `src/tsadbench/viz/heatmap.py` (adapted)
- License: Matplotlib license (PSF-based, BSD-compatible)

```
Copyright (c) 2012- Matplotlib Development Team
```

## MIT license text

The MIT-licensed components above are distributed under these terms:

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Dataset

The ESA Anomaly Detection Benchmark dataset (https://doi.org/10.5281/zenodo.12528696) is
licensed under CC BY 3.0 IGO and is **not** redistributed here; download it from the
original record. `data/metadata.json` lists channel subsets and periods derived from it, and
`results/qgpr_predictions.zip` holds detector labels on its timestamps.
