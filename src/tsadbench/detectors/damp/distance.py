"""Z-normalized distance kernel for the DAMP variant.

SMED uses the raw (non-normalized) Euclidean distance (see detectors/smed/distance.py).
X-Lag-Amnesic DAMP uses the z-normalized Euclidean distance: each length-m subsequence is
mean-centered and scaled to unit variance before comparison, so it matches on shape
regardless of amplitude/offset. This is the single knob in the SMED-vs-DAMP comparison.

Closed form (Mueen MASS, z-normalized): for query x and each length-m window y[k:k+m],
    d2[k] = 2m (1 - corr_k),   corr_k = (dot_k - m*mu_x*mu_y_k) / (m*sigma_x*sigma_y_k)
with dot via FFT and sliding window mean/std. d2 in [0, 4m].

Constant subsequences (variance ~ 0) have no defined shape, so correlation is undefined.
We treat them explicitly: constant-vs-constant -> distance 0; constant-vs-varying -> distance
sqrt(2m) (maximally shape-dissimilar). This avoids the cancellation artifact where a
near-zero-variance window spuriously becomes a perfect (d=0) match.
"""
import math

import numba as nb
import numpy as np
from rocket_fft import numpy_like

from tsadbench.detectors.smed.distance import fft_convolve

numpy_like()

FLAT = 1e-12  # variance below this == constant subsequence


@nb.njit
def mass_znorm(x, y):
    m = x.size
    n = y.size
    M = n - m + 1
    if M <= 0:
        raise ValueError("len(y) must be at least len(x)")

    # query mean/variance
    mu_x = 0.0
    for i in range(m):
        mu_x += x[i]
    mu_x /= m
    var_x = 0.0
    for i in range(m):
        d = x[i] - mu_x
        var_x += d * d
    var_x /= m
    if var_x < 0.0:
        var_x = 0.0
    x_const = var_x < FLAT
    sigma_x = math.sqrt(var_x) if not x_const else 1.0

    # prefix sums of y and y^2 (float64 for stable sliding stats)
    prefix = np.empty(n + 1, dtype=nb.float64)
    prefix2 = np.empty(n + 1, dtype=nb.float64)
    prefix[0] = 0.0
    prefix2[0] = 0.0
    for i in range(n):
        prefix[i + 1] = prefix[i] + y[i]
        prefix2[i + 1] = prefix2[i] + y[i] * y[i]

    # cross term dot[k] = sum_i x[i]*y[k+i] via FFT (convolution with reversed query)
    x_rev = np.empty(m, dtype=nb.float32)
    for i in range(m):
        x_rev[i] = x[m - 1 - i]
    dot = fft_convolve(y, x_rev, True)  # length M

    max_d2 = 2.0 * m  # constant-vs-varying distance^2
    d2 = np.empty(M, dtype=nb.float32)
    for k in range(M):
        sum_y = prefix[k + m] - prefix[k]
        sum_y2 = prefix2[k + m] - prefix2[k]
        mu_y = sum_y / m
        var_y = sum_y2 / m - mu_y * mu_y
        if var_y < 0.0:
            var_y = 0.0
        y_const = var_y < FLAT

        if x_const or y_const:
            d2k = 0.0 if (x_const and y_const) else max_d2
        else:
            sigma_y = math.sqrt(var_y)
            corr = (dot[k] - m * mu_x * mu_y) / (m * sigma_x * sigma_y)
            if corr > 1.0:
                corr = 1.0
            elif corr < -1.0:
                corr = -1.0
            d2k = 2.0 * m * (1.0 - corr)
            if d2k < 0.0:
                d2k = 0.0
        d2[k] = d2k
    return d2


@nb.njit(fastmath=True)
def best_match_znorm(x, y):
    """Minimum z-normalized Euclidean distance of x to all length-len(x) windows of y."""
    d2 = mass_znorm(x, y)
    M = d2.shape[0]
    best_d2 = d2[0]
    best_k = 0
    for k in range(1, M):
        if d2[k] < best_d2:
            best_d2 = d2[k]
            best_k = k
    return math.sqrt(best_d2), best_k
