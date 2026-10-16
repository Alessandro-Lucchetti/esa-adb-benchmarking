import numpy as np
import math
import numba as nb
from rocket_fft import numpy_like

# enable numpy.fft.* inside njit
numpy_like()

@nb.njit
def fft_convolve(a, b, valid_mode: nb.boolean):
    la = a.size
    lb = b.size
    n = la + lb - 1

    # next power of two
    size = 1
    while size < n:
        size <<= 1

    # FFT convolution via rocket-fft
    fa = np.fft.rfft(a, size)
    fb = np.fft.rfft(b, size)
    fc = fa * fb
    full = np.fft.irfft(fc, size)[:n]

    if valid_mode:
        # valid slice indices
        if la >= lb:
            start = lb - 1
            stop = la
        else:
            start = la - 1
            stop = lb
        return full[start:stop]
    else:
        return full


@nb.njit
def mass(x, y):
    """
    Compute the distance profile of x in y using MASS (Mueen’s Algorithm):

    Accumulation is done in float64. The term ``sum_x2 + sum_y2 - 2*dot`` is a small
    difference of large, near-equal values whenever the query nearly matches a reference
    window (true distance ~ 0). In float32 this catastrophic cancellation makes near-zero
    distances depend on the platform / FFT-library build, which is NOT reproducible: the
    same code gave channel_66 = 226 (rocket_fft 0.3.1) vs 399 (rocket_fft 0.2.5). float64
    is numerically exact (matches a direct non-FFT computation) and identical across every
    OS / numpy / numba / rocket_fft combination. Do NOT downgrade these dtypes to float32.
    """
    m = x.size
    n = y.size
    M = n - m + 1
    if M <= 0:
        raise ValueError("len(y) must be at least len(x)")

    # promote inputs to float64 so the whole profile (incl. the FFT cross-term) is double
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    # 1) constant term sum(x_i^2) (O(m))
    sum_x2 = 0.0
    for i in range(m):
        sum_x2 += x[i] * x[i]

    # 2.1) sliding sums of y^2 via prefix‐sum (O(n))
    # prefix[0]=0; prefix[i]=sum_{j< i} y[j]^2
    prefix = np.empty(n + 1, dtype=np.float64)
    prefix[0] = 0.0
    for i in range(n):
        prefix[i + 1] = prefix[i] + y[i] * y[i]

    # 2.2) window sums of y^2
    sum_y2 = np.empty(M, dtype=np.float64)
    for k in range(M):
        sum_y2[k] = prefix[k + m] - prefix[k]

    # 3) cross‐term via FFT convolution
    #    dot[k] = sum_{i=0..m-1} x[i]*y[k+i]
    # nb: we need convolution(y, reverse(x))
    # reverse(x)
    x_rev = np.empty(m, dtype=np.float64)
    for i in range(m):
        x_rev[i] = x[m - 1 - i]
    dot = fft_convolve(y, x_rev, True)  # length M

    # 4) assemble d2
    d2 = np.empty(M, dtype=np.float64)
    for k in range(M):
        d2_k = sum_x2 + sum_y2[k] - 2.0 * dot[k]
        if d2_k < 0.0:  # prevent negative values!
            d2_k = 0.0
        d2[k] = d2_k

    return d2


@nb.njit(fastmath=True)
def best_match(x, y):
    """
    Compute the minimum Euclidean distance between x and all
    valid-length windows of y, returning (d_min, idx_min).
    """
    # 1) compute distance profile d2[k]
    d2 = mass(x, y)
    M = d2.shape[0]

    # 2) find the minimum in d2 - i.e., the best match
    best_d2 = d2[0]
    best_k = 0
    for k in range(1, M):
        if d2[k] < best_d2:
            best_d2 = d2[k]
            best_k = k

    # 3) convert back to squared Euclidean distance
    return math.sqrt(best_d2), best_k