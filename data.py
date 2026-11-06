"""Synthetic seasonal time-series dataset with injected anomalies.

The dataset is a 1-D signal composed of:
  * a linear trend,
  * a dominant sinusoidal seasonal component,
  * small Gaussian noise,
  * a sparse set of injected anomalies (additive spikes).

Assumptions / limitations
-------------------------
* The seasonal period is fixed at 12 samples (one "cycle" every 12 points).
* Anomalies are additive spikes whose magnitude is a fixed multiple of the
  noise standard deviation; they are not multiplicative or structural breaks.
* The signal is fully synthetic; no real-world data is used.
* ``make_dataset`` is deterministic for a given ``(seed, n_samples)`` pair and
  performs no disk I/O.
"""

from __future__ import annotations

import numpy as np


def make_dataset(seed: int = 42, n_samples: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Return a synthetic seasonal time series and a binary anomaly label array.

    Parameters
    ----------
    seed : int
        Random seed controlling both the noise and the anomaly placement.
    n_samples : int
        Number of time points.  Must be at least 32.

    Returns
    -------
    X : np.ndarray, shape (n_samples,)
        The observed signal values (float64).
    y : np.ndarray, shape (n_samples,)
        Binary labels: 1.0 where an anomaly was injected, 0.0 otherwise.

    Raises
    ------
    ValueError
        If ``n_samples < 32``.
    """
    if n_samples < 32:
        raise ValueError(f"n_samples must be >= 32, got {n_samples}")

    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64)

    # --- deterministic signal components ---------------------------------
    # Linear trend
    slope = 0.05
    trend = slope * t

    # Dominant sinusoidal seasonal component (period = 12 samples)
    period = 12.0
    seasonal = 2.0 * np.sin(2.0 * np.pi * t / period)

    # Small Gaussian noise
    noise_std = 0.3
    noise = rng.normal(0.0, noise_std, size=n_samples)

    clean = trend + seasonal + noise

    # --- inject anomalies ------------------------------------------------
    # Anomaly magnitude: 5 * noise_std  (well above 3 * 1.4826 * MAD threshold
    # for the residual distribution, which is roughly 3 * noise_std * 1.4826 /
    # 1.4826 ≈ 3 * noise_std for Gaussian residuals, so 5 * noise_std gives
    # a comfortable margin).
    anomaly_magnitude = 5.0 * noise_std

    # Place anomalies sparsely: roughly one every 20 samples, avoiding the
    # first and last 5 points so that edge effects in the moving-average
    # baseline do not coincide with injected anomalies.
    n_anomalies = max(1, n_samples // 20)
    # Candidate indices: skip the first 5 and last 5 points
    lo, hi = 5, n_samples - 5
    candidates = np.arange(lo, hi)
    anomaly_idx = rng.choice(candidates, size=n_anomalies, replace=False)
    anomaly_idx.sort()

    y = np.zeros(n_samples, dtype=np.float64)
    y[anomaly_idx] = 1.0

    # Additive spikes (random sign)
    signs = rng.choice([-1.0, 1.0], size=n_anomalies)
    X = clean.copy()
    X[anomaly_idx] += signs * anomaly_magnitude

    return X, y
