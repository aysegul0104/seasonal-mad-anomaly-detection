"""Seasonal MAD anomaly detection on a synthetic time series.

This module implements two detectors for additive anomalies in a 1-D
seasonal signal and compares them on a chronological (no-shuffle) test
split:

* **Algorithm** – seasonal decomposition via a linear trend plus a
  sinusoidal seasonal component (frequency from FFT, amplitudes from
  least-squares), followed by robust MAD thresholding on the residuals.
* **Baseline** – a centered 7-point moving average; anomalies are flagged
  when the absolute deviation from the moving average exceeds
  3 × std of the training deviations.

Both detectors use only training-split statistics for their thresholds,
preventing leakage.  Metrics (precision, recall, F1) are computed on the
last 25 % of the series.
"""

from __future__ import annotations

import math

import numpy as np

from data import make_dataset


def _fit_trend_and_seasonal(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Fit a linear trend and a sinusoidal seasonal component.

    The dominant frequency is identified from the FFT of the
    detrended signal.  Amplitudes for sin and cos at that frequency are
    then estimated by least-squares.  Returns the fitted seasonal + trend
    values at the given time points.
    """
    n = len(t)

    # --- linear trend via polyfit ---------------------------------------
    coeffs = np.polyfit(t, x, 1)
    trend = np.polyval(coeffs, t)

    # --- detrend for FFT -------------------------------------------------
    detrended = x - trend

    # FFT to find dominant frequency (ignore DC and Nyquist)
    fft_vals = np.fft.rfft(detrended)
    freqs = np.fft.rfftfreq(n, d=1.0)
    # Exclude DC (index 0); also exclude the very last bin if n is even
    # (Nyquist) to avoid ambiguity.
    search = slice(1, len(fft_vals) - (1 if n % 2 == 0 else 0))
    peak_idx = 1 + int(np.argmax(np.abs(fft_vals[search])))
    if peak_idx >= len(fft_vals):
        peak_idx = len(fft_vals) - 1
    dominant_freq = freqs[peak_idx]

    # --- least-squares for sin/cos amplitudes ----------------------------
    if dominant_freq > 0:
        sin_basis = np.sin(2.0 * np.pi * dominant_freq * t)
        cos_basis = np.cos(2.0 * np.pi * dominant_freq * t)
        A = np.column_stack([sin_basis, cos_basis])
        # Solve for amplitudes
        coeffs_sc, *_ = np.linalg.lstsq(A, detrended, rcond=None)
        seasonal = A @ coeffs_sc
    else:
        seasonal = np.zeros_like(t)

    return trend + seasonal


def _mad_threshold(residuals: np.ndarray) -> float:
    """Compute the robust MAD-based threshold: 3 * 1.4826 * MAD."""
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    return 3.0 * 1.4826 * mad


def _moving_average(x: np.ndarray, window: int = 7) -> np.ndarray:
    """Centered moving average with edge handling (truncated windows)."""
    n = len(x)
    half = window // 2
    ma = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        ma[i] = np.mean(x[lo:hi])
    return ma


def _precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute precision, recall, and F1 for binary labels."""
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def run_experiment(seed: int = 42, n_samples: int = 256) -> dict:
    """Run the seasonal-MAD anomaly detection experiment.

    Parameters
    ----------
    seed : int
        Random seed for dataset generation.
    n_samples : int
        Number of time points (must be >= 32).

    Returns
    -------
    dict
        JSON-serializable dict with keys ``n_samples``, ``metrics``,
        ``baseline_metrics``, and ``explanation``.
    """
    X, y = make_dataset(seed=seed, n_samples=n_samples)
    n = len(X)

    # Chronological split: first 75 % train, last 25 % test.
    split = int(math.ceil(0.75 * n))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    t_train = np.arange(split, dtype=np.float64)
    t_test = np.arange(split, n, dtype=np.float64)

    # ==================== Algorithm: seasonal MAD ========================
    fitted = _fit_trend_and_seasonal(t_train, X_train)
    residuals_train = X_train - fitted

    # Extend the fit to test points using the same model
    fitted_test = _fit_trend_and_seasonal(t_test, X_test)
    # But we should use the model parameters learned on train only.
    # Re-fit on train, then predict on test.
    coeffs = np.polyfit(t_train, X_train, 1)
    trend_test = np.polyval(coeffs, t_test)
    detrended_train = X_train - np.polyval(coeffs, t_train)
    fft_vals = np.fft.rfft(detrended_train)
    freqs = np.fft.rfftfreq(split, d=1.0)
    search = slice(1, len(fft_vals) - (1 if split % 2 == 0 else 0))
    peak_idx = 1 + int(np.argmax(np.abs(fft_vals[search])))
    if peak_idx >= len(fft_vals):
        peak_idx = len(fft_vals) - 1
    dominant_freq = freqs[peak_idx]

    if dominant_freq > 0:
        sin_basis_train = np.sin(2.0 * np.pi * dominant_freq * t_train)
        cos_basis_train = np.cos(2.0 * np.pi * dominant_freq * t_train)
        A_train = np.column_stack([sin_basis_train, cos_basis_train])
        coeffs_sc, *_ = np.linalg.lstsq(A_train, detrended_train, rcond=None)
        sin_basis_test = np.sin(2.0 * np.pi * dominant_freq * t_test)
        cos_basis_test = np.cos(2.0 * np.pi * dominant_freq * t_test)
        A_test = np.column_stack([sin_basis_test, cos_basis_test])
        seasonal_test = A_test @ coeffs_sc
    else:
        seasonal_test = np.zeros_like(t_test)

    fitted_test = trend_test + seasonal_test
    residuals_test = X_test - fitted_test

    threshold = _mad_threshold(residuals_train)
    pred_test = (np.abs(residuals_test) > threshold).astype(np.float64)

    metrics = _precision_recall_f1(y_test, pred_test)

    # ==================== Baseline: moving average =======================
    ma_train = _moving_average(X_train, window=7)
    deviations_train = X_train - ma_train
    std_dev = float(np.std(deviations_train))
    baseline_threshold = 3.0 * std_dev

    ma_test = _moving_average(X_test, window=7)
    deviations_test = X_test - ma_test
    pred_test_baseline = (np.abs(deviations_test) > baseline_threshold).astype(np.float64)

    baseline_metrics = _precision_recall_f1(y_test, pred_test_baseline)

    return {
        "n_samples": int(n),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "explanation": (
            "Chronological 75/25 train/test split (no shuffling). "
            "Algorithm: linear trend + FFT-based sinusoidal seasonal fit on train, "
            "MAD threshold (3 * 1.4826 * MAD) on train residuals. "
            "Baseline: centered 7-point moving average, 3*std threshold on train "
            "deviations. Metrics computed on test split only."
        ),
    }
