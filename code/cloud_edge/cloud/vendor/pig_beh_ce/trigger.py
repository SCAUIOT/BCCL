"""Adaptive cloud-side training trigger via activity-level change detection.

Instead of training at every stage, the cloud monitors the non-lie pig ratio
(``a_t``) arriving from the edge.  When the *relative* change exceeds the
historical mean, the cloud triggers a re-training round because the incoming
frames contain behaviour patterns that are meaningfully different from the
recent past.

Core algorithm (no fixed thresholds — everything is self-calibrating):

    μ_t   = mean(a_{t-k:t})           # adaptive sliding-window baseline
    ã_t   = a_t / (μ_t + ε)           # normalised activity (removes env drift)
    c_t   = |ã_t - ã_{t-1}|           # per-frame change signal
    ĉ     = mean(c_history)           # historical change baseline
    TRIGGER when  c_t > ĉ             # current change exceeds normality
"""

from __future__ import annotations

import numpy as np


def compute_adaptive_baseline(
    a_values: np.ndarray,
    window: int = 100,
    epsilon: float = 0.01,
) -> np.ndarray:
    """Compute the adaptive sliding-window baseline μ_t for each frame.

    μ_t = mean(a_{t-k:t})  where k = min(window, t+1)

    Parameters
    ----------
    a_values : (N,) array
        Non-lie ratios per frame.
    window : int
        Sliding window size for the baseline.
    epsilon : float
        Floor to avoid division by zero.

    Returns
    -------
    mu : (N,) array
        Adaptive baseline for each frame.
    """
    n = len(a_values)
    a = np.asarray(a_values, dtype=np.float64)
    mu = np.empty(n, dtype=np.float64)
    for t in range(n):
        start = max(0, t - window + 1)
        mu[t] = np.mean(a[start:t + 1])
        if mu[t] < epsilon:
            mu[t] = epsilon
    return mu


def compute_normalised_activity(
    a_values: np.ndarray,
    mu: np.ndarray,
    epsilon: float = 0.01,
) -> np.ndarray:
    """Compute ã_t = a_t / (μ_t + ε) — activity relative to local environment.

    Returns
    -------
    a_tilde : (N,) array
        Normalised activity values.
    """
    a = np.asarray(a_values, dtype=np.float64)
    return a / np.maximum(mu, epsilon)


def compute_change_signal(a_tilde: np.ndarray) -> np.ndarray:
    """Compute c_t = |ã_t - ã_{t-1}| — per-frame change signal.

    The first frame's change is defined as 0.

    Returns
    -------
    c : (N,) array
        Change signal per frame.
    """
    a_tilde = np.asarray(a_tilde, dtype=np.float64)
    c = np.empty_like(a_tilde)
    c[0] = 0.0
    if len(a_tilde) > 1:
        c[1:] = np.abs(np.diff(a_tilde))
    return c


def compute_historical_mean(c_values: np.ndarray) -> float:
    """Compute ĉ = mean(c_history) — the dynamic trigger threshold."""
    return float(np.mean(np.asarray(c_values, dtype=np.float64)))


def evaluate_trigger(
    a_values: np.ndarray,
    c_history: list[float] | None = None,
    window: int = 100,
    epsilon: float = 0.01,
) -> dict:
    """Full trigger evaluation for a batch of incoming a_t values.

    This is the main entry point for the cloud-side trigger.  Given a sequence
    of non-lie ratios from the edge, it computes the normalised activity,
    change signal, and decides whether to trigger re-training.

    Parameters
    ----------
    a_values : array-like
        Non-lie ratio per frame (from edge selection output).
    c_history : list[float] | None
        Historical change values from previous stages.  If None or empty,
        the current batch's mean change is used as the threshold (first
        trigger always fires when ``c_history`` is empty).
    window : int
        Adaptive baseline window (number of frames).
    epsilon : float
        Stability constant.

    Returns
    -------
    dict with keys:
        triggered : bool
            Whether training should fire.
        c_mean : float
            Current batch mean change (ĉ_current).
        c_history_mean : float
            Historical mean change (ĉ_history) — the threshold.
        a_tilde_mean : float
            Mean normalised activity in this batch.
        c_values : list[float]
            Per-frame change signals (for appending to history).
        a_tilde : list[float]
            Per-frame normalised activity values.
    """
    if a_values is None or len(a_values) == 0:
        return {
            'triggered': False,
            'c_mean': 0.0,
            'c_history_mean': 0.0,
            'a_tilde_mean': 0.0,
            'c_values': [],
            'a_tilde': [],
        }

    a = np.asarray(a_values, dtype=np.float64)

    # Adaptive baseline & normalised activity
    mu = compute_adaptive_baseline(a, window=window, epsilon=epsilon)
    a_tilde = compute_normalised_activity(a, mu, epsilon=epsilon)

    # Per-frame change signal
    c = compute_change_signal(a_tilde)

    c_mean = float(np.mean(c))
    a_tilde_mean = float(np.mean(a_tilde))

    # ── Dynamic threshold from history ────────────────────────────
    if c_history is None:
        c_history = []
    if len(c_history) == 0:
        # No history yet → trigger on the first batch
        c_history_mean = c_mean * 1.01  # slightly above to avoid spurious 2nd trigger
    else:
        c_history_mean = float(np.mean(c_history))

    triggered = bool(c_mean > c_history_mean)

    return {
        'triggered': triggered,
        'c_mean': c_mean,
        'c_history_mean': c_history_mean,
        'a_tilde_mean': a_tilde_mean,
        'c_values': c.tolist(),
        'a_tilde': a_tilde.tolist(),
    }
