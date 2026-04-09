"""
ivt.py
------
Velocity-Threshold Identification (I-VT) algorithm with K-ratio optimization.

Reference:
    Salvucci & Goldberg (2000); Orioma et al. / Bhandari et al. (2026)
"""

import numpy as np
from .kratio import sweep_thresholds


# ========================
# I-VT Classifier
# ========================

def apply_ivt(point_velo, x_vals, y_vals, threshold):
    """
    Apply I-VT classification: samples with velocity < threshold -> fixation.

    Parameters
    ----------
    point_velo : np.ndarray
    x_vals, y_vals : np.ndarray
    threshold : float
        Velocity threshold in px/s.

    Returns
    -------
    dict with keys: x_fix, y_fix, x_sac, y_sac, classifier
    """
    min_len = min(len(point_velo), len(x_vals), len(y_vals))
    point_velo = point_velo[:min_len]
    x_vals     = x_vals[:min_len]
    y_vals     = y_vals[:min_len]

    x_fix, y_fix = [], []
    x_sac, y_sac = [], []
    classifier   = []

    for i in range(min_len):
        if 0 <= point_velo[i] < threshold:
            x_fix.append(x_vals[i])
            y_fix.append(y_vals[i])
            classifier.append("fixation")
        else:
            x_sac.append(x_vals[i])
            y_sac.append(y_vals[i])
            classifier.append("saccade")

    return {'x_fix': x_fix, 'y_fix': y_fix,
            'x_sac': x_sac, 'y_sac': y_sac,
            'classifier': classifier}


# ========================
# I-VT K-ratio Optimization
# ========================

def optimize_ivt_threshold(point_velo, num_thresholds=200,
                            pct_low=5, pct_high=96):
    """
    Find the I-VT threshold that minimizes the K-ratio.

    Parameters
    ----------
    point_velo : np.ndarray
    num_thresholds : int
    pct_low, pct_high : float

    Returns
    -------
    thresholds : np.ndarray
    k_ratios : np.ndarray
    optimal_threshold : float
    min_idx : int
    """
    thresholds, k_ratios, optimal_threshold, min_idx = sweep_thresholds(
        point_velo,
        n_thresholds=num_thresholds,
        pct_low=pct_low,
        pct_high=pct_high
    )
    return thresholds, k_ratios, optimal_threshold, min_idx
