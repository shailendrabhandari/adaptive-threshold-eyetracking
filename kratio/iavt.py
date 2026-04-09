

"""
iavt.py
-------
Angular velocity-threshold identification (I-AVT) with K-ratio optimization.
 
I-AVT uses effective angular velocity (paper Eq. 3-4):
    V_eff_i = V_i * cos(theta_i - theta_{i-1})
where theta_i = arctan2(y_{i+1}-y_i, x_{i+1}-x_i) is the bearing angle
of displacement vector i with the positive horizontal axis.
 
A Savitzky-Golay smoothing filter is applied to x, y coordinates
before computing V_eff, as stated in the paper, to reduce high-frequency
fluctuations at 1000 Hz sampling rate.
"""
import numpy as np
from scipy.signal import savgol_filter
from .kratio import sweep_thresholds

# Savitzky-Golay Smoothing
def smooth_coordinates(x, y, window_length=11, polyorder=2):
    """
    Apply Savitzky-Golay smoothing to x, y before I-AVT computation.
    Paper: "Savitzky-Golay smoothing filter was applied before computing
    I-AVT to reduce high-frequency fluctuations at 1000 Hz."
    Parameters
    ----------
    x, y : np.ndarray
    window_length : int
        Must be odd. Default 11 (suited for 1000 Hz data).
    polyorder : int
        Polynomial order. Default 2.
    Returns
    -------
    x_smooth, y_smooth : np.ndarray
    """
    if len(x) < window_length:
        return x.copy(), y.copy()
    x_smooth = savgol_filter(x, window_length=window_length, polyorder=polyorder)
    y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)
    return x_smooth, y_smooth


# Effective angular velocity
def compute_effective_velocity_iavt(x, y, t):
    """
    Compute effective angular velocity for I-AVT after Savitzky-Golay smoothing.

    Paper Eq. 3:  V_eff_i = V_i * cos(theta_i - theta_{i-1})
    Paper Eq. 4:  theta_i = arctan2(y_{i+1}-y_i, x_{i+1}-x_i)

    cos(theta_i - theta_{i-1}) is computed as the dot product of consecutive
    unit displacement vectors, which is mathematically equivalent.

    Note: no absolute value is applied, matching the paper formula exactly.
    Negative values (sharp direction reversal) indicate saccadic movement
    and are handled correctly by the threshold comparison in apply_iavt.

    Parameters
    ----------
    x, y : np.ndarray
        Smoothed coordinates — call smooth_coordinates() first.
    t : np.ndarray
        Timestamps.

    Returns
    -------
    v_eff : np.ndarray
    x_aligned : np.ndarray  (positions aligned to v_eff length)
    y_aligned : np.ndarray
    """
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)
    dt = np.where(dt == 0, 1e-6, dt)

    disp = np.sqrt(dx**2 + dy**2)
    V    = disp / dt               
    norms = disp.copy()
    norms[norms == 0] = np.nan        
    ux = dx / norms
    uy = dy / norms
    cos_delta = ux[1:] * ux[:-1] + uy[1:] * uy[:-1]
    cos_delta = np.clip(cos_delta, -1.0, 1.0)   # numerical safety
    valid     = np.isfinite(cos_delta)
    v_eff     = V[:-1][valid] * cos_delta[valid]  # paper Eq. 3, no abs
    x_aligned = x[1:-1][valid]
    y_aligned = y[1:-1][valid]

    return v_eff, x_aligned, y_aligned


# I-AVT Classifier

def apply_iavt(point_velo_eff, x_vals, y_vals, threshold):
    """
    Apply I-AVT classification: samples with V_eff < threshold -> fixation.
    Parameters
    ----------
    point_velo_eff : np.ndarray
        Effective angular velocity from compute_effective_velocity_iavt().
    x_vals, y_vals : np.ndarray
        Aligned coordinate arrays returned alongside v_eff.
    threshold : float
        Optimal threshold from optimize_iavt_threshold().
    Returns
    -------
    dict with keys: x_fix, y_fix, x_sac, y_sac, classifier
    """
    min_len        = min(len(point_velo_eff), len(x_vals), len(y_vals))
    point_velo_eff = point_velo_eff[:min_len]
    x_vals         = x_vals[:min_len]
    y_vals         = y_vals[:min_len]

    x_fix, y_fix = [], []
    x_sac, y_sac = [], []
    classifier   = []

    for i in range(min_len):
        if point_velo_eff[i] < threshold:
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


# I-AVT K-ratio optimization

def optimize_iavt_threshold(point_velo_eff, num_thresholds=200,
                             pct_low=0, pct_high=96):
    """
    Find the I-AVT threshold that minimizes the K-ratio.
    Threshold grid starts from 0 following original notebook convention.
    Parameters
    ----------
    point_velo_eff : np.ndarray
    num_thresholds : int
    pct_low, pct_high : float
    Returns
    -------
    thresholds : np.ndarray
    k_ratios : np.ndarray
    optimal_threshold : float
    min_idx : int
    """
    finite = point_velo_eff[np.isfinite(point_velo_eff)]
    if finite.size < 10:
        return None, None, np.nan, -1

    v_max      = np.percentile(finite, pct_high)
    thresholds = np.linspace(0, v_max, num_thresholds)

    from .kratio import compute_k_ratio_numeric
    k_ratios = np.empty(num_thresholds, dtype=float)
    for i, th in enumerate(thresholds):
        labels01    = (point_velo_eff >= th).astype(int)
        k_ratios[i] = compute_k_ratio_numeric(labels01)

    min_idx = int(np.nanargmin(k_ratios))
    return thresholds, k_ratios, float(thresholds[min_idx]), min_idx
