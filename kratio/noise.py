"""
noise.py
--------
Gaussian noise injection and robustness sweep for I-VT, I-AVT, I-DT.

Reference:
    Orioma et al. / Bhandari et al. (2026), Section 2.1 and Fig. 4
"""

import numpy as np
from .ivt  import apply_ivt,  optimize_ivt_threshold
from .iavt import apply_iavt, optimize_iavt_threshold
from .idt  import apply_idt,  optimize_idt_threshold, grid_search_idt
from .preprocessing import compute_velocity, compute_effective_velocity


# ========================
# Gaussian Noise Injection
# ========================

def add_gaussian_noise(x_coords, y_coords, noise_level):
    """
    Add Gaussian noise to eye-tracking coordinates.

    Parameters
    ----------
    x_coords, y_coords : np.ndarray
    noise_level : float
        Standard deviation sigma of the noise in pixels.
        noise_level=0 returns unmodified copies.

    Returns
    -------
    noisy_x, noisy_y : np.ndarray
    """
    noisy_x = np.copy(x_coords)
    noisy_y = np.copy(y_coords)
    if noise_level > 0:
        noisy_x += np.random.normal(0, noise_level, len(noisy_x))
        noisy_y += np.random.normal(0, noise_level, len(noisy_y))
    return noisy_x, noisy_y


# ========================
# Full Noise Sweep
# ========================

def run_noise_sweep(x_sample, y_sample, time_sample,
                    noise_levels=None,
                    idt_window_size=10,
                    idt_fixed_duration_samples=50,
                    num_thresholds_ivt=200,
                    num_thresholds_idt=50,
                    verbose=True):
    """
    Run adaptive threshold optimization across Gaussian noise levels
    for all three algorithms (I-VT, I-AVT, I-DT).

    Parameters
    ----------
    x_sample, y_sample, time_sample : np.ndarray
        Subset of data to use for the sweep.
    noise_levels : list of float
        Sigma values for noise (default [0,1,2,5,10,30,40,50]).
    idt_window_size : int
    idt_fixed_duration_samples : int
    num_thresholds_ivt : int
    num_thresholds_idt : int
    verbose : bool

    Returns
    -------
    dict with keys:
        noise_levels,
        adaptive_ivt, adaptive_iavt, adaptive_idt,
        ivt_fix_counts, ivt_sac_counts,
        iavt_fix_counts, iavt_sac_counts,
        idt_fix_counts, idt_sac_counts,
        ivt_curves, iavt_curves, idt_curves
    """
    if noise_levels is None:
        noise_levels = [0, 1, 2, 5, 10, 30, 40, 50]

    dt_series = np.diff(time_sample)
    dt_series = np.where(dt_series == 0, 1e-6, dt_series)

    adaptive_ivt,  adaptive_iavt,  adaptive_idt  = [], [], []
    ivt_fix_counts, ivt_sac_counts   = [], []
    iavt_fix_counts, iavt_sac_counts = [], []
    idt_fix_counts, idt_sac_counts   = [], []
    ivt_curves, iavt_curves, idt_curves = {}, {}, {}

    for nl in noise_levels:
        if verbose:
            print(f"  Noise level: {nl}")

        nx, ny = add_gaussian_noise(x_sample, y_sample, nl)

        # ---- I-VT ----
        dx = np.diff(nx);  dy = np.diff(ny)
        vel = np.sqrt(dx**2 + dy**2) / dt_series
        ths, krs, opt, _ = optimize_ivt_threshold(
            vel, num_thresholds=num_thresholds_ivt, pct_low=5, pct_high=96)
        adaptive_ivt.append(opt)
        ivt_curves[nl] = (ths, krs)
        if np.isfinite(opt):
            x_ivt = nx[:len(vel)];  y_ivt = ny[:len(vel)]
            res = apply_ivt(vel, x_ivt, y_ivt, opt)
            ivt_fix_counts.append(res['classifier'].count("fixation"))
            ivt_sac_counts.append(res['classifier'].count("saccade"))
        else:
            ivt_fix_counts.append(np.nan); ivt_sac_counts.append(np.nan)

        # ---- I-AVT ----
        veff, x_corr, y_corr, _ = compute_effective_velocity(nx, ny, time_sample)
        if len(veff) > 10:
            ths_a, krs_a, opt_a, _ = optimize_iavt_threshold(
                veff, num_thresholds=num_thresholds_ivt, pct_low=0, pct_high=96)
            adaptive_iavt.append(opt_a)
            iavt_curves[nl] = (ths_a, krs_a)
            if np.isfinite(opt_a):
                res_a = apply_iavt(veff, x_corr, y_corr, opt_a)
                iavt_fix_counts.append(res_a['classifier'].count("fixation"))
                iavt_sac_counts.append(res_a['classifier'].count("saccade"))
            else:
                iavt_fix_counts.append(np.nan); iavt_sac_counts.append(np.nan)
        else:
            adaptive_iavt.append(np.nan)
            iavt_fix_counts.append(np.nan); iavt_sac_counts.append(np.nan)

        # ---- I-DT ----
        ths_d, krs_d, opt_d, _ = optimize_idt_threshold(
            nx, ny,
            window_size=idt_window_size,
            fixed_duration_samples=idt_fixed_duration_samples,
            num_thresholds=num_thresholds_idt
        )
        adaptive_idt.append(opt_d)
        idt_curves[nl] = (ths_d, krs_d)
        if np.isfinite(opt_d):
            # Use time_sample as timestamps (scaled to seconds consistent with apply_idt)
            res_d = apply_idt(nx, ny, time_sample, opt_d, opt_d, dur_threshold=0.050)
            idt_fix_counts.append(res_d['classifier'].count("fixation"))
            idt_sac_counts.append(res_d['classifier'].count("saccade"))
        else:
            idt_fix_counts.append(np.nan); idt_sac_counts.append(np.nan)

    return dict(
        noise_levels=noise_levels,
        adaptive_ivt=np.array(adaptive_ivt, dtype=float),
        adaptive_iavt=np.array(adaptive_iavt, dtype=float),
        adaptive_idt=np.array(adaptive_idt, dtype=float),
        ivt_fix_counts=ivt_fix_counts, ivt_sac_counts=ivt_sac_counts,
        iavt_fix_counts=iavt_fix_counts, iavt_sac_counts=iavt_sac_counts,
        idt_fix_counts=idt_fix_counts, idt_sac_counts=idt_sac_counts,
        ivt_curves=ivt_curves,
        iavt_curves=iavt_curves,
        idt_curves=idt_curves,
    )
