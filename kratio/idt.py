"""
idt.py
------
Dispersion-threshold identification (I-DT) with K-ratio optimization.
I-DT identifies fixations as windows where spatial dispersion stays below
a threshold for at least Tmin ms.
"""
import numpy as np
from .kratio import compute_k_ratio_numeric
# I-DT Classifier

def apply_idt(x_vals, y_vals, timestamps,
              x_threshold=25, y_threshold=25, dur_threshold=0.150):
    x_fix, y_fix = [], []
    x_sac, y_sac = [], []
    classifier   = []

    temp_x_fix, temp_y_fix = [], []
    temp_timestamps = []

    s_point = 0
    for i in range(len(x_vals)):
        if i < s_point:
            continue

        temp_x_fix.append(x_vals[i])
        temp_y_fix.append(y_vals[i])
        temp_timestamps.append(timestamps[i])

        max_x = max(temp_x_fix);  min_x = min(temp_x_fix)
        max_y = max(temp_y_fix);  min_y = min(temp_y_fix)

        if (max_x - min_x) > x_threshold or (max_y - min_y) > y_threshold:
            dur = temp_timestamps[-1] - temp_timestamps[0] if len(temp_timestamps) > 1 else 0
            if dur >= dur_threshold and len(temp_x_fix) > 1:
                x_fix.extend(temp_x_fix[:-1])
                y_fix.extend(temp_y_fix[:-1])
                classifier.extend(["fixation"] * (len(temp_x_fix) - 1))
            else:
                x_sac.extend(temp_x_fix[:-1])
                y_sac.extend(temp_y_fix[:-1])
                classifier.extend(["saccade"] * (len(temp_x_fix) - 1))

            x_sac.append(x_vals[i])
            y_sac.append(y_vals[i])
            classifier.append("saccade")

            temp_x_fix, temp_y_fix, temp_timestamps = [], [], []
            s_point = i + 1

        if i == len(x_vals) - 1:
            dur = temp_timestamps[-1] - temp_timestamps[0] if len(temp_timestamps) > 1 else 0
            if dur >= dur_threshold:
                x_fix.extend(temp_x_fix)
                y_fix.extend(temp_y_fix)
                classifier.extend(["fixation"] * len(temp_x_fix))
            else:
                x_sac.extend(temp_x_fix)
                y_sac.extend(temp_y_fix)
                classifier.extend(["saccade"] * len(temp_x_fix))

    return {'x_fix': x_fix, 'y_fix': y_fix,
            'x_sac': x_sac, 'y_sac': y_sac,
            'classifier': classifier}


# I-DT dispersion series 
def compute_dispersion_series(x_vals, y_vals, window_size=10):
    n = len(x_vals)
    if n <= window_size:
        return np.array([])
    disp = np.empty(n - window_size, dtype=float)
    for i in range(n - window_size):
        wx = x_vals[i:i + window_size]
        wy = y_vals[i:i + window_size]
        disp[i] = (np.max(wx) - np.min(wx)) + (np.max(wy) - np.min(wy))
    return disp


def optimize_idt_threshold(x_vals, y_vals,
                            window_size=10,
                            fixed_duration_samples=50,
                            num_thresholds=50,
                            pct_low=1, pct_high=98):

    disp_values = compute_dispersion_series(x_vals, y_vals, window_size)
    if len(disp_values) < 5:
        return None, None, np.nan, -1

    d_min = np.percentile(disp_values, pct_low)
    d_max = np.percentile(disp_values, pct_high)
    if not np.isfinite(d_min) or not np.isfinite(d_max) or d_max <= d_min:
        return None, None, np.nan, -1

    thresholds = np.linspace(d_min, d_max, num_thresholds)
    k_ratios   = np.empty(num_thresholds, dtype=float)
    L = len(disp_values)

    for t_idx, threshold in enumerate(thresholds):
        classifier = ["saccade"] * L
        i = 0
        sac_count = 0
        while i < L:
            if disp_values[i] < threshold:
                start = i
                while i < L and disp_values[i] < threshold:
                    i += 1
                end = i
                if (end - start) >= fixed_duration_samples:
                    for j in range(start, end):
                        classifier[j] = "fixation"
                else:
                    for j in range(start, end):
                        classifier[j] = "saccade"
                        sac_count += 1
            else:
                sac_count += 1
                i += 1

        # Paper Eq. 10: K = n_{F->S} / (n_S * (1 - n_S))
        n_sac_tot = sum(1 for c in classifier if c == "saccade")
        P = n_sac_tot / L if L > 0 else 0.0
        if P == 0 or P == 1:
            k_ratios[t_idx] = np.inf
            continue
        n_FS = sum(
            1 for j in range(L - 1)
            if classifier[j] == "fixation" and classifier[j + 1] == "saccade"
        )
        p_emp = n_FS / L          # n_{F->S}
        p_ind = P * (1 - P)       # n_S * (1 - n_S)
        k_ratios[t_idx] = p_emp / p_ind if p_ind != 0 else np.inf

    min_idx = int(np.nanargmin(k_ratios))
    return thresholds, k_ratios, float(thresholds[min_idx]), min_idx


def grid_search_idt(x_vals, y_vals, timestamps,
                    xy_thresholds=None, dur_thresholds=None):
    from .kratio import compute_k_ratio

    if xy_thresholds is None:
        xy_thresholds = np.linspace(5, 90, 10)
    if dur_thresholds is None:
        dur_thresholds = np.linspace(0.05, 0.5, 5)

    best_k_ratio = np.inf
    best_params = {'x_y_threshold': None, 'dur_threshold': None}

    print("Starting I-DT grid search...")
    for xy_th in xy_thresholds:
        for dur_th in dur_thresholds:
            result = apply_idt(x_vals, y_vals, timestamps, xy_th, xy_th, dur_th)
            kr = compute_k_ratio(result['classifier'])
            if kr < best_k_ratio:
                best_k_ratio = kr
                best_params = {'x_y_threshold': xy_th, 'dur_threshold': dur_th}

    print(f"Optimal I-DT params: {best_params}, K-ratio = {best_k_ratio:.4f}")
    return best_params, best_k_ratio
