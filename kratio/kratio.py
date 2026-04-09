"""
kratio.py
---------
Core K-ratio metric for adaptive threshold optimization.
"""

import numpy as np
def compute_k_ratio(classifier):
    """
    Compute K-ratio from a list/array of string labels.

    Parameters
    ----------
    classifier : list or array of str
        Sequence of "fixation" or "saccade" labels.

    Returns
    -------
    float
        K-ratio value. Returns np.inf if degenerate.
    """
    L = len(classifier)
    if L < 2:
        return np.inf

    n_sac = sum(1 for c in classifier if c == "saccade")
    P = n_sac / L
    if P == 0 or P == 1:
        return np.inf

    # Paper Eq. 10: K = n_{F->S} / (n_S * (1 - n_S))
    # n_{F->S} = count(F->S transitions) / N   (only F->S, not S->F)
    # denominator = n_S * (1 - n_S)
    n_FS = sum(
        1 for j in range(L - 1)
        if classifier[j] == "fixation" and classifier[j + 1] == "saccade"
    )
    p_emp = n_FS / L          # n_{F->S}
    p_ind = P * (1 - P)       # n_S * (1 - n_S)
    return p_emp / p_ind if p_ind != 0 else np.inf


def compute_k_ratio_numeric(labels01):
    """
    Compute K-ratio from a binary numpy array.

    Parameters
    ----------
    labels01 : np.ndarray of int
        0 = fixation, 1 = saccade.

    Returns
    -------
    float
    """
    labels01 = np.asarray(labels01, dtype=int)
    L = labels01.size
    if L < 2:
        return np.inf

    nS = labels01.mean()
    if nS <= 0 or nS >= 1:
        return np.inf

    p_ind = nS * (1 - nS)
    prev = labels01[:-1]
    nxt  = labels01[1:]
    p_emp = np.mean((prev == 0) & (nxt == 1))
    return p_emp / p_ind if p_ind > 0 else np.inf


def sweep_thresholds(feature, n_thresholds=200, pct_low=5, pct_high=96):
    feature = np.asarray(feature, dtype=float)
    finite = feature[np.isfinite(feature)]
    if finite.size < 10:
        return None, None, np.nan, -1

    v_min = np.percentile(finite, pct_low)
    v_max = np.percentile(finite, pct_high)
    if not np.isfinite(v_min) or not np.isfinite(v_max) or v_max <= v_min:
        return None, None, np.nan, -1

    thresholds = np.linspace(v_min, v_max, n_thresholds)
    k_ratios = np.empty(n_thresholds, dtype=float)

    for i, th in enumerate(thresholds):
        labels01 = (feature >= th).astype(int)
        k_ratios[i] = compute_k_ratio_numeric(labels01)

    min_idx = int(np.nanargmin(k_ratios))
    return thresholds, k_ratios, float(thresholds[min_idx]), min_idx
