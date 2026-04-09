"""
kratio-eyetracking
==================
Adaptive threshold-based fixation/saccade classification using K-ratio minimization.
Modules
-------
preprocessing   Load and preprocess EyeLink Waldo task data
kratio          Core K-ratio metric
ivt             I-VT algorithm + optimization
iavt            I-AVT algorithm + optimization
idt             I-DT algorithm + optimization
noise           Gaussian noise injection and robustness sweep
markov          Stationarity and Markov adequacy diagnostics
"""

from .preprocessing import (
    load_waldo_directory,
    preprocess_waldo,
    compute_velocity,
    compute_effective_velocity,
    binocular_coordination,
)
from .kratio import compute_k_ratio, compute_k_ratio_numeric, sweep_thresholds
from .ivt    import apply_ivt,  optimize_ivt_threshold
from .iavt   import (apply_iavt, optimize_iavt_threshold,
                     smooth_coordinates, compute_effective_velocity_iavt)
from .idt    import apply_idt,  optimize_idt_threshold, grid_search_idt, compute_dispersion_series
from .noise  import add_gaussian_noise, run_noise_sweep
from .markov import (
    blockwise_kratio_stability,
    markov_tk_deviation,
    plot_markov_diagnostics,
)

__version__ = "1.0.0"
__all__ = [
    "load_waldo_directory", "preprocess_waldo",
    "compute_velocity", "compute_effective_velocity", "binocular_coordination",
    "compute_k_ratio", "compute_k_ratio_numeric", "sweep_thresholds",
    "apply_ivt", "optimize_ivt_threshold",
    "apply_iavt", "optimize_iavt_threshold",
    "smooth_coordinates", "compute_effective_velocity_iavt",
    "apply_idt", "optimize_idt_threshold", "grid_search_idt",
    "compute_dispersion_series",
    "add_gaussian_noise", "run_noise_sweep",
    "blockwise_kratio_stability", "markov_tk_deviation", "plot_markov_diagnostics",
]
