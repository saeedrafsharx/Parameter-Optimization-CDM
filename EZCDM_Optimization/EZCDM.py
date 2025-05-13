"""
EZ-CDM method of parameter recovery adapted to work in my framework.
"""
# Package importing
from scipy.optimize import differential_evolution
import numpy as np

def optimize_ez_cdm_framework(datasets, initial_params, bounds, maxiter=1000, popsize=15):
    """
    Optimize parameters using EZ-CDM for all parameter sets and trial counts.

    Parameters
    ----------
    datasets : dict
        Nested dictionary of datasets for each parameter set and trial count.
        Format: {param_set_idx: {n_trials: numpy_array_with_data}}.

    initial_params : list
        A list of initial parameters for each parameter set.
        Format: [[initial_params for param_set_0], [initial_params for param_set_1], ...].

    bounds : list of tuples
        Bounds for each parameter to optimize in differential_evolution.

    maxiter : int, optional
        Maximum number of iterations for Differential Evolution (default is 1000).

    popsize : int, optional
        Population size for Differential Evolution (default is 15).

    Returns
    -------
    results : dict
        Nested dictionary containing optimization results for each parameter set and trial count.
    """
    def ez_cdm_objective(params, data):
        a, v, bias, t0 = params
        CA, RT = data

        if t0 <= 0 or a <= 0 or v <= 0:
            return np.inf  # Invalid parameters

        ll = 0
        for i in range(len(CA)):
            if RT[i] <= t0:
                return np.inf

            adjusted_rt = RT[i] - t0
            angle_diff = np.cos(CA[i] - bias)
            radial_term = (v * angle_diff)**2 / (2 * adjusted_rt)

            likelihood = (1 / (2 * np.pi * adjusted_rt)) * np.exp(-radial_term)
            ll += np.log(likelihood + 1e-10)

        return -ll

    results = {}

    for param_set_idx, trials_data in datasets.items():
        results[param_set_idx] = {}
        for n_trials, data in trials_data.items():
            if not isinstance(data, np.ndarray) or data.shape[0] != 2:
                raise ValueError(f"Invalid data format for set {param_set_idx}, trials {n_trials}. Expected shape (2, n).")

            result = differential_evolution(
                lambda params: ez_cdm_objective(params, data),
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                disp=True
            )
            results[param_set_idx][n_trials] = result

    return results