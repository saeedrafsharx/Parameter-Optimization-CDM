import numpy as np
from scipy.optimize import differential_evolution

def KS_CDM(params, data, A, T, P):
    """
    KS_CDM(params, data)

    Calculate the Kolmogorov-Smirnov (KS) value of the data for the Circular Diffusion Model (CDM).

    Parameters
    ----------
    params : list
        A list of real numbers containing the parameter values as [decision criterion, multiplier for decision criterion range, 
        drift length, drift angle, multiplier for the standard deviation of radial component of the drift, multiplier for the 
        standard deviation of tangental component of the drift, non-decision time, range of the non-decision time variability].

    data: numpy array
        An array of shape (2,n) containing the choice angle and response time pairs for all trials.

    A: list
        A list of real numbers as [lower bound of the range, higher bound of the range, step size] indicating the range of the decision criterion 
        at which the series is calculated 

    T: list
        A list of real numbers as [lower bound of the range, higher bound of the range, step size] indicating the range of the response time
        at which the series is calculated.

    P: numpy array
        A 2D array of calculated values of the series for the range of decision criterion and response time values.

    Returns
    -------
    out : float
        The Kolmogorov-Smirnov (KS) value.

    Notes
    -----
    Dynamically handles variations in the number of parameters. Missing parameters are set to default values.
    """

    a = params[0]
    sa = params[1] * a if len(params) > 1 else 0
    v = params[2] if len(params) > 2 else 1
    bias = params[3] if len(params) > 3 else 0
    eta1 = params[4] * v if len(params) > 4 else 0
    eta2 = params[5] * v if len(params) > 5 else 0
    t0 = params[6] if len(params) > 6 else 0.3
    st = params[7] if len(params) > 7 else 0

    CA, RT = data
    a_ = np.arange(A[0], A[1], A[2])
    t_ = np.arange(T[0], T[1], T[2])
    da = A[2]
    dt = T[2]

    if int((min(RT) - t0 + st / 2) / dt) <= 0:
        return np.inf  # KS should be high for invalid inputs

    ks_stat = 0
    for i in range(len(CA)):
        A_start = max(0, int((a - sa / 2) / da))
        A_end = min(len(a_), int((a + sa / 2) / da) + 1)
        T_start = max(0, int((RT[i] - t0 - st / 2) / dt))
        T_end = min(len(t_), int((RT[i] - t0 + st / 2) / dt) + 1)

        A_ = a_[A_start:A_end]

        if RT[i] >= t0 + st / 2:
            T_ = t_[T_start:T_end]
            P_ = P[A_start:A_end, T_start:T_end]
        else:
            T_ = t_[:T_end]
            P_ = P[A_start:A_end, :T_end]

        T_ = T_.reshape((1, len(T_)))
        A_ = A_.reshape((len(A_), 1))

        Z = (np.exp((-v**2 * T_ + A_**2 * np.cos(CA[i] - bias)**2 * eta1**2 + 2 * v * A_ * np.cos(CA[i] - bias)) /
             ((eta1**2 * T_ + 1) / 2) + A_**2 * np.sin(CA[i] - bias)**2 * eta2**2 / ((eta2**2 * T_ + 1) / 2)) /
             (np.sqrt(eta1**2 * T_ + 1) * np.sqrt(eta2**2 * T_ + 1)))

        Z[P_ == 0] = 0

        if Z.size == 0 or np.sum(P_) == 0:
            return np.inf  # KS should be high for invalid outputs

        cumulative_observed = np.cumsum(P_ / np.sum(P_))
        cumulative_expected = np.cumsum(Z * P_ / np.sum(Z * P_))

        ks_stat = max(ks_stat, np.max(np.abs(cumulative_observed - cumulative_expected)))

    return ks_stat

def optimize_cdm_ks(params_list, datasets, A, T, P, bounds, maxiter=1000, popsize=15):
    """
    Optimize the parameters of the Circular Diffusion Model (CDM) using Kolmogorov-Smirnov (KS) as the objective function.

    Parameters
    ----------
    params_list : list of lists
        A list of initial parameter sets for optimization.
    datasets : dict
        A dictionary of simulated datasets where keys are parameter set indices and values are dictionaries
        with n_trials as keys and datasets as values.
    A : list
        Decision criterion range [lower bound, upper bound, step size].
    T : list
        Response time range [lower bound, upper bound, step size].
    P : numpy array
        Precomputed series for likelihood calculations.
    bounds : list of tuples
        Bounds for each parameter to optimize in differential_evolution.
    maxiter : int, optional
        Maximum number of iterations for Differential Evolution (default is 1000).
    popsize : int, optional
        Population size for Differential Evolution (default is 15).

    Returns
    -------
    results : dict
        A dictionary containing optimization results. The outer keys are parameter set indices, the inner keys
        are `n_trials`, and values are the optimization result for the KS objective.
    """
    from tqdm import tqdm

    results = {}

    for i, initial_params in tqdm(enumerate(params_list), total=len(params_list), desc="Optimizing parameter sets (KS)"):
        param_results = {}

        for n_trials, data in tqdm(datasets[i].items(), desc=f"Parameter set {i+1}", leave=False):
            # Perform KS optimization
            ks_result = differential_evolution(
                lambda params: KS_CDM(params, data, A, T, P),
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                disp=False  # Suppress built-in display of minimization steps
            )

            # Store results for this n_trials
            param_results[n_trials] = ks_result

        # Store results for this parameter set
        results[i] = param_results

    return results
