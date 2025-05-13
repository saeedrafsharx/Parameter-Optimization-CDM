"""
HSDM Method of parameter optimization adopted for currenct workframe.
"""
import numpy as np
from scipy.optimize import differential_evolution
from HSDM_fpt import fpt

def SimpleHSDMLL(params, data, dt=0.01):
    """
    Compute the log-likelihood for the HSDM using simple CDM parameters and datasets.

    Parameters
    ----------
    params : list
        Model parameters [criterion (a), drift (v), bias (drift angle in radians), non-decision time (t0)].
    data : numpy array
        Observed data as (choice angles, response times).
    dt : float
        Time step for numerical approximation.

    Returns
    -------
    log_likelihood : float
        Log-likelihood for the observed data given the parameters.
    """
    # Unpack observed data
    CA, RT = data  # Choice angles and response times

    # Unpack model parameters
    a = params[0]  # Decision criterion (boundary)
    v = params[1]  # Drift rate
    bias = params[2]  # Drift angle (direction)
    t0 = params[3]  # Non-decision time

    # Compute maximum response time
    T_max = max(RT)

    # Define the drift-related functions
    def a2(t): return a**2  # Squared criterion
    def da2(t): return 0  # No collapsing threshold

    # Define drift vector
    mu = np.array([v * np.cos(bias), v * np.sin(bias)])

    # Compute first passage time density (PDF)
    from HSDM_fpt import fpt
    pdf = fpt(a2, da2, len(mu), 1e-6, dt=dt, T_max=T_max)

    # Initialize log-likelihood
    log_likelihood = 0

    # Loop through data points to compute log-likelihood
    for rt, theta in zip(RT, CA):
        # Check if the response time is valid (after non-decision time and within max RT)
        if rt - t0 > 1e-4 and rt - t0 < T_max:
            mu_dot_x0 = mu[0] * np.cos(theta)
            mu_dot_x1 = mu[1] * np.sin(theta)

            # Drift contributions
            term1 = a * (mu_dot_x0 + mu_dot_x1)
            term2 = 0.5 * np.linalg.norm(mu)**2 * (rt - t0)

            # Compute density and log-likelihood
            density = np.exp(term1 - term2) * pdf(rt - t0)
            log_likelihood += -np.log(density) if density > 1e-14 else -np.log(1e-14)
        else:
            log_likelihood += -np.log(1e-14)

    return log_likelihood

def optimize_parameters_hsdm(params_list, datasets, bounds, maxiter=10, popsize=5):

    results = {}

    for i, initial_params in enumerate(params_list):
        param_results = {}
        
        for n_trials, data in datasets[i].items():
            # Perform log-likelihood optimization
            log_likelihood_result = differential_evolution(
                lambda params: SimpleHSDMLL(params, data, dt=0.01),  # Minimizing negative log-likelihood
                bounds,
                args=(),
                maxiter=maxiter,
                popsize=popsize,
                disp=True
            )

            # Store results for this n_trials
            param_results[n_trials] = log_likelihood_result

        # Store results for this parameter set
        results[i] = param_results

    return results