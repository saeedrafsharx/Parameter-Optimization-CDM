'''
This utility file contains all the funcitons needed to do simulation, parameter generation
and optmization using Chi-Square, and MLE as objective functions and differential- 
evolution as the method for CDM model.
'''
# Library imports
import numpy as np
import mpmath
from numpy import exp, cos, sin, sqrt, mean, sum, log, mean
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from itertools import product
from tqdm import tqdm

def SimCDM(params, dt=.001, n=100):
    
    """
    SimCDM(params, dt=.001, n=100)
    
    Simulate a dataset from the Circular Diffusion Model.
    
    Parameters
    ----------
    params : list
        A list of real numbers containing the parameter values as [decision criterion, multiplier for decision criterion range, 
        drift length, drift angle, multiplier for the standard deviation of radial component of the drift, multiplier for the 
        standard deviation of tangental component of the drift, non-decision time, range of the non-decision time variability]

    dt: float
        The time step.

    n: int
        The number of trials.
        

    Returns
    -------
    out : numpy array
        An array of shape (2,n) containing the choice angle and responce time pairs for all trials.


    Notes
    -----
    The variability on the radial and tangental components of the drift vector is considered to follow the Normal distribution. The 
    variability on the decision criterion and non-decision time is considered to follow the Uniform distribution.

    Set variability parameter values to zero to simulate data from the simple CDM.
    """
    
    a = params[0] 
    sa = a*params[1]
    v = params[2] 
    bias = params[3]
    eta1 = v*params[4] 
    eta2 = v*params[5] 
    t0 = params[6] 
    st = params[7]
    hit = False
    x = [0,0]
    t = 0
    RT = []
    CA = []
    for i in range(n):
        A = np.random.uniform(a-sa/2, a+sa/2)
        A2 = A**2
        drift = np.random.normal([v,0], [eta1,eta2])
        V = np.array([drift[0]*np.cos(bias)-drift[1]*np.sin(bias), drift[0]*np.sin(bias)+drift[1]*np.cos(bias)])
        T = np.random.uniform(t0-st/2, t0+st/2)
        while not hit:
            x += np.random.normal(V*dt, [np.sqrt(dt),np.sqrt(dt)])
            t += dt
            hit = x[0]**2 + x[1]**2 >= A2
        else:
            RT.append(t+T)
            CA.append(np.arctan2(x[1],x[0]))
            x = [0,0]
            t = 0
            hit = False
    return np.array([CA, RT])

def Series(A=[.05,9,.05], T=[0,18,.05], n=50, r=1e-10, dps=5):
    
    """
    Series(a=[.0005,9,.0005], t=[0,18,.0005], n_terms=2000, dps=50)
    
    Calculates the series used in the likelihood function of the CDM.
    
    Parameters
    ----------
    A: list
        A list of real numbers as [lower bound of the range, higher bound of the range, step size] indicating the range of the decision criterion 
        at which the series needs to be calculated (The range should contain only positive values).

    T: list
        A list of real numbers as [lower bound of the range, higher bound of the range, step size] indicating the range of the response time
        at which the series needs to be calculated (The range should contain only non-negative values).
        
    n: int
        Maximum number of terms that the series will be calculated.

    r: float
        Criterion for relative value of the terms which terminates the calculation process.
    
    dps: int
        The precision of the calculations as number of decimal places.
        

    Returns
    -------
    out : numpy array
        A 2D array of calculated values of the series for the range of decision criterion and response time values.


    Notes
    -----
    The higher precision is needed only in the calculation pf the series. So at the end, the calculated values could be transformed 
    to the default floating point precision of the Python.
    """

    mpmath.mp.dps = dps
    
    j0 = np.empty(n, dtype=mpmath.mpf)
    J1 = np.empty(n, dtype=mpmath.mpf)
    
    for i in range(n):
        j0[i] = mpmath.besseljzero(mpmath.mpf(0),i+1)
        J1[i] = mpmath.besselj(mpmath.mpf(1),j0[i])

    a = np.arange(A[0],A[1],A[2])
    t = np.arange(T[0],T[1],T[2])
    P = np.empty((len(a),len(t)))
    for i in range(len(a)):
        for j in range(len(t)):
            a_ = mpmath.mpf(a[i])
            t_ = mpmath.mpf(t[j])
            a2 = a_**2
            series = 0
            for k in range(n):
                term = j0[k]/J1[k]*mpmath.exp(-j0[k]**2*t_/2/a2)
                series += term
                if abs(term/series)<r:
                    break
            series = series/a2/2/mpmath.pi
            if series<0:
                series = 0
            P[i,j] = series
            
    return P




def CDM(params, data, A, T, P):
    
    """
    CDM(params, data)
    
    Calculate the log-likelihood value of the data.
    
    Parameters
    ----------
    params : list
        A list of real numbers containing the parameter values as [decision criterion, multiplier for decision criterion range, 
        drift length, drift angle, multiplier for the standard deviation of radial component of the drift, multiplier for the 
        standard deviation of tangental component of the drift, non-decision time, range of the non-decision time variability]

    data: numpy array
        An array of shape (2,n) containing the choice angle and responce time pairs for all trials.

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
        logarithm of the likelihood value.


    Notes
    -----
    'dt' and 'da' are the step size in which the decision criterion and response time is chosen in calculation of 'P'.
    They are used as the step size in the calculation of the integrals.
    """

    
    a = params[0]
    sa = params[1]*a
    v = params[2]
    bias = params[3]
    eta1 = params[4]*v
    eta2 = params[5]*v
    t0 = params[6]
    st = params[7]
    CA,RT = data
    a_ = np.arange(A[0],A[1],A[2])
    t_ = np.arange(T[0],T[1],T[2])
    da = A[2]
    dt = T[2]
    if int((min(RT)-t0+st/2)/dt)<=0:
        return -np.inf
    ll = 0
    for i in range(len(CA)):
        A_ = a_[int((a-sa/2)/da):int((a+sa/2)/da)+1]
        if RT[i]>=t0+st/2:
            T_ = t_[int((RT[i]-t0-st/2)/dt):int((RT[i]-t0+st/2)/dt)+1]
            P_ = P[int((a-sa/2)/da):int((a+sa/2)/da)+1, int((RT[i]-t0-st/2)/dt):int((RT[i]-t0+st/2)/dt)+1]
            mT = 1
        else:
            T_ = t_[0:int((RT[i]-t0+st/2)/dt)+1]
            P_ = P[int((a-sa/2)/da):int((a+sa/2)/da)+1, 0:int((RT[i]-t0+st/2)/dt)+1]
            mT = (int((RT[i]-t0+st/2)/dt)+1) / (int((RT[i]-t0+st/2)/dt) - int((RT[i]-t0-st/2)/dt)+1)
        T_ = T_.reshape((1,len(T_)))
        A_ = A_.reshape((len(A_),1))
        Z = exp((-v**2*T_+A_**2*cos(CA[i]-bias)**2*eta1**2+2*v*A_*cos(CA[i]-bias))/(eta1**2*T_+1)/2 + A_**2*sin(CA[i]-bias)**2*eta2**2/(eta2**2*T_+1)/2)/sqrt(eta1**2*T_+1)/sqrt(eta2**2*T_+1)
        Z[P_==0] = 0
        ll += log((mean(Z*P_) + 1e-10)*mT)
    return ll

def ChiSquareCDM(params, data, A, T, P, bins=10):
    """
    ChiSquareCDM calculates the chi-square value for the Circular Diffusion Model.
    
    Parameters
    ----------
    params : list
        Parameter list for the CDM, similar to the format for SimCDM.
    data : numpy array
        Dataset containing choice angles (CA) and response times (RT).
    A : list
        Decision criterion range [lower, upper, step] for likelihood calculations.
    T : list
        Response time range [lower, upper, step] for likelihood calculations.
    P : numpy array
        Precomputed series for the likelihood calculations.
    bins : int
        Number of bins to divide data for chi-square analysis (default is 10).

    Returns
    -------
    chi_square : float
        The chi-square value.
    """
    
    # Extract CA and RT from data
    CA, RT = data
    n_trials = len(CA)
    
    # Step 1: Binning CA and RT data
    CA_bins = np.linspace(min(CA), max(CA), bins + 1)
    RT_bins = np.linspace(min(RT), max(RT), bins + 1)
    
    # Step 2: Quantiles for RT and CA
    CA_quantiles = np.percentile(CA, [20, 40, 60, 80, 100])  # Quantiles as specified
    RT_quantiles = np.percentile(RT, [10, 30, 50, 70, 90])   # Quantiles as specified
    
    # Step 3: Observed values in each bin
    observed_counts = np.zeros((bins, bins))
    for i in range(bins):
        for j in range(bins):
            observed_counts[i, j] = np.sum((CA >= CA_bins[i]) & (CA < CA_bins[i+1]) & 
                                           (RT >= RT_bins[j]) & (RT < RT_bins[j+1]))
    
    # Step 4 & 5: Expected values
    expected_counts = np.zeros((bins, bins))
    
    # Define step sizes for numerical integration
    da = A[2]
    dt = T[2]
    
    for i in range(bins):
        for j in range(bins):
            ca_bin = (CA_bins[i], CA_bins[i+1])
            rt_bin = (RT_bins[j], RT_bins[j+1])
            
            # Calculate expected probability within each bin (using trapezoidal integration over likelihood)
            ca_range = np.arange(ca_bin[0], ca_bin[1], da)
            rt_range = np.arange(rt_bin[0], rt_bin[1], dt)
            ca_mesh, rt_mesh = np.meshgrid(ca_range, rt_range, indexing='ij')
            
            likelihood = np.array([
                np.exp(CDM(params, np.array([[ca], [rt]]), A, T, P)) for ca, rt in zip(ca_mesh.flatten(), rt_mesh.flatten())
            ]).reshape(ca_mesh.shape)  # remove log and exponentiate
            
            # Trapezoidal integration for the expected count in the bin
            integral = np.trapz(np.trapz(likelihood, ca_range, axis=0), rt_range)
            expected_counts[i, j] = integral * n_trials  # Scale by number of trials to get counts
    
    # Step 6: Calculate chi-square value
    chi_square = np.sum(((observed_counts - expected_counts) ** 2) / (expected_counts + 1e-10))  # avoid division by zero
    
    return chi_square


def generate_parameters(n):
    """
    Generates 'n' sets of parameters for the SimCDM function based on specified ranges.

    Parameters
    ----------
    n : int
        The number of parameter sets to generate.

    Returns
    -------
    param_sets : list of lists
        A list of parameter sets, where each set is a list of 8 parameter values.
    """
    param_sets = []
    for _ in range(n):
        params = [
            np.random.uniform(0.5, 3.0),      # decision_criterion
            np.random.uniform(0.0, 1.0),      # sa
            np.random.uniform(0.1, 3.0),      # drift_length
            np.random.uniform(0, 2*np.pi), # drift_angle
            np.random.uniform(0.0, 0.7),      # radial_variability
            np.random.uniform(0.0, 0.7),      # tangential_variability
            np.random.uniform(0.1, 1.0),      # non_decision_time
            np.random.uniform(0.0, 0.5)       # non_decision_time_variability
        ]
        param_sets.append(params)
    
    return param_sets

# Simple CDM
def simple_parameters_empty(n):
    """
    Generates 'n' sets of parameters for the SimCDM function based on specified ranges.

    Parameters
    ----------
    n : int
        The number of parameter sets to generate.

    Returns
    -------
    param_sets : list of lists
        A list of parameter sets, where each set is a list of 8 parameter values.
    """
    simple_param_sets = []
    for _ in range(n):
        params = [
            np.random.uniform(0.5, 3.0),      # decision_criterion
            np.random.uniform(0.0, 0.0),      # sa
            np.random.uniform(0.1, 3.0),      # drift_length
            np.random.uniform(0.0, 2*np.pi), # drift_angle
            np.random.uniform(0.0, 0.0),      # radial_variability
            np.random.uniform(0.0, 0.0),      # tangential_variability
            np.random.uniform(0.1, 1.0),      # non_decision_time
            np.random.uniform(0.0, 0.0)       # non_decision_time_variability
        ]
        simple_param_sets.append(params)
    
    return simple_param_sets

def simulate_datasets(params_list, n_trials_list=[50, 100, 200, 300, 500, 1000]):
    """
    Simulates datasets using the SimCDM function for each parameter set and each value in n_trials_list.

    Parameters
    ----------
    params_list : list of lists
        A list of parameter sets, where each set is a list of 8 parameter values for SimCDM.
    n_trials_list : list of int, optional
        A list of trial counts to simulate data for each parameter set (default is [50, 100, 200, 300, 500, 1000]).

    Returns
    -------
    simulations : dict
        A dictionary with keys as parameter set indices. Each value is another dictionary where keys are `n_trials`
        values and values are the simulated datasets (numpy arrays) from SimCDM.
    """
    simulations = {}

    for i, params in enumerate(params_list):
        trials_data = {}
        for n_trials in n_trials_list:
            trials_data[n_trials] = SimCDM(params, n=n_trials)  # Replace 100 with `n_trials` for variation.
        simulations[i] = trials_data

    return simulations

def optimize_parameters_mle(params_list, datasets, A, T, P, bounds, maxiter=10, popsize=5):
    """
    Optimizes parameters for each parameter set and dataset using Maximum Likelihood Estimation.

    Parameters
    ----------
    params_list : list of lists
        A list of initial parameter sets for SimCDM.
    datasets : dict
        A dictionary of simulated datasets where keys are parameter set indices and values are dictionaries
        with n_trials as keys and datasets as values.
    A : list
        Decision criterion range [lower bound, upper bound, step size] for series calculation in CDM.
    T : list
        Response time range [lower bound, upper bound, step size] for series calculation in CDM.
    P : numpy array
        Precomputed series for likelihood calculations.
    bounds : list of tuples
        Bounds for each parameter to optimize in differential_evolution.
    maxiter : int, optional
        Maximum number of iterations for differential_evolution (default is 10).
    popsize : int, optional
        Population size for differential_evolution (default is 5).

    Returns
    -------
    results : dict
        A dictionary containing optimization results. The outer keys are parameter set indices, the inner keys
        are `n_trials`, and values are the optimization result for the log-likelihood objective.
    """
    results = {}

    for i, initial_params in enumerate(params_list):
        param_results = {}
        
        for n_trials, data in datasets[i].items():
            # Perform log-likelihood optimization
            log_likelihood_result = differential_evolution(
                lambda params: -CDM(params, data, A, T, P),  # Minimizing negative log-likelihood
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


def optimize_parameters_chi_square(params_list, datasets, A, T, P, bounds, maxiter=10, popsize=5, seed=42):
    """
    Optimizes parameters for each parameter set and dataset using the Chi-Square objective.

    Parameters
    ----------
    params_list : list of lists
        A list of initial parameter sets for SimCDM.
    datasets : dict
        A dictionary of simulated datasets where keys are parameter set indices and values are dictionaries
        with n_trials as keys and datasets as values.
    A : list
        Decision criterion range [lower bound, upper bound, step size] for series calculation in CDM.
    T : list
        Response time range [lower bound, upper bound, step size] for series calculation in CDM.
    P : numpy array
        Precomputed series for likelihood calculations.
    bounds : list of tuples
        Bounds for each parameter to optimize in differential_evolution.
    maxiter : int, optional
        Maximum number of iterations for differential_evolution (default is 10).
    popsize : int, optional
        Population size for differential_evolution (default is 5).
    seed : int, optional
        Seed for reproducibility (default is 42).

    Returns
    -------
    results : dict
        A dictionary containing optimization results. The outer keys are parameter set indices, the inner keys
        are `n_trials`, and values are the optimization result for the chi-square objective.
    """
    results = {}

    for i, initial_params in enumerate(params_list):
        param_results = {}
        
        for n_trials, data in datasets[i].items():
            # Perform chi-square optimization
            chi_square_result = differential_evolution(
                ChiSquareCDM,
                bounds,
                args=(data, A, T, P),
                maxiter=maxiter,
                popsize=popsize,
                disp=True,
                seed=seed
            )

            # Store results for this n_trials
            param_results[n_trials] = chi_square_result

        # Store results for this parameter set
        results[i] = param_results

    return results

# TODO NOT TESTED
def extract_parameter_values(results, param_index):
    """
    Extracts the values of a specific parameter from optimized parameters for each trial count and dataset.

    Parameters
    ----------
    results : dict
        A dictionary containing optimization results. The outer keys are parameter set indices, the inner keys
        are `n_trials`, and values are the optimization results (which include optimized parameter sets).
    param_index : int
        The index of the parameter to extract from the optimized parameter sets.

    Returns
    -------
    extracted_values : dict
        A dictionary where the outer keys are parameter set indices, the inner keys are `n_trials`, and
        values are the extracted parameter values for the specified parameter index.
    """
    extracted_values = {}

    for param_set_idx, trials_results in results.items():
        trial_values = {}
        
        for n_trials, optimization_result in trials_results.items():
            # Extract the specific parameter's optimized value
            optimized_params = optimization_result.x  # Access the optimized parameters
            trial_values[n_trials] = optimized_params[param_index]
        
        # Store the extracted values for each parameter set
        extracted_values[param_set_idx] = trial_values

    return extracted_values


import numpy as np

def KS2D_CDM(params, data, A, T, P, grid_points=100):
    """
    KS2D_CDM_quadrant calculates a 2D KS-like statistic for the Circular Diffusion Model using quadrant-based CDFs.

    Parameters
    ----------
    params : list
        Parameter list for the CDM, similar to the format for SimCDM.
    data : numpy array
        Dataset containing choice angles (CA) and response times (RT).
    A : list
        Decision criterion range [lower, upper, step] for likelihood calculations.
    T : list
        Response time range [lower, upper, step] for likelihood calculations.
    P : numpy array
        Precomputed series for likelihood calculations.
    grid_points : int, optional
        Number of grid points in each dimension to evaluate the theoretical CDF (default is 100).

    Returns
    -------
    ks_statistic : float
        The 2D KS-like statistic, maximum difference between empirical and model-based quadrant CDFs.
    """
    CA, RT = data
    n_trials = len(CA)
    
    # Create the theoretical probability grid from the CDM model
    ca_values = np.linspace(min(CA), max(CA), grid_points)
    rt_values = np.linspace(min(RT), max(RT), grid_points)
    ca_mesh, rt_mesh = np.meshgrid(ca_values, rt_values, indexing='ij')
    ca_flat, rt_flat = ca_mesh.flatten(), rt_mesh.flatten()

    # Calculate the theoretical probability distribution from CDM model
    model_probs = np.array([
        np.exp(CDM(params, np.array([[ca], [rt]]), A, T, P)) for ca, rt in zip(ca_flat, rt_flat)
    ]).reshape(ca_mesh.shape)

    # Normalize model probabilities to make it a valid CDF
    model_cdf = np.cumsum(np.cumsum(model_probs, axis=0), axis=1)
    model_cdf /= model_cdf[-1, -1]

    # Calculate empirical CDF in quadrants around each data point
    ks_statistic = 0
    for i in range(n_trials):
        x, y = CA[i], RT[i]

        # Quadrant boundaries for empirical CDFs
        quad_1 = np.sum((CA <= x) & (RT <= y)) / n_trials
        quad_2 = np.sum((CA > x) & (RT <= y)) / n_trials
        quad_3 = np.sum((CA <= x) & (RT > y)) / n_trials
        quad_4 = np.sum((CA > x) & (RT > y)) / n_trials
        
        # Find nearest model grid point indices for this data point
        x_idx = np.searchsorted(ca_values, x)
        y_idx = np.searchsorted(rt_values, y)
        
        # Model-based CDFs in quadrants around the grid point closest to (x, y)
        model_quad_1 = model_cdf[x_idx, y_idx]
        model_quad_2 = model_cdf[-1, y_idx] - model_cdf[x_idx, y_idx]
        model_quad_3 = model_cdf[x_idx, -1] - model_cdf[x_idx, y_idx]
        model_quad_4 = model_cdf[-1, -1] - (model_quad_1 + model_quad_2 + model_quad_3)

        # Calculate maximum difference in CDFs across quadrants
        d1 = abs(quad_1 - model_quad_1)
        d2 = abs(quad_2 - model_quad_2)
        d3 = abs(quad_3 - model_quad_3)
        d4 = abs(quad_4 - model_quad_4)
        
        # Update KS statistic with maximum deviation
        ks_statistic = max(ks_statistic, d1, d2, d3, d4)

    return ks_statistic

# TODO HAVEN'T BEEN TESTED

from scipy.optimize import differential_evolution

def optimize_parameters_ks(params_list, datasets, A, T, P, bounds, maxiter=10, popsize=5, seed=42):
    """
    Optimizes parameters for each parameter set and dataset using the 2D KS statistic.

    Parameters
    ----------
    params_list : list of lists
        A list of initial parameter sets for the CDM.
    datasets : dict
        A dictionary of simulated datasets where keys are parameter set indices and values are dictionaries
        with n_trials as keys and datasets as values.
    A : list
        Decision criterion range [lower bound, upper bound, step size] for series calculation in CDM.
    T : list
        Response time range [lower bound, upper bound, step size] for series calculation in CDM.
    P : numpy array
        Precomputed series for likelihood calculations.
    bounds : list of tuples
        Bounds for each parameter to optimize in `differential_evolution`.
    maxiter : int, optional
        Maximum number of iterations for `differential_evolution` (default is 10).
    popsize : int, optional
        Population size for `differential_evolution` (default is 5).
    seed : int, optional
        Seed for reproducibility (default is 42).

    Returns
    -------
    results : dict
        A dictionary containing optimization results. The outer keys are parameter set indices, the inner keys
        are `n_trials`, and values are the optimization result using KS statistic as the objective.
    """
    results = {}

    for i, initial_params in enumerate(params_list):
        param_results = {}
        
        for n_trials, data in datasets[i].items():
            # Define args specific for KS2D_CDM_quadrant
            args = (data, A, T, P)
            
            # Perform optimization using the KS2D_CDM_quadrant function directly as the objective
            ks_result = differential_evolution(
                KS2D_CDM,
                bounds,
                args=args,
                maxiter=maxiter,
                popsize=popsize,
                disp=True,
                seed=seed
            )

            # Store results for this n_trials
            param_results[n_trials] = ks_result

        # Store results for this parameter set
        results[i] = param_results

    return results

def calculate_rmse(true_values, recovered_values):
    """
    Calculate the Root Mean Squared Error (RMSE) between true and recovered values.
    """
    return np.sqrt(np.mean((np.array(true_values) - np.array(recovered_values)) ** 2))


def plot_parameter_recovery_by_trials(results, initial_params, param_index, param_name):
    """
    Plot parameter recovery and compute RMSE for a specific parameter, separately for each n_trials.

    Parameters
    ----------
    results : dict
        A dictionary containing optimization results.
    initial_params : list of lists
        A list of initial parameter sets for SimCDM.
    param_index : int
        The index of the parameter to plot.
    param_name : str
        The name of the parameter being plotted.
    """
    extracted_values = extract_parameter_values(results, param_index)

    for n_trials in {trial for param_set in extracted_values.values() for trial in param_set.keys()}:
        # Set up the plot
        fig, ax = plt.subplots(figsize=(6, 6))

        all_true_values = []
        all_recovered_values = []

        for param_set_idx, trials_results in extracted_values.items():
            if n_trials in trials_results:  # Ensure n_trials exists for this parameter set
                recovered_value = trials_results[n_trials]
                true_value = initial_params[param_set_idx][param_index]

                all_true_values.append(true_value)
                all_recovered_values.append(recovered_value)

        # Calculate RMSE for this n_trials
        rmse = calculate_rmse(all_true_values, all_recovered_values)

        # Scatter plot
        ax.scatter(all_true_values, all_recovered_values, s=50, alpha=0.7, color='green', label=f"RMSE = {rmse:.3f}")
        ax.plot([min(all_true_values), max(all_true_values)], 
                [min(all_true_values), max(all_true_values)], 
                color='black', linestyle='--')  # Identity line

        # Customize plot
        ax.set_title(f"{param_name} Recovery (N = {n_trials})")
        ax.set_xlabel("True Values")
        ax.set_ylabel("Recovered Values")
        ax.legend(loc="upper left", fontsize='small', frameon=True)
        ax.grid(True)

        plt.show()


def plot_all_parameters_by_trials(results, initial_params, param_names):
    """
    Plot parameter recovery separately for each n_trials for all parameters.

    Parameters
    ----------
    results : dict
        A dictionary containing optimization results.
    initial_params : list of lists
        A list of initial parameter sets for SimCDM.
    param_names : list of str
        A list of parameter names to plot.
    """
    for param_index, param_name in enumerate(param_names):
        plot_parameter_recovery_by_trials(results, initial_params, param_index, param_name)


param_names = [
    "decision_criterion", "sa", "drift_length", "drift_angle", 
    "radial_variability", "tangential_variability", 
    "non_decision_time", "non_decision_time_variability"
]


"""
Here lays the Simple sould of Simple CDM model, everything SIMPLE :)
"""

def simple_parameters(n):
    """
    Generates 'n' sets of parameters for the SimCDM function based on specified ranges.

    Parameters
    ----------
    n : int
        The number of parameter sets to generate.

    Returns
    -------
    param_sets : list of lists
        A list of parameter sets, where each set is a list of 8 parameter values.
    """
    simple_param_sets = []
    for _ in range(n):
        params = [
            np.random.uniform(0.5, 3.0),      # decision_criterion
            np.random.uniform(0.1, 3.0),      # drift_length
            np.random.uniform(0, 2*np.pi),    # drift_angle
            np.random.uniform(0.1, 1.0),      # non_decision_time
        ]
        simple_param_sets.append(params)
    
    return simple_param_sets

def simpleCDM(params, dt=0.001, n=100):
    """
    simpleCDM(params, dt=0.001, n=100)
    
    Simulate a dataset from the Simple Circular Diffusion Model (CDM).
    
    Parameters
    ----------
    params : list
        A list of four parameter values:
        [criterion (a), drift (v), bias (drift angle in radians), non-decision time (t0)].
        
    dt : float
        The time step for simulation.
        
    n : int
        The number of trials to simulate.

    Returns
    -------
    out : numpy array
        An array of shape (2, n) containing the choice angles and response times for all trials.
    """
    # Unpack parameters
    a = params[0]  # Decision criterion (radius)
    v = params[1]  # Drift rate (magnitude of the drift vector)
    bias = params[2]  # Drift direction (angle in radians)
    t0 = params[3]  # Non-decision time (constant)
    
    RT = []  # List to store response times
    CA = []  # List to store choice angles

    # Simulate each trial
    for _ in range(n):
        x = [0, 0]  # Initial position (x, y)
        t = 0  # Initialize time
        hit = False  # Initialize boundary crossing condition

        # Precompute drift vector components
        Vx = v * np.cos(bias)
        Vy = v * np.sin(bias)
        
        while not hit:
            # Update position with drift and Gaussian noise
            x[0] += Vx * dt + np.random.normal(0, np.sqrt(dt))
            x[1] += Vy * dt + np.random.normal(0, np.sqrt(dt))
            t += dt
            
            # Check if the boundary (criterion) is reached
            hit = (x[0]**2 + x[1]**2) >= a**2
        
        # Store results
        RT.append(t + t0)  # Add non-decision time
        CA.append(np.arctan2(x[1], x[0]))  # Store choice angle

    return np.array([CA, RT])

def simple_datasets(params_list, n_trials_list=[50, 100, 200, 300, 500, 1000]):
    """
    Simulate datasets using the simpleCDM function for each parameter set and trial count.

    Parameters
    ----------
    params_list : list of lists
        A list of parameter sets, where each set is a list of four parameter values:
        [criterion (a), drift (v), bias (drift angle in radians), non-decision time (t0)].
        
    n_trials_list : list of int, optional
        A list of trial counts to simulate data for each parameter set (default: [50, 100, 200, 300, 500, 1000]).

    Returns
    -------
    simulations : dict
        A dictionary with keys as parameter set indices. Each value is another dictionary where keys are `n_trials`
        values and values are the simulated datasets (numpy arrays) from simpleCDM.
    """
    simulations = {}

    for i, params in enumerate(params_list):
        trials_data = {}
        for n_trials in n_trials_list:
            trials_data[n_trials] = simpleCDM(params, n=n_trials)
        simulations[i] = trials_data

    return simulations

def Simple_CDM_Loglikelihoods(params, data, A, T, P):
    """
    Calculate the log-likelihood value of the data using a simplified parameter set.

    Parameters
    ----------
    params : list
        A list of real numbers containing the parameter values as [decision criterion (a), 
        drift length (v), drift bias (bias), non-decision time (t0)].

    data : numpy array
        An array of shape (2, n) containing the choice angle and response time pairs for all trials.

    A : list
        A list of real numbers as [lower bound of the range, higher bound of the range, step size] 
        indicating the range of the decision criterion at which the series is calculated.

    T : list
        A list of real numbers as [lower bound of the range, higher bound of the range, step size] 
        indicating the range of the response time at which the series is calculated.

    P : numpy array
        A 2D array of calculated values of the series for the range of decision criterion and response time values.

    Returns
    -------
    out : float
        Logarithm of the likelihood value.
    """

    # Unpack only the necessary parameters
    a = params[0]
    v = params[1]
    bias = params[2]
    t0 = params[3]

    # Extract choice angles and response times from data
    CA, RT = data

    # Prepare ranges for decision criterion and response time
    a_ = np.arange(A[0], A[1], A[2])
    t_ = np.arange(T[0], T[1], T[2])
    da = A[2]
    dt = T[2]

    # Check if the minimum response time is valid
    if int((min(RT) - t0) / dt) <= 0:
        return -np.inf

    # Initialize log-likelihood
    ll = 0

    # Compute log-likelihood for each trial
    for i in range(len(CA)):
        if RT[i] >= t0:
            T_ = t_[int((RT[i] - t0) / dt):]
            P_ = P[:, int((RT[i] - t0) / dt):]
        else:
            T_ = t_[:int((RT[i] - t0) / dt) + 1]
            P_ = P[:, :int((RT[i] - t0) / dt) + 1]

        T_ = T_.reshape((1, len(T_)))
        A_ = a_.reshape((len(a_), 1))
        
        # Calculate Z
        Z = np.exp((-v**2 * T_ + A_**2 * np.cos(CA[i] - bias)**2 / 2 +
                    2 * v * A_ * np.cos(CA[i] - bias)) / 2) / np.sqrt(2 * np.pi * T_)

        # Ignore regions where P is zero
        Z[P_ == 0] = 0
        
        # Add log-likelihood contribution for this trial
        ll += np.log((np.mean(Z * P_) + 1e-10))
    
    return ll

from scipy.optimize import differential_evolution
import numpy as np

def optimize_parameters_simple_cdm_mle(params_list, datasets, A, T, P, bounds, maxiter=10, popsize=5):
    """
    Optimizes parameters for each parameter set and dataset using Maximum Likelihood Estimation.

    Parameters
    ----------
    params_list : list of lists
        A list of initial parameter sets for the simplified CDM.
    datasets : dict
        A dictionary of simulated datasets where keys are parameter set indices and values are dictionaries
        with n_trials as keys and datasets as values.
    A : list
        Decision criterion range [lower bound, upper bound, step size] for series calculation in CDM.
    T : list
        Response time range [lower bound, upper bound, step size] for series calculation in CDM.
    P : numpy array
        Precomputed series for likelihood calculations.
    bounds : list of tuples
        Bounds for each parameter to optimize in differential_evolution.
    maxiter : int, optional
        Maximum number of iterations for differential_evolution (default is 10).
    popsize : int, optional
        Population size for differential_evolution (default is 5).

    Returns
    -------
    results : dict
        A dictionary containing optimization results. The outer keys are parameter set indices, the inner keys
        are `n_trials`, and values are the optimization result for the log-likelihood objective.
    """
    results = {}

    for i, initial_params in enumerate(params_list):
        param_results = {}

        for n_trials, data in datasets[i].items():
            # Perform log-likelihood optimization
            log_likelihood_result = differential_evolution(
                lambda params: -Simple_CDM_Loglikelihoods(params, data, A, T, P),  # Minimizing negative log-likelihood
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

"""
Everything Paralleled
"""

def compute_series_for_a_t(args):
    """Compute the series for a single pair of (a, t)."""
    a, t, j0, J1, n, r = args
    a_ = mpmath.mpf(a)
    t_ = mpmath.mpf(t)
    a2 = a_**2
    series = 0
    for k in range(n):
        term = j0[k] / J1[k] * mpmath.exp(-j0[k]**2 * t_ / (2 * a2))
        series += term
        if abs(term / series) < r:
            break
    series = series / a2 / (2 * mpmath.pi)
    return max(series, 0)  # Ensure no negative values

def Series_Parallel(f, n=50, r=1e-10, dps=5):
    """
    Optimized parallelized version of the Series function with progress tracking and float output.
    """
    mpmath.mp.dps = dps

    # Precompute Bessel values
    j0 = np.array([mpmath.besseljzero(0, i + 1) for i in range(n)], dtype=mpmath.mpf)
    J1 = np.array([mpmath.besselj(1, j0[i]) for i in range(n)], dtype=mpmath.mpf)

    # Generate ranges for a and t
    a_values = np.arange(A[0], A[1], A[2])
    t_values = np.arange(T[0], T[1], T[2])

    # Prepare tasks for parallel processing
    tasks = [(a, t, j0, J1, n, r) for a, t in product(a_values, t_values)]

    # Use multiprocessing to calculate the series
    results = []
    with Pool(cpu_count()) as pool:
        # Wrap pool.imap with tqdm for progress tracking
        for result in tqdm(pool.imap(compute_series_for_a_t, tasks), total=len(tasks), desc="Computing Series"):
            results.append(result)

    # Reshape results into 2D array and convert to float
    P = np.array(results).reshape(len(a_values), len(t_values))
    P = P.astype(float)  # Convert all mpf values to float
    return P

def optimize_task(args):
    """Optimization task for a single parameter set and dataset."""
    i, n_trials, data, A, T, P, bounds, maxiter, popsize = args
    result = differential_evolution(
        lambda params: -CDM(params, data, A, T, P),  # Minimizing negative log-likelihood
        bounds,
        args=(),
        maxiter=maxiter,
        popsize=popsize,
        disp=False,  # Disable individual DE progress
    )
    return i, n_trials, result

def optimize_parameters_mle_parallel(params_list, datasets, A, T, P, bounds, maxiter=10, popsize=5, seed=42):
    """
    Parallelized and trackable parameter optimization function.
    """
    results = {}
    tasks = []

    # Prepare tasks for parallel processing
    for i, initial_params in enumerate(params_list):
        for n_trials, data in datasets[i].items():
            tasks.append((i, n_trials, data, A, T, P, bounds, maxiter, popsize, seed))

    # Use multiprocessing to process tasks in parallel
    with Pool(cpu_count()) as pool:
        # Wrap pool.imap with tqdm for progress tracking
        for i, n_trials, log_likelihood_result in tqdm(
            pool.imap(optimize_task, tasks),
            total=len(tasks),
            desc="Optimizing Parameters"
        ):
            # Organize results by parameter set and n_trials
            if i not in results:
                results[i] = {}
            results[i][n_trials] = log_likelihood_result

    return results
