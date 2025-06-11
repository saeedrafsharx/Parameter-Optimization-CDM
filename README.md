# Circular Diffusion Model (CDM) Optimization Framework

This repository contains a comprehensive framework for simulating, optimizing, and analyzing Circular Diffusion Models (CDMs) using various optimization techniques. The framework is organized into multiple folders, each focusing on a specific optimization method or utility.

## Repository Structure

### 1. `Chi-Squared_Optimization/`
This folder contains the implementation of the Chi-Squared optimization method for CDMs.

- **`ChiSqauredCDM.py`**: Implements the Chi-Squared objective function and optimization routines for parameter recovery in CDMs. It uses binned data to calculate the Chi-Squared statistic and optimize parameters using the `scipy.optimize.differential_evolution` method.

---

### 2. `EZCDM_Optimization/`
This folder contains the implementation of the EZ-CDM method for parameter recovery.

- **`EZCDM.py`**: Implements a simplified parameter recovery method for CDMs using the EZ-CDM approach. It calculates the likelihood of observed data and optimizes parameters using differential evolution.

---

### 4. `KS_Optimization/`
This folder contains the implementation of the Kolmogorov-Smirnov (KS) optimization method for CDMs.

- **`KSOptimizeCDM.py`**: Implements the KS statistic as an objective function for parameter recovery in CDMs. It calculates the maximum difference between empirical and model-based cumulative distributions.

---

### 5. `MLE_Optimization/`
This folder contains the implementation of Maximum Likelihood Estimation (MLE) for CDMs, along with utility functions and testing scripts.

- **`MLECDM.py`**: Implements the MLE objective function and optimization routines for CDMs. It uses differential evolution to optimize parameters.
- **`Utils.py`**: Contains utility functions for simulation, parameter generation, and optimization. Includes implementations for CDM simulation, series calculations, and plotting.
- **`Test.ipynb`**: A Jupyter Notebook for testing MLE recovery and parameter optimization. It demonstrates the usage of the framework with simulated datasets.

---

### 6. `Literature/`
This folder contains relevant research papers and references used in the development of the framework.

- Example files:
  - **`10-Smith 2016.pdf`**
  - **`11-Smith 2020 Circular Diffusion model.pdf`**
  - **`EZ-CDM Fast, simple, robust, and accurate estimation of circular.pdf`**

---

## Key Features

- **Simulation**: Generate synthetic datasets using the Circular Diffusion Model.
- **Optimization Methods**:
  - Maximum Likelihood Estimation (MLE)
  - Chi-Squared Optimization
  - Kolmogorov-Smirnov (KS) Optimization
  - EZ-CDM Simplified Optimization
  - Hierarchical Sequential Diffusion Model (HSDM) Optimization
- **Parallelization**: Leverages multiprocessing for efficient computation.
- **Visualization**: Includes plotting utilities for parameter recovery and comparison.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Required libraries: `numpy`, `scipy`, `matplotlib`, `tqdm`, `mpmath`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Parameter-Optimization-CDM/CDM-Optimization-Framework.git
   cd CDM-Optimization-Framework
