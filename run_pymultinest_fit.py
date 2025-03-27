# Content of run_pymultinest_fit.py:
# run_pymultinest_fit.py

import numpy as np
import pandas as pd
import pymultinest
import os

# Load data
csv_path = "tuning_data.csv"
df = pd.read_csv(csv_path)
s = df["s"].values
response = df["response"].values
error = df["error"].values


def tuning_curve(s, r_max, s_max, sigma_f):
    # Tuning curve model
    return r_max * np.exp(-0.5 * ((s - s_max) / sigma_f)**2)



def prior_transform(cube, ndim, nparams):
    # Prior transform for MultiNest: Mapping Unit Cube to Parameter Space
    # This function defines how PyMultiNest transforms a unit cube (i.e., uniform samples in [0, 1]^3) into the actual parameter space.
    # Each element of `cube` is scaled to its respective prior range:
    # - r_max in [10, 70] 
    # - s_max
    # - sigma_f
    # This approach corresponds to using uniform priors over these ranges, which is often suitable when we lack strong prior knowledge.

    r_min, r_max = 10, 70
    s_min, s_max = -50, 50
    sigma_f_min, sigma_f_max = 1, 40

    cube[0] = r_min + (r_max - r_min) * cube[0]
    cube[1] = s_min + (s_max - s_min) * cube[1]
    cube[2] = sigma_f_min + (sigma_f_max - sigma_f_min) * cube[2]


def loglike(cube, ndim, nparams):
    # Gaussian Log-Likelihood
    # This function computes the log-likelihood of the model given the data. It assumes Gaussian observation noise and compares the model prediction to the observed neural responses.
    # Steps:
    # - unpack parameters: r_mx, s_max, sigma_f
    # - evaluate the tuning curve at each stimulus value
    # - compute the residuals normalized by the known noise
    # - return the log-likelihood using the Gaussian error model logL

    r_max, s_max, sigma_f = cube[0], cube[1], cube[2]
    model = tuning_curve(s, r_max, s_max, sigma_f)
    residual = (response - model) / error
    logL = -0.5 * np.sum(residual**2 + np.log(2 * np.pi * error**2))
    return logL


# Output folder for chains (directory set up)
# This ensures the output directory (`chains/`) exists. PyMultiNest will store its results here, including:
# - posterior samples
# - evidence estimates
# - diagnostic information
# We also define a prefix (`fit_`) for all output files in this directory.
output_dir = "chains"
os.makedirs(output_dir, exist_ok=True)
output_prefix = os.path.join(output_dir, "fit_")


# Run nested sampling
# This is the main call to PyMultiNest. It performs Bayesian inference using nested sampling with:
# - n_dims = 3 : three model parameters
# - n_live_points = 500 : number of live points for exploration (more = more accurate, but slower)
# - resume = False : do not restart from previous runs
# The output includes posterior samples and an estimate of the Bayesian evidence.
pymultinest.run(loglike, prior_transform, n_dims=3,
                outputfiles_basename=output_prefix,
                resume=False, verbose=True,
                n_live_points=500)

# Save posterior samples for later use
# After the run, we load the posterior samples (equal-weighted) from PyMultiNest’s output and save them as a .npy` # file for easier use in the analysis and plotting phase.
samples_file = output_prefix + "post_equal_weights.dat"
posterior_samples = np.loadtxt(samples_file)
np.save("posterior_samples.npy", posterior_samples)

