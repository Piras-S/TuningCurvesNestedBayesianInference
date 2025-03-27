#  Bayesian Inference of Neural Tuning Curves with Nested Sampling

This project demonstrates how to simulate and analyze neural tuning curve data using **Bayesian inference** via **nested sampling**. It combines theory, code, and visual intuition to explain how posterior distributions are obtained, and why they should be interpreted with care.

<p align="center">
  <img src="figures/nested_sampling.gif" width="500"/>
</p>

---

## Project Overview

I simulate the firing response of a neuron to different stimulus angles (e.g., orientation of a visual stimulus), assuming a Gaussian-shaped tuning curve. Using synthetic data, I then perform **Bayesian parameter estimation** with PyMultiNest to recover the model parameters:
- $r_{\text{max}}$: maximum firing rate,
- $s_{\text{max}}$: preferred stimulus orientation,
- $\sigma_f$: tuning width.


---

## Concepts Covered

- Bayesian inference: likelihood, priors, posteriors, evidence
- Nested sampling algorithm (with theory + toy implementation)
- Parameter estimation in a neural tuning model
- Posterior uncertainty, model mismatch, and identifiability
- Practical diagnostics for PyMultiNest fits

---

## Visual Highlights

### Toy Nested Sampling Animation
A minimal 2D implementation of nested sampling illustrates the algorithm’s core idea: removing low-likelihood regions and progressively zooming in on high-probability space.

<p align="center">
  <img src="figures/nested_sampling.gif" width="400"/>
</p>


---

**Dependencies**:
   numpy matplotlib pandas pymultinest corner imageio
   
   

**References**

Dayan, P., & Abbott, L. F. (2001). Theoretical Neuroscience: Computational and Mathematical Modeling of Neural Systems. MIT Press.

Feroz, F., Hobson, M. P., & Bridges, M. (2009). MultiNest: an efficient and robust Bayesian inference tool for cosmology and particle physics.

---

## Project Structure
```bash
├── simulation.ipynb # Generate and visualize synthetic data 
├── fitting.ipynb # Educational toy example of nested sampling  + Run and analyze full model fitting 
├── run_pymultinest_fit.py # Script to perform inference with PyMultiNest 
├── tuning_data.csv # Simulated dataset 
├── figures/ 
└── nested_sampling.gif # Animation of toy nested sampling

