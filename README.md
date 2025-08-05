Epoch-Dependent QSTF Cosmology Model (v3.2 - MCMC Analysis)
Author: Tõnis Leissoo
Date: August 5, 2025

Overview
This repository contains the Python implementation of the Quantum Spacetime Fluid (QSTF) model, a candidate physical theory for emergent spacetime. This single, unified framework proposes that both dark energy and dark matter arise from the quantum mechanical properties of spacetime itself, which is modeled as a superfluid governed by the Gross-Pitaevskii Equation (GPE).

The primary achievement of this model and script is its ability to simultaneously resolve three of the most significant problems in modern cosmology:

The Hubble Tension: By modeling dark energy as the bulk evolution of the spacetime fluid, which has an epoch-dependent equation of state.

The Core-Cusp Problem: By modeling dark matter halos as stable, cored solitons (localized condensates) of the fluid.

The S8 Tension: By modeling the repulsive self-interaction and effective quantum pressure between these solitons on cosmological scales.

This script performs a full Markov Chain Monte Carlo (MCMC) statistical analysis to fit the QSTF model to real-world cosmological data and then uses the best-fit parameters to make new, testable physical predictions.

Theoretical Framework
The entire model is derived from the Gross-Pitaevskii Equation (GPE), a well-established equation in quantum mechanics. The core hypothesis is that spacetime itself is a quantum condensate, and its dynamics are governed by the GPE.

In this framework:

Dark Energy is the manifestation of the background, bulk energy of the spacetime fluid.

Dark Matter is the manifestation of localized, gravitationally self-trapped excitations (solitons) within that same fluid.

For a complete mathematical description and the theoretical underpinnings, please refer to the accompanying papers: QSTF.pdf and Definitive Analysis of the QSTF Model.pdf.

How the Script Works: A Step-by-Step Guide
The script is designed to be a complete workflow, from data analysis to physical prediction. When executed, it performs the following steps:

Step 1: Initialization
The script begins by loading the observational data from multiple cosmological surveys into memory. This includes Planck 2018 distance priors, SH0ES H0 measurements, and other relevant datasets. It also sets up the parameter space, defining the prior bounds for the 13 free parameters of the QSTF model.

Step 2: MCMC Sampler Setup
The script initializes the emcee MCMC sampler. A large number of "walkers" (e.g., 100) are created. Each walker is an independent explorer in the 13-dimensional parameter space, starting from a slightly different initial position centered around a plausible guess.

Step 3: The MCMC Likelihood Loop
The main computational task begins as the sampler runs for a large number of steps (e.g., 10,000). For each step, every walker proposes a move to a new set of parameters. The script then evaluates the "goodness" of this new point by calculating the log_probability, which involves:

Checking the Prior: It first checks if the new parameters are within the allowed physical bounds. If not, the point is rejected.

Calculating the Likelihood: If the prior is valid, the script calculates the total chi-squared (
chi 
2
 ) value. This involves:

Computing the QSTF dark energy evolution, w(z).

Numerically integrating the Friedmann equation to get the expansion history, H(z).

Calculating the theoretical values for all observables (e.g., cosmic distances).

Comparing these theoretical values to the real observational data to get the final 
chi 
2
 .

The log_probability is then returned to the sampler, which decides whether to accept or reject the proposed step.

Step 4: Convergence and Diagnostics
After the run is complete, the script performs essential diagnostic checks to ensure the MCMC has converged correctly:

It calculates and displays the mean acceptance fraction. A value between ~0.2 and ~0.5 indicates the sampler was exploring efficiently.

It generates a trace plot (qstf_trace_plot.png), which visualizes the path of every walker for every parameter. This allows for a visual check that the walkers have explored the full parameter space and not become "stuck."

Step 5: Statistical Results
The script processes the output of the MCMC run (the "chain") after discarding the initial "burn-in" phase.

It calculates the final parameter constraints, reporting the median value and the 1-sigma (68%) credible intervals for all 13 parameters.

It generates a corner plot (qstf_corner_plot.png), the standard visualization for MCMC results, showing the full posterior probability distribution.

Step 6: Physical Predictions
Finally, the script takes the best-fit parameters (the median values from the MCMC) and uses them to make new physical predictions that were not part of the fitting data. This demonstrates the predictive power of the theory:

Core-Cusp Solution: It calculates the predicted core radius (in kpc) of a Milky Way-sized dark matter halo.

S8 Tension Solution: It calculates the predicted suppression of the matter power spectrum at a key cosmological scale (k=0.1 h/Mpc).

How to Run
Ensure you have the required Python dependencies installed. Then, simply run the script from your terminal:

python qstf_mcmc_analysis.py

Note: The MCMC analysis is computationally intensive and may take a significant amount of time to complete (several hours to over a day), depending on your system's hardware. The script will save its progress to qstf_mcmc_chain.h5.

Expected Output
After the run finishes, you will see a detailed summary printed to your console, showing both the statistical diagnostics and the final results. You will also have two new image files in your directory: qstf_trace_plot.png and qstf_corner_plot.png.

Dependencies
numpy

scipy

emcee

corner

matplotlib

h5py (for saving MCMC progress)
