# Learning the bistable cortical dynamics of the sleep-onset period

This repository contains code to reproduce analyses and selected figures for the preprint **“Learning the bistable cortical dynamics of the sleep-onset period.”** The paper introduces a minimally parameterized stochastic dynamical model in which a slowly varying control parameter drives the wake-to-sleep transition, capturing bistable cortical dynamics at sleep onset.

## Preprint
- bioRxiv: https://www.biorxiv.org/content/10.1101/2025.07.17.665340v3

## Code requirements for figures
- To generate **Figures 7, 11, and 12**, download the **MCMC results** from Zenodo and make them available to the code (see plots/fig7_real_validation.m for expected paths):  
  DOI: https://doi.org/10.5281/zenodo.16907957

## Setup
Create the conda environment defined in `environment.yml`:
```bash
conda env create -f environment.yml
# then activate the environment named inside environment.yml, e.g.:
conda activate <env-name>
```