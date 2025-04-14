
# GPy Extension for Gaussian Process Dynamical Models

An extension to the GPy library for Gaussian Process Dynamical Models (GPDMs) and hierarchical GPLVMs with a focus on human movement modeling.

## Features

- Hierarchical GPLVMs: Advanced latent variable models with hierarchical structure
- Node-based computation: Computational framework for processing factorized hierarchical model joint distributions
- Modular back-constraint framework: Including the novel Gaussian Process Back-Constraint (GP BC)
- GPDM implementations:
  - Standard GPDMs
  - GPDM sparsification
  - GPDM mixture models (GPDMMs)
- Comprehensive initialization procedures: For optimizing latent space organization

## Analysis Tools

- Data management:
  - Pre-processing pipelines
  - Format conversion
  - Data storage utilities
- Visualization:
  - Latent space plotting
  - 2D/3D articulated human motion animation
- Performance evaluation:
  - Comprehensive metrics calculation

## Statistical Framework

- Parallel simulation processing
- Hyperparameter optimization (Bayesian, grid search)
- Monte Carlo cross-validation


Note: ./HGPLVM/GPy_mods contains files modified from the GPy Gaussian process package from Sheffield Machine Learning. I have maintained their copyright assertions in the files, but we have no affiliation with their lab or research.