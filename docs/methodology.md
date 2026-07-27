# Methodology

## 1. Cox Proportional Hazards Model
The hazard function for subject $i$ at time $t$ is modeled as:
$$\lambda(t | X_i) = \lambda_0(t) \exp(\beta^T X_i)$$

where $\lambda_0(t)$ is the baseline hazard and $\beta$ represents log hazard ratios.

## 2. Bayesian Cox Formulation
In the Bayesian framework:
- **Priors**: 
  $$\beta_k \sim \mathcal{N}(\mu_0, \sigma_0^2)$$
  $$\lambda_0(t) \sim \text{Gamma}(\alpha, \beta) \quad \text{or piecewise constant hazard priors}$$

- **Likelihood**: 
  Calculated using the partial likelihood formulation or piecewise exponential formulation for right-censored data:
  $$L(\beta) = \prod_{i: \delta_i=1} \frac{\exp(\beta^T X_i)}{\sum_{j \in R(t_i)} \exp(\beta^T X_j)}$$

- **MCMC Sampling**: Hamiltonian Monte Carlo (HMC) / No-U-Turn Sampler (NUTS) with Gelman-Rubin ($\hat{R}$) convergence checks.

## 3. Evaluation Metrics
- **Harrell's Concordance Index (C-Index)**: Measures discriminatory power.
- **Brier Score & Integrated Brier Score (IBS)**: Evaluates calibration and prediction accuracy across time.
- **Uncertainty Bounds**: 95% Highest Posterior Density (HPD) intervals for survival probabilities and hazard ratios.
