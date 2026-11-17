---
title: 'Gaussian processes'
description: 'GP regression with Cholesky, kernel choice, hyperparameter fitting by marginal likelihood, and scalable approximations: inducing points, structured kernels, random features.'
aliases:
  [
    Gaussian process,
    GP regression,
    kriging,
    kernel,
    covariance function,
    squared exponential,
    RBF kernel,
    Matern,
    periodic kernel,
    marginal likelihood,
    hyperparameters,
    inducing points,
    sparse GP,
    SVGP,
    FITC,
    random Fourier features,
    Nystrom,
    KISS-GP,
    GP classification,
    Bayesian optimization
  ]
tags: [Bayesian, regression, kernels, surrogate]
order: 5
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- A GP is a prior over functions defined by a mean and a **kernel** $k(x, x')$. With Gaussian noise, the posterior is closed form: one **Cholesky** of the $n\times n$ kernel matrix, $O(n^3)$ time, $O(n^2)$ memory. Fine up to $n \approx 10^4$.
- **Kernel choice** encodes assumptions: Matérn-5/2 is the robust default for physical responses; squared exponential (RBF) assumes infinitely smooth functions.
- Fit hyperparameters (length scales, signal variance, noise) by maximising the **log marginal likelihood**, with several restarts, in log-parameter space.
- **Large $n$:** inducing points (SVGP, $O(nm^2)$), structured kernels on grids (Kronecker/Toeplitz), or random features. 1D time series: state-space form, $O(n)$.
- GPs give calibrated uncertainty for smooth, low-dimensional problems (≲ 10–20 inputs); the basis of Bayesian optimisation and many surrogates.

## Decision table

| Kernel                   | Formula ($r = \lVert x - x'\rVert/\ell$)                  | Sample paths           | Use when                            |
| ------------------------ | --------------------------------------------------------- | ---------------------- | ----------------------------------- |
| Squared exponential      | $\sigma^2e^{-r^2/2}$                                      | infinitely smooth      | very smooth functions               |
| Matérn-3/2               | $\sigma^2(1 + \sqrt3 r)e^{-\sqrt3 r}$                     | once differentiable    | rough physical responses            |
| Matérn-5/2               | $\sigma^2(1 + \sqrt5 r + \tfrac53 r^2)e^{-\sqrt5 r}$      | twice differentiable   | **default** for surrogates / BO     |
| Exponential (Matérn-1/2) | $\sigma^2e^{-r}$                                          | continuous, rough (OU) | rough processes, Ornstein–Uhlenbeck |
| Periodic                 | $\sigma^2\exp(-2\sin^2(\pi\lvert x - x'\rvert/p)/\ell^2)$ | periodic               | seasonal / oscillatory signals      |
| Linear                   | $\sigma^2x^\top x'$                                       | linear functions       | Bayesian linear regression, trend   |
| ARD                      | separate $\ell_j$ per input                               | —                      | automatic relevance of inputs       |
| Sum / product            | $k_1 + k_2$, $k_1k_2$                                     | combined structure     | trend + periodic + noise            |

| Scaling method                 | Cost                   | Use when                                 |
| ------------------------------ | ---------------------- | ---------------------------------------- |
| Exact (Cholesky)               | $O(n^3)$               | $n \lesssim 10^4$                        |
| Exact with CG + GPU (BBMM)     | $O(n^2)$ per iteration | $n \sim 10^5$ with GPUs                  |
| Inducing points (SGPR, SVGP)   | $O(nm^2)$ / minibatch  | large $n$, smooth functions              |
| State-space (Kalman)           | $O(n)$                 | 1D (time) with Matérn kernels            |
| Kronecker / Toeplitz (KISS-GP) | ~$O(n\log n)$          | gridded or low-dimensional inputs        |
| Random Fourier features        | $O(nD^2)$              | stationary kernels, quick approximations |
| Hierarchical matrices (HODLR)  | $O(n\log^2 n)$         | low-dimensional inputs, exact-ish        |

## Variants

### GP regression

With training data $X, y$, noise variance $\sigma_n^2$, $K = k(X, X) + \sigma_n^2I$:

1. Cholesky $K = LL^\top$.
2. $\alpha = L^\top\backslash(L\backslash y)$.
3. **Posterior mean** at test points: $\mu_* = K_{*X}\alpha$.
4. **Posterior variance:** $v = L\backslash K_{X*}$, $\sigma_*^2 = k_{**} - v^\top v$.
5. **Log marginal likelihood:** $-\tfrac12y^\top\alpha - \sum_i\log L_{ii} - \tfrac n2\log 2\pi$.

### Hyperparameter fitting

- Maximise the log marginal likelihood w.r.t. $\theta = (\ell, \sigma^2, \sigma_n^2)$ with L-BFGS; gradient $\tfrac12\operatorname{tr}\big((\alpha\alpha^\top - K^{-1})\,\partial K/\partial\theta\big)$.
- Optimise in log space; use multiple restarts (the landscape is multimodal: short length scale + low noise vs long length scale + high noise).
- Fully Bayesian alternative: put priors on $\theta$ and sample (HMC), integrating over hyperparameter uncertainty.

### Sparse / inducing-point GPs

- Summarise the function by values at $m \ll n$ inducing inputs $Z$.
- **SGPR (Titsias):** variational lower bound, exact data likelihood approximation, $O(nm^2)$.
- **SVGP (Hensman):** stochastic variational version with minibatches; works for $n$ in the millions and non-Gaussian likelihoods.
- FITC/DTC: older approximations, FITC can underestimate noise.

### Non-Gaussian likelihoods

- Classification, counts: posterior no longer Gaussian. Approximate with Laplace, expectation propagation, or variational inference; or MCMC for small $n$.

## Other lenses

- **Kernel methods.** The GP posterior mean equals kernel ridge regression with regulariser $\sigma_n^2$; the kernel defines an RKHS. Splines are GPs with particular kernels.
- **Linear algebra.** Everything reduces to solves and log-determinants with $K$: Cholesky, CG + stochastic Lanczos quadrature for large $K$, structure exploitation (Toeplitz, Kronecker, H-matrices; see [hierarchical matrices](/ref/solve/hierarchical-matrices-fmm)).
- **SDEs.** Matérn GPs in 1D are stationary solutions of linear SDEs; GP regression becomes Kalman filtering/smoothing in $O(n)$ (see [Kalman filters](/ref/signals/kalman-filters)).
- **Neural networks.** Infinitely wide random networks are GPs (NNGP, neural tangent kernel).

## Pitfalls & diagnostics

- **Ill-conditioned $K$** (nearly duplicate inputs, long length scales, tiny noise): add jitter ($10^{-8}$–$10^{-6}\times$ the signal variance) or fix a minimum noise.
- **Input scaling:** standardise inputs and outputs; length scales are then interpretable and optimisers behave.
- **Stationary kernels can't extrapolate trends;** add a mean function or linear kernel.
- **Overconfident uncertainties** usually mean wrong hyperparameters or model misspecification; check calibration on held-out data (coverage of predictive intervals).
- **High dimensions** (≳ 20 inputs) dilute the kernel's notion of similarity; use ARD, additive kernels or feature learning.

## See also

- [Surrogate models](/ref/approx/surrogates)
- [Global & derivative-free optimisation](/ref/optimise/global-derivative-free) (Bayesian optimisation)
- [Kalman filters](/ref/signals/kalman-filters)
- [Hierarchical matrices & FMM](/ref/solve/hierarchical-matrices-fmm)
- [Matrix calculus](/ref/diff/matrix-calculus)

**References:**

- Rasmussen & Williams, _Gaussian Processes for Machine Learning_.
- Hensman, Fusi & Lawrence, "Gaussian processes for big data", UAI (2013).
- Gardner et al., "GPyTorch: blackbox matrix-matrix Gaussian process inference with GPU acceleration", NeurIPS (2018).
