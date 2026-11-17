---
title: 'Surrogate models'
description: 'Replacing expensive simulations with cheap approximations: RBFs, polynomial chaos, Gaussian processes, neural networks, and active learning to choose where to sample.'
aliases:
  [
    surrogate model,
    metamodel,
    emulator,
    response surface,
    radial basis functions,
    RBF interpolation,
    polynomial chaos,
    PCE,
    stochastic collocation,
    Gaussian process emulator,
    neural surrogate,
    active learning,
    adaptive sampling,
    design of experiments,
    Latin hypercube,
    space-filling design,
    multi-fidelity,
    cokriging
  ]
tags: [surrogate, approximation, uncertainty quantification]
order: 6
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- A surrogate approximates an expensive input → output map so that optimisation, UQ, sensitivity analysis or inference can call it thousands of times.
- **Few expensive runs (≲ 1000), low dimension (≲ 20), need uncertainty:** Gaussian process.
- **Smooth response in uncertain inputs with known distributions:** polynomial chaos (Sobol indices and moments for free).
- **Scattered interpolation, cheap to fit:** radial basis functions.
- **Many runs available, high-dimensional or structured outputs (fields, time series):** neural networks (or POD + regression).
- Choose training points with a **space-filling design** (LHS, Sobol), then refine adaptively (active learning) where the surrogate is uncertain or where it matters for the task.

## Decision table

| Surrogate                   | Training data needed              | Dimension            | Uncertainty            | Use when                                         |
| --------------------------- | --------------------------------- | -------------------- | ---------------------- | ------------------------------------------------ |
| Polynomial response surface | $O(d^2)$                          | low                  | regression intervals   | local, smooth, quadratic-ish responses           |
| RBF interpolation           | $\sim 10d$–$100d$                 | ≲ 20                 | none (unless GP view)  | cheap, accurate interpolation; optimisation      |
| Gaussian process            | $\sim 10d$–$10^4$                 | ≲ 20                 | yes, calibrated        | expensive models, Bayesian optimisation, UQ      |
| Polynomial chaos (PCE)      | $\binom{d + p}{p}$ × oversampling | ≲ 10–30 (sparse PCE) | via input distribution | UQ with known input distributions, Sobol indices |
| Neural network              | $10^3$–$10^6$                     | high                 | ensembles / Bayesian   | lots of data, high-dimensional or field outputs  |
| POD + regression (ROM)      | snapshots                         | high (outputs)       | limited                | field outputs (PDE solutions)                    |
| Multi-fidelity (cokriging)  | many cheap + few expensive runs   | low–moderate         | yes                    | hierarchy of model fidelities                    |

## Variants

### Designs of experiments

- **Latin hypercube:** each input's range split into $N$ strata, one sample per stratum; optimise for maximin distance.
- **Sobol / Halton sequences:** low discrepancy, extendable (add points without redesign) (see [quasi-Monte Carlo](/ref/random/quasi-monte-carlo)).
- Sample in transformed coordinates (log for rate constants, normalised to $[0, 1]$).

### Radial basis functions

- $s(x) = \sum_j w_j\phi(\lVert x - x_j\rVert) + p(x)$ with low-degree polynomial $p$; solve the $(n + q)$ linear system for weights.
- Kernels: thin plate spline $r^2\log r$, cubic $r^3$ (no shape parameter, conditionally positive definite), Gaussian and multiquadric (shape parameter $\varepsilon$: flat limit accurate but ill-conditioned).
- Equivalent to GP mean prediction for a matching kernel; no uncertainty unless treated as a GP.

### Polynomial chaos expansions

- Expand $y(\xi) \approx \sum_\alpha c_\alpha\Psi_\alpha(\xi)$ in polynomials orthogonal w.r.t. the input distribution (Hermite for Gaussian, Legendre for uniform, …).
- **Coefficients:** projection with (sparse-grid) quadrature, or regression on samples (least squares; LASSO / LARS for sparse PCE in higher dimensions).
- **Free outputs:** mean $= c_0$, variance $= \sum_{\alpha \ne 0}c_\alpha^2\lVert\Psi_\alpha\rVert^2$, Sobol indices by grouping coefficients.

### Neural surrogates

- MLPs for scalar outputs; convolutional/graph/operator networks for fields (DeepONet, FNO; see [physics-informed ML](/ref/learn/physics-informed-ml)).
- Need far more data than GPs; uncertainty via deep ensembles; physics constraints via architecture or loss.

### Active learning

1. Fit the surrogate on an initial design.
2. Choose the next point(s) by an acquisition criterion:
   - maximum predictive variance (global accuracy)
   - expected improvement / UCB (optimisation, see [global optimisation](/ref/optimise/global-derivative-free))
   - integrated variance reduction or expected information gain (UQ, inference)
3. Run the expensive model there, refit, repeat until the budget or accuracy target is reached.

### Multi-fidelity

- Combine cheap low-fidelity models (coarse mesh, reduced model) with few high-fidelity runs: autoregressive cokriging $f_{\text{high}} = \rho f_{\text{low}} + \delta$, or multi-fidelity neural networks.

## Other lenses

- **Approximation theory.** Surrogates are function approximation from scattered samples; the curse of dimensionality limits all of them unless the function has low effective dimension or structure (see [high-dimensional numerics](/ref/approx/high-dimensional-numerics)).
- **Bayesian inference.** GP surrogates of the log-likelihood accelerate MCMC or ABC for expensive models; the surrogate's uncertainty must be propagated (see [likelihood-free inference](/ref/bayes/likelihood-free)).
- **Model reduction.** Projection-based ROMs keep the physics equations (cheaper but intrusive); data-driven surrogates are non-intrusive black boxes (see [timescale reduction](/ref/dynamics/timescale-reduction)).

## Pitfalls & diagnostics

- **Validate on held-out runs,** not training error; report error relative to output variability ($Q^2$, normalised RMSE) and coverage of uncertainty intervals.
- **Extrapolation** outside the training domain is unreliable for every surrogate type.
- **Discontinuities and regime changes** (bifurcations, thresholds) break smooth surrogates; partition the domain or use classifiers for regimes.
- **Noisy simulators** (stochastic models, adaptive solvers): include a noise term rather than interpolating exactly.
- **Surrogate error vs Monte Carlo error:** account for surrogate error when using it for UQ or inference conclusions.

## See also

- [Gaussian processes](/ref/approx/gaussian-processes)
- [Global sensitivity analysis](/ref/dynamics/global-sensitivity)
- [High-dimensional numerics](/ref/approx/high-dimensional-numerics)
- [Global & derivative-free optimisation](/ref/optimise/global-derivative-free)
- [Experimental design](/ref/freq/experimental-design)

**References:**

- Forrester, Sóbester & Keane, _Engineering Design via Surrogate Modelling_.
- Sudret, "Global sensitivity analysis using polynomial chaos expansions", Reliab. Eng. Syst. Saf. (2008).
- Gramacy, _Surrogates: Gaussian Process Modeling, Design, and Optimization for the Applied Sciences_.
