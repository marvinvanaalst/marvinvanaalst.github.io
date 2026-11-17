---
title: 'Nonlinear least squares'
description: 'Gauss–Newton, Levenberg–Marquardt and trust-region reflective methods for curve fitting and parameter estimation, with residual Jacobians and robust losses.'
aliases:
  [
    NLLS,
    curve fitting,
    Gauss-Newton,
    Levenberg-Marquardt,
    LM,
    trust region reflective,
    TRF,
    dogleg,
    residual Jacobian,
    robust loss,
    Huber,
    Cauchy loss,
    parameter estimation,
    variable projection,
    geodesic acceleration,
    weighted least squares
  ]
tags: [optimisation, least squares, parameter estimation, regression]
order: 2
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- Problem: $\min_\theta\tfrac12\lVert r(\theta)\rVert^2$ with residuals $r_i = (y_i - \text{model}_i(\theta))/\sigma_i$.
- **Gauss–Newton** approximates the Hessian by $J^\top J$ (no second derivatives): quadratic-like convergence when residuals are small at the solution.
- **Levenberg–Marquardt** (damped Gauss–Newton, trust region) is the robust default for unconstrained fits. **Trust-region reflective** (TRF) for bound constraints.
- Exploit structure: solve the inner linear least-squares problem with QR, not normal equations; use **variable projection** when some parameters enter linearly.
- Outliers: robust losses (Huber, soft-L1, Cauchy) via IRLS-style reweighting.

## Decision table

| Method                            | Hessian model                            | Bounds            | Use when                                                      | Pitfalls                                      |
| --------------------------------- | ---------------------------------------- | ----------------- | ------------------------------------------------------------- | --------------------------------------------- |
| Gauss–Newton (+ line search)      | $J^\top J$                               | no                | small residuals, good start                                   | fails for large residuals, rank-deficient $J$ |
| Levenberg–Marquardt               | $J^\top J + \lambda D$                   | no (or projected) | **default** unconstrained fitting                             | slow for large-residual problems              |
| Trust-region reflective (TRF)     | $J^\top J$ in a scaled TR                | yes               | bound-constrained fits                                        |                                               |
| Dogleg                            | $J^\top J$                               | no                | well-behaved problems, cheap steps                            |                                               |
| Full Newton / quasi-Newton (BFGS) | exact or approximated Hessian            | depends           | large residuals at optimum                                    | slower per iteration                          |
| Hybrid (NL2SOL)                   | $J^\top J + S$ (quasi-Newton correction) | yes               | large-residual problems                                       |                                               |
| Variable projection (VarPro)      | eliminates linear parameters             | —                 | separable models (sums of exponentials, linear in amplitudes) | needs implementation care                     |
| Robust loss (Huber, Cauchy)       | reweighted $J^\top WJ$                   | —                 | outliers                                                      | non-convex losses (Cauchy) add local minima   |

## Variants

### Gauss–Newton

- Linearise residuals: $r(\theta + p) \approx r + Jp$, minimise $\lVert r + Jp\rVert^2$ → linear least squares.
- Solve with QR of $J$ (or LSQR for large sparse $J$), not via $J^\top J$, to avoid squaring the condition number.
- Hessian of $\tfrac12\lVert r\rVert^2$ is $J^\top J + \sum_i r_i\nabla^2r_i$; the dropped term is small when residuals are small or nearly linear.

### Levenberg–Marquardt

1. Solve $(J^\top J + \lambda D^\top D)\,p = -J^\top r$ (as the augmented least-squares problem $\big[\begin{smallmatrix} J \\ \sqrt\lambda D\end{smallmatrix}\big]p \approx \big[\begin{smallmatrix} -r \\ 0\end{smallmatrix}\big]$).
2. Accept if the actual reduction is good compared to the predicted one; then decrease $\lambda$ (more Gauss–Newton-like). Otherwise increase $\lambda$ (more gradient-descent-like, shorter step) and retry.
3. $D$ = diagonal scaling from the column norms of $J$ (Marquardt's scaling) makes it invariant to parameter units.

- **Geodesic acceleration** adds a second-order correction along the step, helping in narrow curved valleys.

### Bound constraints

- **TRF** (Branch–Coleman–Li): scale the trust region so steps approach bounds gracefully, reflect off them.
- Alternatively transform: $\theta = e^\phi$ for positivity, logistic for intervals. Changes conditioning (often for the better with rate constants).

### Variable projection

- For models linear in some parameters, $\text{model}(\theta, c) = \Phi(\theta)c$: for fixed $\theta$, $c^*(\theta)$ solves a linear least-squares problem.
- Optimise only over $\theta$ with the projected residual $(I - \Phi\Phi^+)y$; fewer parameters, better conditioned, more robust convergence.

### Robust losses

- Replace $\tfrac12r_i^2$ by $\rho(r_i)$: Huber (quadratic near 0, linear tails), soft-L1, Cauchy $\log(1 + r^2)$.
- Implemented by reweighting residuals and Jacobian rows each iteration (Triggs correction), so any NLLS solver can be reused.

### Statistics at the solution

- Parameter covariance (Gaussian noise, correct $\sigma_i$): $\operatorname{Cov}(\hat\theta) \approx (J^\top J)^{-1}$; with estimated noise, scale by $\lVert r\rVert^2/(m - n)$.
- Linearised intervals can be badly wrong for nonlinear models; use profile likelihood (see [confidence intervals](/ref/freq/confidence-intervals)).

## Other lenses

- **Statistics.** NLLS is maximum likelihood under independent Gaussian errors; $J^\top J$ is the Fisher information. Weights must be $1/\sigma_i^2$ for the statistics to mean anything (see [estimation & MLE](/ref/freq/estimation-mle)).
- **Linear algebra.** Every iteration is a linear least-squares problem; LM's damping is Tikhonov regularisation of that subproblem (see [least squares](/ref/solve/least-squares)).
- **Geometry.** The model manifold (all model predictions as $\theta$ varies) is often long and thin (sloppy); LM steps follow it; geodesic acceleration uses its curvature. Sloppiness = huge spread in eigenvalues of $J^\top J$.

## Pitfalls & diagnostics

- **Rank-deficient or ill-conditioned $J$** (non-identifiable parameter combinations): LM still converges but covariances explode; check singular values of the scaled $J$ (see [identifiability](/ref/freq/identifiability)).
- **Local minima:** fit from multiple starts (Latin hypercube in log space); exponentials, oscillations and ODE models are notorious.
- **Jacobian accuracy:** finite-difference Jacobians of ODE models with loose tolerances stall LM; use sensitivities or AD (see [ODE sensitivities](/ref/diff/ode-sensitivities)).
- **Residual diagnostics:** plot weighted residuals vs time/inputs; systematic patterns mean model misspecification.
- **Weights:** unweighted fits of data spanning orders of magnitude are dominated by the largest values; weight or fit in log space for multiplicative noise.

## See also

- [Least squares](/ref/solve/least-squares)
- [Unconstrained local optimisation](/ref/optimise/unconstrained-local)
- [Fitting ODE models](/ref/optimise/fitting-odes)
- [Identifiability](/ref/freq/identifiability)
- [Confidence intervals](/ref/freq/confidence-intervals)

**References:**

- Nocedal & Wright, _Numerical Optimization_, ch. 10.
- Moré, "The Levenberg–Marquardt algorithm: implementation and theory" (1978).
- Transtrum & Sethna, "Improvements to the Levenberg–Marquardt algorithm for nonlinear least-squares minimization" (2012).
