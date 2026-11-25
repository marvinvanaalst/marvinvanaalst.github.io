---
title: 'Kalman filters & smoothers'
description: 'Linear Kalman filter, EKF, UKF and ensemble Kalman filter for state estimation, RTS smoothing, and parameter estimation through the filter likelihood.'
aliases:
  [
    Kalman filter,
    KF,
    extended Kalman filter,
    EKF,
    unscented Kalman filter,
    UKF,
    sigma points,
    ensemble Kalman filter,
    EnKF,
    Rauch-Tung-Striebel smoother,
    RTS smoother,
    state-space model,
    data assimilation,
    innovation,
    prediction error decomposition,
    square-root filter,
    information filter,
    moving horizon estimation,
    sensor fusion
  ]
tags: [state estimation, filtering, time series, Bayesian]
order: 1
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- Estimate hidden states $`x_t`$ from noisy measurements $`y_t`$ in a state-space model, recursively as data arrive.
- *Linear-Gaussian:* the *Kalman filter* is exact and optimal; $`O(n^3)`$ per step for $`n`$ states. *RTS smoother* gives estimates using all data.
- *Mildly nonlinear:* *EKF* (linearise with Jacobians) or *UKF* (propagate sigma points; no Jacobians, usually more accurate).
- *High-dimensional* (weather, oceans, large PDE models): *ensemble Kalman filter* with localisation and inflation.
- *Strongly nonlinear / multimodal / non-Gaussian:* particle filters (see [particle filters](/ref/signals/particle-filters)).
- The filter also gives the exact (linear) or approximate *likelihood* via the innovations, for estimating parameters.

## Decision table

| Filter | Model | Cost per step | Accuracy | Use when |
| --- | --- | --- | --- | --- |
| Kalman filter | linear, Gaussian | $`O(n^3 + m^3)`$ | exact | linear systems, GP regression in 1D |
| Square-root / UD KF | linear, Gaussian | similar | exact, numerically robust | ill-conditioned covariances, long runs |
| Information filter | linear, Gaussian | cheap updates with many sensors | exact | decentralised sensor fusion |
| EKF | nonlinear, Gaussian approx. | + Jacobians | first-order | smooth, mildly nonlinear; Jacobians available |
| Iterated EKF | nonlinear | repeated linearisation in update | better for nonlinear measurements |  |
| UKF | nonlinear | $`2n + 1`$ model evaluations | \~second-order | *default nonlinear Gaussian filter* |
| EnKF | nonlinear, high-dimensional | $`N_e`$ model runs (ensemble) | Monte Carlo, low-rank covariance | $`n`$ huge ($`10^6`$+), geophysics |
| Particle filter | arbitrary | $`N_p`$ model runs | exact as $`N_p \to \infty`$ | multimodal, strongly non-Gaussian, small $`n`$ |
| Moving horizon estimation | nonlinear, constraints | optimisation over a window | handles constraints | constrained states, MPC pairing |

## Variants

### Kalman filter

Model: $`x_t = Fx_{t-1} + w_t`$, $`w_t \sim\mathcal N(0, Q)`$; $`y_t = Hx_t + v_t`$, $`v_t \sim\mathcal N(0, R)`$.

1. *Predict:* $`\hat x^-_t = F\hat x_{t-1}`$, $`P^-_t = FP_{t-1}F^\top + Q`$.
2. *Innovation:* $`e_t = y_t - H\hat x^-_t`$, $`S_t = HP^-_tH^\top + R`$.
3. *Gain:* $`K_t = P^-_tH^\top S_t^{-1}`$.
4. *Update:* $`\hat x_t = \hat x^-_t + K_te_t`$, $`P_t = (I - K_tH)P^-_t`$ (prefer the Joseph form for numerical symmetry).

- *Log-likelihood* (prediction error decomposition): $`\sum_t -\tfrac12\big(\log\det(2\pi S_t) + e_t^\top S_t^{-1}e_t\big)`$. Maximise over $`Q`$, $`R`$ and model parameters.
- Continuous-time models: discretise $`F = e^{A\Delta t}`$, $`Q`$ from the Van Loan matrix exponential (see [matrix functions](/ref/solve/matrix-functions)).

### RTS smoother

- Backward pass after filtering: $`G_t = P_tF^\top(P^-_{t+1})^{-1}`$, $`\hat x^s_t = \hat x_t + G_t(\hat x^s_{t+1} - \hat x^-_{t+1})`$, $`P^s_t = P_t + G_t(P^s_{t+1} - P^-_{t+1})G_t^\top`$.

### EKF

- Propagate the mean through the nonlinear model $`x^-_t = f(\hat x_{t-1})`$; covariances with Jacobians $`F_t = \partial f/\partial x`$, $`H_t = \partial h/\partial x`$ (AD).
- Fails for strong nonlinearity (linearisation error) and can diverge; iterated EKF improves the update.

### UKF

1. Choose $`2n + 1`$ sigma points $`\hat x \pm\sqrt{(n + \lambda)P}`$ columns (Cholesky), with weights.
2. Propagate each through $`f`$ (or $`h`$).
3. Recover mean and covariance (and cross-covariance for the gain) from the transformed points.

- Exact for quadratic nonlinearities in the mean; no derivatives; similar cost to EKF for moderate $`n`$.

### Ensemble Kalman filter

- Represent the distribution by $`N_e`$ ensemble members (≈ 20–100); forecast each through the model; update each with the Kalman gain from the ensemble covariance (stochastic EnKF with perturbed observations, or deterministic square-root variants like ETKF).
- *Localisation* (taper spurious long-range correlations) and *inflation* (counter ensemble collapse) are essential with small ensembles.

## Other lenses

- *Bayesian inference.* The Kalman filter is exact recursive Bayesian updating for Gaussians; particle filters are the Monte Carlo version for anything else (see [sequential Monte Carlo](/ref/bayes/sequential-monte-carlo)).
- *Gaussian processes.* GP regression with Matérn kernels in time is Kalman filtering/smoothing on an equivalent linear SDE, in $`O(n)`$ instead of $`O(n^3)`$ (see [Gaussian processes](/ref/approx/gaussian-processes)).
- *Least squares.* The smoother solves a (block-tridiagonal) weighted least-squares problem over the whole trajectory; recursive least squares is a Kalman filter for constant parameters (see [least squares](/ref/solve/least-squares)).
- *Control.* Kalman filtering is the dual of LQR; together they form LQG control (see [optimal control](/ref/optimise/optimal-control-mpc)).

## Pitfalls & diagnostics

- *Tuning $`Q`$ and $`R`$:* wrong noise levels give over- or under-confident filters; estimate them by maximum likelihood or check innovation statistics.
- *Innovation diagnostics:* normalised innovations $`S_t^{-1/2}e_t`$ should be white with unit variance; NIS/NEES tests.
- *Covariance losing positive definiteness:* use Joseph form or square-root filters.
- *Divergence of EKF* for strongly nonlinear systems or poor initialisation: switch to UKF, iterated EKF or particle filters.
- *Unobservable states* stay at prior uncertainty; check observability before trusting estimates.

## See also

- [Particle filters](/ref/signals/particle-filters)
- [Time-series models](/ref/signals/time-series-models)
- [Gaussian processes](/ref/approx/gaussian-processes)
- [Optimal control & MPC](/ref/optimise/optimal-control-mpc)

*References:*

- Särkkä, _Bayesian Filtering and Smoothing_ (2nd ed.).
- Simon, _Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches_.
- Evensen, _Data Assimilation: The Ensemble Kalman Filter_ (2nd ed.).
