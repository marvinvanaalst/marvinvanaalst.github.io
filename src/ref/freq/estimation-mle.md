---
title: 'Estimation & maximum likelihood'
description: 'Maximum likelihood, Fisher information and standard errors, the Cramér–Rao bound, sandwich estimators under misspecification, M-estimators and the method of moments.'
aliases:
  [
    maximum likelihood,
    MLE,
    likelihood function,
    log-likelihood,
    score function,
    Fisher information,
    observed information,
    Cramer-Rao bound,
    asymptotic normality,
    standard errors,
    sandwich estimator,
    robust standard errors,
    misspecification,
    M-estimator,
    method of moments,
    GMM,
    bias,
    variance,
    consistency,
    efficiency,
    delta method
  ]
tags: [statistics, estimation, likelihood]
order: 1
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **MLE:** $\hat\theta = \arg\max_\theta\ell(\theta)$, $\ell = \sum_i\log p(y_i\mid\theta)$. Consistent, asymptotically normal and efficient under regularity conditions.
- **Standard errors:** inverse of the **observed information** (negative Hessian of $\ell$ at $\hat\theta$), or expected Fisher information. Gaussian errors: $(J^\top WJ)^{-1}$ from the residual Jacobian.
- **Model possibly misspecified** (wrong noise model, dependence)? Use the **sandwich** covariance $H^{-1}SH^{-1}$.
- **Delta method** for functions of parameters: $\operatorname{Var}(g(\hat\theta)) \approx \nabla g^\top\Sigma\nabla g$.
- Asymptotic (Wald) intervals can be badly wrong for nonlinear models and small data; use profile likelihood or bootstrap (see [confidence intervals](/ref/freq/confidence-intervals)).

## Decision table

| Estimator / quantity                | Formula                                                           | Use when                                          |
| ----------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------- |
| MLE                                 | $\arg\max\ell(\theta)$                                            | default with a credible probability model         |
| Least squares                       | MLE for Gaussian errors with known/common variance                | regression, curve fitting                         |
| Method of moments                   | match sample moments to model moments                             | quick initial estimates, closed forms             |
| Generalised method of moments (GMM) | weighted moment conditions                                        | econometrics, partially specified models          |
| M-estimators (Huber, Tukey)         | minimise $\sum\rho(r_i)$ with robust $\rho$                       | outliers, heavy tails                             |
| Observed information                | $-\nabla^2\ell(\hat\theta)$                                       | standard errors after fitting (default)           |
| Expected Fisher information         | $\mathbb E[-\nabla^2\ell] = \mathbb E[\nabla\ell\nabla\ell^\top]$ | design of experiments, closed-form models         |
| Sandwich covariance                 | $H^{-1}\big(\sum_i s_is_i^\top\big)H^{-1}$                        | misspecification, heteroscedasticity              |
| REML                                | likelihood of error contrasts                                     | variance components in mixed models (less biased) |

## Variants

### Maximum likelihood in practice

- Maximise the log-likelihood (sums, not products) with a gradient-based optimiser; reparametrise to unconstrained scales (log for variances and rates).
- **Score:** $s(\theta) = \nabla\ell$; at the MLE $s(\hat\theta) = 0$.
- **Invariance:** the MLE of $g(\theta)$ is $g(\hat\theta)$.
- **Noise parameters:** estimate them jointly (e.g. $\sigma$ in Gaussian models) or profile them out analytically.

### Fisher information and Cramér–Rao

- Fisher information $I(\theta) = \mathbb E[s s^\top] = -\mathbb E[\nabla^2\ell]$ (under regularity).
- **Cramér–Rao:** any unbiased estimator has $\operatorname{Cov}(\hat\theta) \succeq I(\theta)^{-1}$; the MLE attains it asymptotically.
- For Gaussian observations with sensitivities $S = \partial\mu/\partial\theta$: $I = S^\top\Sigma^{-1}S$. This is what experimental design optimises (see [experimental design](/ref/freq/experimental-design)).

### Asymptotics

- $\sqrt n(\hat\theta - \theta_0) \to \mathcal N(0, I_1(\theta_0)^{-1})$ (per-observation information $I_1$).
- Regularity conditions fail for: parameters on the boundary (variance = 0), non-identifiable parameters, support depending on $\theta$, number of parameters growing with $n$.

### Misspecification and the sandwich

- If the model is wrong, the MLE converges to the KL-closest model parameter $\theta^*$, with covariance $H^{-1}JH^{-1}$, $H = -\mathbb E[\nabla^2\ell_i]$, $J = \mathbb E[s_is_i^\top]$.
- Equals $I^{-1}$ when the model is correct ($H = J$). Clustered/heteroscedasticity-robust standard errors are sandwich variants.

### Delta method

- For $\psi = g(\theta)$: $\operatorname{Var}(\hat\psi) \approx \nabla g(\hat\theta)^\top\widehat{\operatorname{Cov}}(\hat\theta)\nabla g(\hat\theta)$.
- Poor for strongly nonlinear $g$ or wide uncertainties; profile likelihood or bootstrap instead.

## Other lenses

- **Information theory.** Maximising likelihood = minimising the KL divergence from the empirical distribution to the model (cross-entropy loss in ML) (see [entropy & KL](/ref/info/entropy-kl-mi)).
- **Geometry.** Fisher information is a Riemannian metric on the parameter space; its eigenvalues reveal stiff and sloppy parameter combinations (see [Fisher information & geometry](/ref/info/fisher-information-geometry)).
- **Bayesian.** With flat priors the posterior mode is the MLE; with lots of data the posterior is approximately $\mathcal N(\hat\theta, I^{-1})$ (Bernstein–von Mises) (see [Bayesian foundations](/ref/bayes/foundations-priors)).

## Pitfalls & diagnostics

- **Hessian not positive definite** at the "optimum": not converged, a saddle point, or non-identifiable parameters.
- **Wald standard errors for parameters near bounds** (rates near 0, variances near 0) are meaningless; use log scales or profile likelihood.
- **Finite-difference Hessians** of noisy objectives (ODE solutions with loose tolerances) are garbage; use AD or sensitivities.
- **Small samples:** the MLE can be biased (e.g. variance with $1/n$); REML or bias corrections help.
- **Check the noise model** with residual diagnostics; standard errors inherit its errors.

## See also

- [Confidence intervals](/ref/freq/confidence-intervals)
- [Identifiability](/ref/freq/identifiability)
- [Nonlinear least squares](/ref/optimise/nonlinear-least-squares)
- [Fisher information & geometry](/ref/info/fisher-information-geometry)
- [Matrix calculus](/ref/diff/matrix-calculus)

**References:**

- Casella & Berger, _Statistical Inference_ (2nd ed.).
- Pawitan, _In All Likelihood: Statistical Modelling and Inference Using Likelihood_.
- White, "Maximum likelihood estimation of misspecified models", Econometrica (1982).
