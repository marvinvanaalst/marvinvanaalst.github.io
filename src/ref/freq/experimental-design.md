---
title: 'Optimal experimental design'
description: 'Fisher-information criteria (D-, A-, E-optimality), Bayesian design by expected information gain, sequential and adaptive designs, and model discrimination.'
aliases:
  [
    experimental design,
    optimal design,
    design of experiments,
    DoE,
    D-optimal,
    A-optimal,
    E-optimal,
    Fisher information,
    local design,
    robust design,
    Bayesian experimental design,
    expected information gain,
    EIG,
    mutual information,
    sequential design,
    adaptive design,
    model discrimination,
    sampling times,
    factorial design,
    response surface design
  ]
tags: [statistics, experimental design, Fisher information]
order: 9
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- Choose experiments (sampling times, inputs, doses, measured species) that make parameters or predictions as precise as possible, **before** spending lab time.
- **Local (FIM-based) design:** maximise a scalar function of the Fisher information at a nominal parameter guess. **D-optimal** ($\log\det$, smallest confidence ellipsoid) is the default; **A** (average variance), **E** (worst direction).
- **Robust to parameter uncertainty:** average the criterion over a prior (pseudo-Bayesian) or optimise the worst case.
- **Bayesian design:** maximise **expected information gain** (mutual information between parameters and data); handles nonlinearity and multimodality, but costs nested Monte Carlo or variational bounds.
- **Sequential design** (fit → design → measure → refit) usually beats one-shot design for nonlinear models.
- **Classical DoE** (factorial, fractional factorial, response surface) for screening many factors with simple models.

## Decision table

| Criterion / method                   | Optimises                                               | Cost                           | Use when                                       |
| ------------------------------------ | ------------------------------------------------------- | ------------------------------ | ---------------------------------------------- |
| D-optimal                            | $\log\det\mathrm{FIM}$ (volume of confidence region)    | sensitivities per candidate    | **default** parameter estimation               |
| A-optimal                            | $-\operatorname{tr}(\mathrm{FIM}^{-1})$ (mean variance) | same                           | all parameters equally important               |
| E-optimal                            | $\lambda_{\min}(\mathrm{FIM})$                          | same                           | fix the worst-determined direction             |
| D$_s$ / targeted                     | subset of parameters or predictions                     | same                           | nuisance parameters, prediction-focused design |
| V / I-optimal                        | average prediction variance over a region               | same                           | prediction rather than parameters              |
| Pseudo-Bayesian (robust)             | $\mathbb E_{\text{prior}}[\log\det\mathrm{FIM}]$        | FIMs at prior samples          | uncertain nominal parameters                   |
| Bayesian EIG                         | $I(\theta; y\mid d)$                                    | nested MC / variational bounds | nonlinear, multimodal, expensive data          |
| Model discrimination (T-optimal, KL) | difference between model predictions                    | simulations per model          | choosing between mechanisms                    |
| Factorial / fractional factorial     | main effects and interactions                           | none (tables)                  | screening factors, linear-ish responses        |
| Space-filling (LHS, Sobol)           | coverage                                                | none                           | computer experiments, surrogate training       |

## Variants

### Fisher-information design

1. Nominal parameters $\theta_0$ (from literature or a pilot fit), candidate experiments $d$ (time points, inputs).
2. Sensitivities $S(d) = \partial y(d)/\partial\theta$ (w.r.t. log-parameters), noise covariance $\Sigma$.
3. $\mathrm{FIM}(d) = S^\top\Sigma^{-1}S$; for several experiments, information adds.
4. Optimise the criterion over $d$: continuous designs (weights on candidate points, convex for D-optimality) or exact designs (integer numbers of samples; exchange algorithms, greedy selection).

- Invariance: D-optimality is invariant to reparametrisation (scaling), A and E are not; use log-parameters for A/E.
- Optimal designs often put samples at few distinct points (number ≈ number of parameters); spread some extra points to check model fit.

### Robust and Bayesian design

- **Pseudo-Bayesian:** average $\log\det\mathrm{FIM}(d, \theta)$ over prior samples of $\theta$.
- **Expected information gain:** $\mathrm{EIG}(d) = \mathbb E_{p(\theta)p(y\mid\theta, d)}[\log p(y\mid\theta, d) - \log p(y\mid d)]$.
  - Nested Monte Carlo estimator (biased, expensive); variational bounds (Barber–Agakov, InfoNCE) with neural networks; Laplace approximations.
  - For linear-Gaussian models EIG reduces to $\tfrac12\log\det(I + \Sigma_{\text{prior}}\mathrm{FIM})$, i.e. Bayesian D-optimality.

### Sequential / adaptive design

1. Fit the model to current data.
2. Design the next experiment with the updated estimates or posterior.
3. Measure, append data, repeat.

- Avoids the circularity of needing the parameters to design the experiment that estimates them.

### Model discrimination

- Choose $d$ maximising divergence between competing models' predictive distributions (T-optimality, Box–Hill, expected KL), so the next data best separates mechanisms.

## Other lenses

- **Information theory.** EIG is mutual information between parameters and data; D-optimality is its Gaussian approximation (see [entropy & KL](/ref/info/entropy-kl-mi)).
- **Identifiability.** Design is the constructive answer to practical non-identifiability: find measurements whose sensitivities break parameter correlations (see [identifiability](/ref/freq/identifiability)).
- **Active learning / Bayesian optimisation.** Choosing the next simulation or experiment by an acquisition function is sequential design with a surrogate (see [surrogates](/ref/approx/surrogates)).

## Pitfalls & diagnostics

- **Local designs depend on the nominal guess;** check robustness by evaluating the design at other plausible parameter values.
- **Model misspecification:** optimal designs leave no points to detect lack of fit; add replicates and space-filling points.
- **Correct noise model:** FIM depends on $\Sigma$; proportional noise changes optimal sampling times dramatically.
- **Practical constraints** (minimum spacing, number of samples per animal, costs) belong in the optimisation, not afterwards.
- **Validate** the design by simulation: generate synthetic data from the design, refit, and look at the resulting parameter uncertainty.

## See also

- [Identifiability](/ref/freq/identifiability)
- [Estimation & MLE](/ref/freq/estimation-mle)
- [ODE sensitivities](/ref/diff/ode-sensitivities)
- [Fisher information & geometry](/ref/info/fisher-information-geometry)
- [Surrogate models](/ref/approx/surrogates)

**References:**

- Atkinson, Donev & Tobias, _Optimum Experimental Designs, with SAS_.
- Rainforth, Foster, Ivanova & Bickford Smith, "Modern Bayesian experimental design", Statistical Science (2024).
- Kreutz & Timmer, "Systems biology: experimental design", FEBS J. (2009).
