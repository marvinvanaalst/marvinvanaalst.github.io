---
title: 'Identifiability'
description: 'Structural vs practical identifiability, differential-algebra and Taylor-series tests, FIM rank and eigenvalues, profile likelihood, sloppiness, and what to do about non-identifiable parameters.'
aliases:
  [
    identifiability,
    structural identifiability,
    practical identifiability,
    parameter identifiability,
    observability,
    differential algebra,
    Taylor series approach,
    generating series,
    StructuralIdentifiability,
    DAISY,
    Fisher information rank,
    FIM eigenvalues,
    profile likelihood,
    sloppiness,
    sloppy models,
    parameter correlations,
    reparametrization,
    identifiable combinations
  ]
tags: [statistics, parameter estimation, dynamical systems]
order: 3
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Structural identifiability:** with perfect, noise-free data of the measured outputs, are the parameters uniquely determined? A property of model + observables + inputs, checkable symbolically before any experiment.
- **Practical identifiability:** with the actual noisy, finite data, are the parameters estimated precisely? Checked with **profile likelihood** (best) or the Fisher information.
- Non-identifiable parameters show up as flat profiles, near-zero FIM eigenvalues, huge correlations, parameters drifting to bounds and multistart fits with equal objective but different parameters.
- **Fixes:** reparametrise into identifiable combinations, fix parameters from literature, measure more observables, design more informative experiments, or accept and report prediction uncertainty instead.
- Most biological models are **sloppy**: FIM eigenvalues spread over many orders of magnitude. Predictions can still be well constrained even when parameters aren't.

## Decision table

| Method                                                     | Type                          | Cost                           | Answers                                                 | Use when                                      |
| ---------------------------------------------------------- | ----------------------------- | ------------------------------ | ------------------------------------------------------- | --------------------------------------------- |
| Differential algebra (StructuralIdentifiability.jl, DAISY) | structural                    | symbolic, can be heavy         | global/local identifiability, identifiable combinations | rational ODE models, before experiments       |
| Taylor / generating series                                 | structural                    | symbolic                       | local identifiability                                   | small models                                  |
| Sensitivity rank (random parameter points)                 | structural (local, numerical) | sensitivities                  | locally identifiable or not                             | larger models, quick check                    |
| FIM rank and eigenvalues                                   | practical (local)             | sensitivities at the fit       | sloppy / stiff directions                               | quick local assessment, experimental design   |
| Profile likelihood                                         | practical                     | 1D optimisations per parameter | finite/infinite confidence intervals                    | **default** after fitting                     |
| Multistart parameter clouds                                | practical                     | many fits                      | spread of equally good fits                             | side product of multistart fitting            |
| Bayesian posterior vs prior                                | practical                     | MCMC                           | which parameters learn from data                        | Bayesian workflow                             |
| Correlation matrix from FIM                                | practical                     | free                           | pairwise compensation                                   | first look (misses higher-order dependencies) |

## Variants

### Structural identifiability

- **Global:** a unique parameter vector produces the output. **Local:** unique within a neighbourhood (finitely many solutions). **Non-identifiable:** a continuum.
- **Differential algebra:** eliminate unobserved states to get input-output equations; their coefficients are functions of the parameters; identifiability = injectivity of that map. Tools return identifiable functions of parameters (e.g. only $k_1k_2$ and $k_1 + k_3$).
- **Numerical local check:** compute the sensitivity matrix of outputs w.r.t. parameters (and unknown initial conditions) at random parameter values with many time points; rank deficiency ⇒ structural non-identifiability.
- Unknown initial conditions, scaling factors (relative data) and offsets must be included as parameters.

### Practical identifiability via profiles

- Profile each parameter (see [confidence intervals](/ref/freq/confidence-intervals)):
  - bounded on both sides by the threshold: identifiable
  - flat: structurally non-identifiable
  - rising on one side only: practically non-identifiable (data too limited)
- Trajectories of the other parameters along a profile reveal which parameters compensate (functional relationships).

### Fisher information and sloppiness

- $\mathrm{FIM} = S^\top\Sigma^{-1}S$ with sensitivities $S$ (preferably w.r.t. log-parameters).
- Eigen-decomposition: small eigenvalues ↔ poorly determined parameter combinations (eigenvectors); spread over $10^6$ or more is "sloppy".
- **Local only:** the FIM describes the quadratic approximation at one point; profiles capture nonlinear dependence.

### What to do

1. **Reparametrise:** replace non-identifiable parameters by the identifiable combinations (e.g. $V_{\max}$ and $K_M$ instead of elementary rate constants) or nondimensionalise.
2. **Fix** parameters to literature values (and test sensitivity of conclusions).
3. **New data:** measure additional species, add perturbations/inputs that excite missing dynamics, sample time points where sensitivities differ (see [experimental design](/ref/freq/experimental-design)).
4. **Regularise or use priors** (Bayesian), reporting that the posterior there reflects the prior.
5. **Focus on predictions:** compute prediction profiles or posterior predictive intervals; sloppy models often predict well along stiff directions.

## Other lenses

- **Linear algebra.** Identifiability is rank: structural = generic rank of the sensitivity matrix; practical = numerical rank relative to noise (singular values vs noise level) (see [conditioning](/ref/reliable/conditioning-stability)).
- **Control theory.** Observability (can states be reconstructed?) is the state analogue; treating parameters as constant states turns identifiability into observability.
- **Model reduction.** Timescale separation makes fast parameters appear only in combinations; QSSA explains many identifiability results (see [timescale reduction](/ref/dynamics/timescale-reduction)).

## Pitfalls & diagnostics

- **Interpreting point estimates of non-identifiable parameters** as biological constants is the classic mistake.
- **Correlation matrices** from the FIM miss non-identifiabilities involving more than two parameters; use eigenvectors or profiles.
- **Bounds hide problems:** parameters hitting optimiser bounds are a symptom, not a solution.
- **Structural tools assume exact model and data;** practical identifiability can be worse but never better.
- **Check with synthetic data:** simulate data from the fitted model with realistic noise and test whether refits recover the parameters.

## See also

- [Confidence intervals](/ref/freq/confidence-intervals)
- [Experimental design](/ref/freq/experimental-design)
- [Fitting ODE models](/ref/optimise/fitting-odes)
- [ODE sensitivities](/ref/diff/ode-sensitivities)
- [Fisher information & geometry](/ref/info/fisher-information-geometry)

**References:**

- Raue et al., "Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood", Bioinformatics (2009).
- Gutenkunst et al., "Universally sloppy parameter sensitivities in systems biology models", PLoS Comput. Biol. (2007).
- Dong et al., "Differential elimination for dynamical models via projections with applications to structural identifiability", SIAM J. Appl. Algebra Geom. (2023).
