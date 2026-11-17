---
title: 'Sensitivity analysis: local & global'
description: 'Local derivatives and control coefficients vs global variance-based methods: Morris screening, Sobol indices with Saltelli estimators, FAST and metabolic control analysis.'
aliases:
  [
    sensitivity analysis,
    local sensitivity,
    global sensitivity,
    Morris method,
    elementary effects,
    Sobol indices,
    total-order index,
    first-order index,
    Saltelli sampling,
    FAST,
    eFAST,
    variance decomposition,
    ANOVA decomposition,
    metabolic control analysis,
    MCA,
    control coefficients,
    elasticities,
    summation theorem,
    PRCC
  ]
tags: [sensitivity, uncertainty quantification, dynamical systems]
order: 4
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Local:** derivatives at one parameter point, $\partial y/\partial p$ or normalised $\partial\log y/\partial\log p$. Cheap (AD or sensitivity equations), but blind to nonlinearity and interactions.
- **Screening (Morris):** ~$10$–$50\times(k + 1)$ model runs for $k$ parameters; ranks parameters, finds the unimportant ones.
- **Variance-based (Sobol):** decomposes output variance into contributions of parameters and interactions. First-order $S_i$ and total-order $S_{T_i}$ indices with Saltelli/Jansen estimators cost $N(k + 2)$ runs, $N \sim 10^3$–$10^4$.
- **Metabolic control analysis:** local, normalised sensitivities of steady-state fluxes and concentrations to enzyme activities, with structural summation and connectivity theorems.
- Expensive models: build a surrogate (GP, polynomial chaos) and compute Sobol indices from it analytically.

## Decision table

| Method                          | Type             | Cost (runs)                      | Captures interactions    | Use when                               |
| ------------------------------- | ---------------- | -------------------------------- | ------------------------ | -------------------------------------- |
| Local derivatives               | local            | 1 forward or adjoint pass        | no                       | gradients, identifiability, MCA        |
| One-at-a-time (OAT)             | local-ish        | $k + 1$ (or a few per parameter) | no                       | sanity checks only                     |
| Morris elementary effects       | screening        | $r(k + 1)$, $r \sim 10$–$50$     | qualitatively ($\sigma$) | many parameters, find unimportant ones |
| Sobol (Saltelli / Jansen)       | global, variance | $N(k + 2)$                       | yes (via $S_T - S$)      | quantitative global importance         |
| FAST / eFAST                    | global, variance | ~$N k$ (first-order cheap)       | eFAST: total order       | first-order indices with few runs      |
| Regression / PRCC               | global, monotone | $N$                              | no                       | monotone relationships, quick overview |
| Derivative-based GSM (DGSM)     | global           | $N$ gradients                    | bounds on $S_T$          | gradients available, screening         |
| Polynomial chaos / GP surrogate | global           | training set                     | yes (analytic)           | expensive models                       |

## Variants

### Local sensitivities

- Absolute $\partial y/\partial p_j$; **normalised (relative)** $\dfrac{\partial\log y}{\partial\log p_j} = \dfrac{p_j}{y}\dfrac{\partial y}{\partial p_j}$, unitless and comparable across parameters.
- Computed with forward sensitivities or adjoints for ODEs (see [ODE sensitivities](/ref/diff/ode-sensitivities)); at steady state via the implicit function theorem, $\partial x^*/\partial p = -J^{-1}\partial f/\partial p$.

### Metabolic control analysis

- **Elasticities:** local, $\varepsilon^{v}_{x} = \partial\log v/\partial\log x$ (property of an isolated reaction).
- **Control coefficients:** systemic, $C^J_{e_i} = \partial\log J/\partial\log e_i$ for flux $J$ and enzyme level $e_i$; similarly for concentrations.
- **Theorems:**
  - Summation: $\sum_i C^J_{e_i} = 1$, $\sum_i C^x_{e_i} = 0$.
  - Connectivity: $\sum_i C^J_{e_i}\varepsilon^{v_i}_x = 0$.
- Computed from the stoichiometric matrix and elasticities by linear algebra; "rate-limiting step" becomes a distribution of control.

### Morris screening

1. Discretise each parameter range into $L$ levels.
2. Build $r$ random trajectories, each changing one parameter at a time by $\Delta$.
3. Elementary effect $EE_j = [y(p + \Delta e_j) - y(p)]/\Delta$ for each step.
4. Summaries: $\mu^*_j$ = mean of $\lvert EE_j\rvert$ (importance), $\sigma_j$ = standard deviation (nonlinearity or interactions).

### Sobol indices

- ANOVA decomposition $y = f_0 + \sum_i f_i(p_i) + \sum_{i \lt j} f_{ij}(p_i, p_j) + \dots$ with orthogonal terms, so $\operatorname{Var}(y) = \sum_i V_i + \sum_{i \lt j} V_{ij} + \dots$
- **First-order** $S_i = V_i/\operatorname{Var}(y)$: effect of $p_i$ alone. **Total-order** $S_{T_i}$: all terms involving $p_i$.
- **Estimation (Saltelli design):**
  1. Two independent sample matrices $A$, $B$ ($N\times k$), ideally from a Sobol sequence.
  2. Matrices $A_B^{(i)}$: $A$ with column $i$ taken from $B$.
  3. Evaluate the model on $A$, $B$ and all $A_B^{(i)}$: $N(k + 2)$ runs.
  4. $V_i \approx \tfrac1N\sum f(B)\,[f(A_B^{(i)}) - f(A)]$ (Saltelli 2010), $V_{T_i} \approx \tfrac1{2N}\sum[f(A) - f(A_B^{(i)})]^2$ (Jansen).
- Bootstrap confidence intervals; negative small estimates = Monte Carlo noise.

### Surrogate-based

- **Polynomial chaos:** expand $y$ in orthogonal polynomials of the inputs; Sobol indices are sums of squared coefficients (no extra runs).
- **Gaussian processes:** Sobol indices of the posterior mean, with uncertainty from the GP (see [surrogates](/ref/approx/surrogates)).

## Other lenses

- **Statistics.** Sobol indices are a functional ANOVA; total-order indices measure how much variance remains if all other parameters were known.
- **Identifiability.** Low sensitivity means a parameter cannot be estimated; but high sensitivities can still be non-identifiable if correlated. The FIM (Gram matrix of local sensitivities) captures both (see [identifiability](/ref/freq/identifiability)).
- **Explainable AI.** Shapley-value attribution (SHAP) and Sobol/Shapley effects are close relatives: both apportion output variation to inputs, Shapley effects being Sobol indices shared fairly among interacting inputs (see [feature attribution](/ref/explain/feature-attribution)).

## Pitfalls & diagnostics

- **Results depend on input distributions and ranges;** report them. Uniform on a log scale is often more sensible for rate constants.
- **Correlated inputs** break the ANOVA decomposition; use Shapley effects or methods for dependent inputs.
- **Scalar outputs only:** for time courses, compute indices per time point or for summary features.
- **Convergence:** increase $N$ until indices stabilise; Saltelli needs thousands of samples for small indices.
- **OAT around a nominal point** is not a global analysis, whatever the plots look like.

## See also

- [ODE sensitivities](/ref/diff/ode-sensitivities)
- [Identifiability](/ref/freq/identifiability)
- [Quasi-Monte Carlo](/ref/random/quasi-monte-carlo)
- [Surrogate models](/ref/approx/surrogates)
- [Feature attribution](/ref/explain/feature-attribution)

**References:**

- Saltelli et al., _Global Sensitivity Analysis: The Primer_.
- Saltelli et al., "Variance based sensitivity analysis of model output", Comput. Phys. Commun. (2010).
- Fell, _Understanding the Control of Metabolism_.
