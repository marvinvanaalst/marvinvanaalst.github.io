---
title: 'Fitting ODE models'
description: 'Single vs multiple shooting vs collocation, log-parameters and scaling, objective functions and noise models, local minima, and a robust practical workflow.'
aliases:
  [
    parameter estimation,
    ODE fitting,
    model calibration,
    single shooting,
    multiple shooting,
    collocation,
    simultaneous approach,
    log parameters,
    multistart,
    objective function,
    weighted least squares,
    error model,
    initial conditions,
    observation function,
    data2dynamics,
    PEtab,
    gradient-based fitting
  ]
tags: [parameter estimation, ODE, optimisation, systems biology]
order: 7
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Objective:* negative log-likelihood of the data given the model (weighted least squares for Gaussian noise; include noise parameters if unknown).
- *Parametrise in log space* for positive parameters (rate constants, concentrations): better conditioning, natural multiplicative uncertainty, positivity for free.
- *Single shooting + multistart + gradient-based local optimiser* (LM/TRF or L-BFGS with sensitivities) is the robust workhorse for systems-biology-sized models.
- *Multiple shooting* or *collocation* when single shooting gets stuck: unstable/chaotic dynamics, long horizons, oscillators, or bad initial guesses.
- Gradients from forward sensitivities (few parameters) or adjoints (many); never from finite differences of a loosely solved ODE.
- After fitting: check identifiability (profile likelihood) and residuals before interpreting parameter values.

## Decision table

| Approach | Unknowns | Robustness to bad starts | Cost per iteration | Use when |
| --- | --- | --- | --- | --- |
| Single shooting | parameters (+ initial states) | low–moderate | one ODE solve + sensitivities | default, stable dynamics, moderate horizons |
| Multiple shooting | parameters + states at segment starts | good | solves per segment (parallel) | oscillators, chaos, long horizons, stiff transients |
| Collocation (simultaneous) | parameters + all states on a mesh | very good | one large sparse NLP | difficult dynamics, path constraints (IPOPT) |
| Two-stage / gradient matching | parameters only | good start, biased | cheap (no ODE solves) | initial guesses from smoothed data |
| Bayesian (MCMC) | full posterior | depends on sampler | many solves | uncertainty quantification (see [MCMC](/ref/bayes/mcmc)) |
| Global search (CMA-ES, DE) | parameters | good | many solves | no gradients, rugged landscapes |

## Variants

### Objective and noise model

- Gaussian noise: $`-\log L = \tfrac12\sum_i\big[(y_i - h(x(t_i), \theta))^2/\sigma_i^2 + \log(2\pi\sigma_i^2)\big]`$. Keep the $`\log\sigma`$ term if $`\sigma`$ is estimated.
- *Multiplicative noise:* fit in log space of the observables (log-normal noise).
- *Observation function* $`h`$: scaling factors and offsets (relative data, e.g. Western blots) can be optimised analytically (hierarchical optimisation) instead of jointly.
- *Initial conditions:* fixed, estimated, or from a pre-equilibration steady state (simulate to steady state under the pre-stimulus condition).

### Single shooting

1. For given $`\theta`$, integrate the ODE from $`t_0`$ and evaluate residuals at data times.
2. Get the residual Jacobian from forward sensitivities (or the gradient via adjoints).
3. Step with LM/TRF (least-squares structure) or L-BFGS/IPOPT; repeat.

- Tight integration tolerances (rtol ≈ $`10^{-8}`$) for consistent gradients; loose tolerances make the objective noisy.

### Multiple shooting

- Split $`[t_0, T]`$ into segments; unknowns are $`\theta`$ and the state $`s_j`$ at each segment start.
- Continuity constraints $`x(t_{j+1}; s_j, \theta) = s_{j+1}`$, enforced as equality constraints (SQP / constrained Gauss–Newton) or as penalties that tighten.
- Segment states can be initialised from data, so the optimiser never has to integrate a long, divergent trajectory with a bad $`\theta`$. Converges from much worse starts; parallel over segments.

### Collocation

- Discretise states on a mesh (polynomials per interval), enforce the ODE at collocation points as constraints: one large sparse NLP in $`(\theta, x_{\text{mesh}})`$, solved by interior point.
- Very robust, handles path constraints; the NLP is big and mesh choice matters.

### Practical workflow

1. *Nondimensionalise / scale* states and time; choose log parameters with sensible bounds (e.g. $`10^{-5}`$–$`10^{5}`$).
2. *Check simulation* at nominal parameters; set tolerances; make sure the solver doesn't fail in the parameter region.
3. *Multistart* (≥ 50–100 starts, Latin hypercube in log space), local optimiser with exact gradients.
4. *Waterfall plot:* sorted final objective values; a plateau reached by many starts suggests the global optimum.
5. *Identifiability:* profile likelihoods for each parameter (see [identifiability](/ref/freq/identifiability)).
6. *Validation:* predict held-out conditions; inspect residuals.

## Other lenses

- *Optimal control.* Parameter estimation is an optimal control problem with constant controls; shooting and collocation come from there (see [optimal control & MPC](/ref/optimise/optimal-control-mpc)).
- *Statistics.* The fitted objective plus its curvature gives Fisher information and confidence intervals; the same likelihood plugs into Bayesian inference (see [estimation & MLE](/ref/freq/estimation-mle)).
- *Machine learning.* Training neural ODEs is the same problem with many more parameters; adjoints and minibatching over trajectories/time windows carry over (see [neural ODEs & UDEs](/ref/optimise/neural-odes-udes)).

## Pitfalls & diagnostics

- *Integration failures* inside the optimiser (stiff regions, blow-ups at extreme parameters): return a large objective / fail gracefully, use stiff solvers, restrict bounds.
- *Local minima* are the rule; a single fit from one start is not a result.
- *Non-identifiability* shows up as flat profiles, huge confidence intervals, parameters running to bounds; fix or combine parameters, or design better experiments (see [experimental design](/ref/freq/experimental-design)).
- *Overfitting with many parameters:* compare models with AIC/BIC or cross-validation (see [model selection](/ref/freq/model-selection)).
- *Noise model misspecification* biases estimates and confidence intervals; check normalised residuals.

## See also

- [Nonlinear least squares](/ref/optimise/nonlinear-least-squares)
- [ODE sensitivities](/ref/diff/ode-sensitivities)
- [Identifiability](/ref/freq/identifiability)
- [Global & derivative-free optimisation](/ref/optimise/global-derivative-free)
- [Optimal control & MPC](/ref/optimise/optimal-control-mpc)

*References:*

- Raue et al., "Lessons learned from quantitative dynamical modeling in systems biology", PLoS ONE (2013).
- Bock, "Recent advances in parameter identification techniques for ODE" (1983) — multiple shooting.
- Villaverde et al., "Benchmarking optimization methods for parameter estimation in large kinetic models", Bioinformatics (2019).
