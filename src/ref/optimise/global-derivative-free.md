---
title: 'Global & derivative-free optimisation'
description: 'Nelder–Mead, pattern search, CMA-ES, differential evolution, PSO, simulated annealing, multistart and Bayesian optimisation: which to use for which budget and landscape.'
aliases:
  [
    derivative-free optimization,
    black-box optimization,
    Nelder-Mead,
    simplex method,
    pattern search,
    COBYLA,
    BOBYQA,
    NEWUOA,
    CMA-ES,
    evolution strategies,
    differential evolution,
    particle swarm,
    PSO,
    simulated annealing,
    genetic algorithm,
    multistart,
    Bayesian optimization,
    expected improvement,
    DIRECT,
    global optimization
  ]
tags: [optimisation, global, derivative-free, black-box]
order: 5
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Gradients available?** Then **multistart + a local gradient method** usually beats any "global" heuristic per function evaluation.
- **No gradients, smooth, low dimension (≲ 20–50):** model-based trust region (BOBYQA, NEWUOA; COBYLA with constraints). Nelder–Mead is simple but unreliable beyond ~5–10 dimensions.
- **Rugged / multimodal, moderate budget ($10^3$–$10^5$ evaluations), $d \lesssim 100$:** **CMA-ES** (robust default), differential evolution.
- **Very expensive evaluations (≲ a few hundred), $d \lesssim 20$:** **Bayesian optimisation** with a GP surrogate.
- **Guaranteed global optimum:** only with structure: branch and bound with bounds (interval methods, convex relaxations), Lipschitz methods (DIRECT) in low dimensions.

## Decision table

| Method                                | Evaluations needed                | Dimension                  | Noise tolerance | Use when                                                                                       |
| ------------------------------------- | --------------------------------- | -------------------------- | --------------- | ---------------------------------------------------------------------------------------------- |
| Multistart + L-BFGS / LM              | starts × local cost               | any                        | low             | gradients available (incl. AD, sensitivities)                                                  |
| Nelder–Mead                           | $10^2$–$10^4$                     | ≲ 10                       | low–moderate    | quick, low-dimensional, rough start                                                            |
| Pattern / mesh adaptive search (MADS) | $10^3$+                           | ≲ 50                       | moderate        | nonsmooth, provable local convergence                                                          |
| BOBYQA / NEWUOA / COBYLA              | $O(d^2)$ per model, $10^2$–$10^4$ | ≲ 100                      | low             | smooth, no gradients, bounds / constraints                                                     |
| CMA-ES                                | $10^3$–$10^6$                     | ≲ 100 (more with variants) | good            | **default** for rugged black-box problems                                                      |
| Differential evolution                | $10^3$–$10^6$                     | ≲ 100                      | good            | multimodal, bounded, easy to parallelise                                                       |
| Particle swarm                        | $10^3$–$10^6$                     | ≲ 100                      | moderate        | popular, but usually dominated by CMA-ES/DE                                                    |
| Simulated annealing                   | many                              | discrete or continuous     | good            | combinatorial problems, simple to implement                                                    |
| Bayesian optimisation                 | $10$–$500$                        | ≲ 20                       | good (GP noise) | very expensive evaluations                                                                     |
| DIRECT / Lipschitz methods            | grows exponentially with $d$      | ≲ 10                       | low             | deterministic global search, small $d$                                                         |
| Branch & bound / interval             | problem-dependent                 | small–moderate             | none            | certified global optimum (see [interval arithmetic](/ref/reliable/interval-affine-arithmetic)) |

## Variants

### Nelder–Mead

- Maintain a simplex of $d + 1$ points; replace the worst by reflection, expansion, contraction or shrink.
- No convergence guarantees in general (can stall on non-stationary points); restart from the best point to recover.

### Model-based trust region (Powell)

- Build a quadratic interpolation model from $O(d)$–$O(d^2)$ points, minimise it within a trust region, update the model with the new point.
- Very efficient per evaluation on smooth problems; BOBYQA handles bounds, COBYLA nonlinear constraints (linear models).

### CMA-ES

1. Sample $\lambda$ candidates $x_i \sim \mathcal N(m, \sigma^2C)$.
2. Rank by objective; new mean = weighted average of the best $\mu$.
3. Update the covariance $C$ (rank-one from the evolution path, rank-$\mu$ from successful steps) and the step size $\sigma$ (cumulative step-length adaptation).
4. Restart with larger populations (IPOP) when stuck.

- Learns a quadratic-like metric (≈ inverse Hessian) without gradients; invariant to monotone transforms of $f$ and rotations of the search space.

### Differential evolution

- For each population member $x_i$: mutant $v = x_a + F(x_b - x_c)$, crossover with $x_i$ (rate CR), keep the better of trial and $x_i$.
- Typical $F \approx 0.5$–0.8, CR ≈ 0.9; population ~10$d$. Self-adaptive variants (JADE, SHADE) tune these.

### Bayesian optimisation

1. Initial space-filling design (~$5d$–$10d$ points).
2. Fit a GP surrogate (Matérn-5/2, ARD) to observed values.
3. Maximise an acquisition function: expected improvement (EI), upper confidence bound (UCB), or knowledge gradient / entropy search.
4. Evaluate the expensive function there; repeat.

- Batch/parallel variants (q-EI), constraints, multi-fidelity. Acquisition optimisation is itself a (cheap) global optimisation.

### Multistart

- Sample starting points (Sobol/LHS in transformed coordinates), run a local optimiser from each, cluster the results.
- Number of distinct local minima found vs starts tells you how multimodal the landscape is; for ODE fitting this is usually the most effective strategy (see [fitting ODE models](/ref/optimise/fitting-odes)).

## Other lenses

- **Probability.** CMA-ES is natural-gradient ascent on the expected fitness of a Gaussian search distribution; BO is decision-making under a GP posterior; simulated annealing samples from $e^{-f/T}$ as $T \to 0$ (related to MCMC).
- **Surrogates.** Model-based DFO and BO both replace $f$ by a cheap model; they differ in model type (local quadratic vs global GP) and in how they handle exploration (see [surrogates](/ref/approx/surrogates)).
- **No free lunch.** Averaged over all problems no method wins; heuristics win on problems with structure they exploit (smoothness, separability, low effective dimension).

## Pitfalls & diagnostics

- **Benchmarks with tiny budgets** mislead; compare methods by best value vs number of evaluations (convergence curves, data profiles), with several seeds.
- **Bounds and scaling:** normalise variables to $[0, 1]$ or log scales; CMA-ES's initial $\sigma$ should cover ~1/3 of the range.
- **Noise:** re-evaluate promising points, use noise-aware variants (UH-CMA-ES, noisy EI); don't trust a single lucky evaluation.
- **Expensive evaluations wasted** by population methods on smooth problems; if gradients are obtainable (AD, adjoints), get them.
- **Local polish:** finish global searches with a local method for accuracy.

## See also

- [Unconstrained local optimisation](/ref/optimise/unconstrained-local)
- [Gaussian processes](/ref/approx/gaussian-processes)
- [Surrogate models](/ref/approx/surrogates)
- [Fitting ODE models](/ref/optimise/fitting-odes)
- [Mixed-integer & SAT/SMT](/ref/optimise/mixed-integer-sat-smt)

**References:**

- Conn, Scheinberg & Vicente, _Introduction to Derivative-Free Optimization_.
- Hansen, "The CMA evolution strategy: a tutorial" (2016).
- Garnett, _Bayesian Optimization_.
