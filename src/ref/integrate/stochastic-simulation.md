---
title: 'Stochastic simulation of reaction networks'
description: 'Gillespie SSA, next-reaction method, τ-leaping, the chemical Langevin equation and hybrid methods: exact vs approximate discrete stochastic kinetics.'
aliases:
  [
    Gillespie algorithm,
    SSA,
    direct method,
    first reaction method,
    next reaction method,
    Gibson-Bruck,
    tau-leaping,
    chemical Langevin equation,
    CLE,
    chemical master equation,
    propensity,
    hybrid methods,
    slow-scale SSA,
    delay SSA,
    jump processes,
    Poisson process,
    DDE
  ]
tags: [stochastic, chemical kinetics, simulation]
order: 8
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- When molecule counts are small (gene expression, signalling), deterministic ODEs miss noise-driven behaviour. Simulate the **chemical master equation** (CME) with exact sample paths.
- **Gillespie's direct method:** exact, $O(M)$ per reaction event for $M$ reactions. Fine for small networks.
- **Next reaction method / composition-rejection:** exact, $O(\log M)$ or $O(1)$ per event. For large networks.
- **τ-leaping:** fire many reactions per step with Poisson numbers; approximate but much faster when counts are moderate.
- **Chemical Langevin equation:** SDE approximation for large counts; the ODE (reaction rate equation) is its deterministic limit.
- **Stiff or multiscale:** slow-scale SSA, hybrid (deterministic fast species + stochastic slow ones), or implicit τ-leaping.

## Decision table

| Method                       | Exact               | Cost per event / step                  | Regime                                      | Pitfalls                             |
| ---------------------------- | ------------------- | -------------------------------------- | ------------------------------------------- | ------------------------------------ |
| Direct method (Gillespie)    | yes                 | $O(M)$ per event                       | small networks, low counts                  | slow when events are frequent        |
| Optimised direct / sorting   | yes                 | ~$O(M)$, better constants              | moderate networks                           |                                      |
| Next reaction (Gibson–Bruck) | yes                 | $O(\log M)$ (priority queue)           | large sparse networks                       | dependency graph bookkeeping         |
| Composition-rejection        | yes                 | $O(1)$ amortised                       | very large networks, wide propensity range  |                                      |
| Explicit τ-leaping           | no                  | one Poisson draw per reaction per leap | moderate counts                             | negative populations if τ too large  |
| Implicit τ-leaping           | no                  | nonlinear solve per leap               | stiff networks                              | damps fluctuations of fast reactions |
| Chemical Langevin (CLE)      | no                  | SDE step                               | large counts ($\gtrsim 100$ per species)    | negative values, poor at low counts  |
| Hybrid / slow-scale SSA      | no                  | depends                                | multiscale (fast/slow reactions or species) | partition choice                     |
| Finite state projection      | yes (bounded error) | solve CME directly $e^{tA}p_0$         | few species, small state space              | state space explodes combinatorially |

## Variants

### The model

- State $x \in \mathbb N^S$ (molecule counts); reaction $j$ has propensity $a_j(x)$ and state change $\nu_j$.
- Mass action propensities: $a = c\,x_A$ (unimolecular), $c\,x_A x_B$, $c\,x_A(x_A - 1)/2$ (dimerisation). Note $c$ relates to the deterministic rate constant via the volume: $c = k/(N_A V)$ for bimolecular reactions.
- **CME:** $\dfrac{dP(x, t)}{dt} = \sum_j\big[a_j(x - \nu_j)P(x - \nu_j, t) - a_j(x)P(x, t)\big]$.

### Gillespie's direct method

1. Compute all propensities and $a_0 = \sum_j a_j(x)$.
2. Time to next event: $\tau = -\ln(r_1)/a_0$ (exponential with rate $a_0$).
3. Which reaction: smallest $j$ with $\sum_{k \le j} a_k \gt r_2 a_0$.
4. Update $t \leftarrow t + \tau$, $x \leftarrow x + \nu_j$; recompute affected propensities; repeat.

### Next reaction method

- Each reaction keeps an absolute putative firing time in an indexed priority queue.
- Fire the earliest, update only reactions whose propensities depend on the changed species (dependency graph), rescaling their times instead of drawing new random numbers.
- Equivalent to simulating independent unit-rate Poisson processes with time changes (random time change representation), which also enables coupling for variance reduction and sensitivities.

### τ-leaping

- Choose $\tau$ so that propensities change little during the leap (Cao–Gillespie–Petzold step selection), then fire $K_j \sim \operatorname{Poisson}(a_j(x)\tau)$ reactions: $x \leftarrow x + \sum_j K_j\nu_j$.
- Fall back to exact SSA for a few steps when populations of critical species are low; binomial leaping prevents negative counts.

### Chemical Langevin equation

- $dX = \sum_j\nu_j a_j(X)\,dt + \sum_j\nu_j\sqrt{a_j(X)}\,dW_j$ (Itô).
- Valid when every reaction fires many times per relevant time interval. Its deterministic limit (drop noise, volume → ∞) is the reaction rate equation $x' = \sum_j\nu_j a_j(x)$.
- Linear noise approximation (van Kampen): Gaussian fluctuations around the ODE solution, covariance from a Lyapunov equation. Cheap noise estimates without simulation.

### Delays and other jump processes

- **Delay SSA:** reactions whose products appear after a delay (transcription, translation); keep a queue of scheduled completions.
- **DDEs** (deterministic delays) need history interpolation and discontinuity tracking: the solution is non-smooth at $t_0 + k\tau$; method of steps or solvers with dense-output history.
- The same SSA machinery simulates epidemics (SIR), population dynamics and queues.

## Other lenses

- **Markov chains.** The CME is the forward equation of a continuous-time Markov chain; SSA samples its paths exactly. Stationary distributions, first-passage times and mixing follow from CTMC theory (see [stochastic processes](/ref/dynamics/stochastic-processes)).
- **Linear algebra.** For small state spaces, solve $P' = AP$ directly: finite state projection + Krylov matrix exponential (see [matrix functions](/ref/solve/matrix-functions)), or compute the stationary distribution as a null vector.
- **Statistics.** Parameter inference from stochastic simulations is likelihood-free in general: ABC, synthetic likelihood, particle MCMC (see [likelihood-free inference](/ref/bayes/likelihood-free)).

## Pitfalls & diagnostics

- **Rate constant conversion** between deterministic (concentration) and stochastic (count) units is the most common error; check the volume factor and the factor 2 for dimerisation.
- **Statistics, not single paths:** report distributions over many runs; one trajectory is anecdotal.
- **Stiffness:** fast reversible reactions dominate event counts while slow reactions drive the interesting dynamics; use slow-scale SSA or hybrid methods.
- **Random number streams:** parallel ensembles need independent streams (see [PRNGs](/ref/random/prngs)).
- **Validate approximations** (τ-leaping, CLE) against exact SSA on a reduced case: compare means, variances and, ideally, distributions.

## See also

- [Stochastic processes](/ref/dynamics/stochastic-processes)
- [SDEs](/ref/integrate/sdes)
- [Reaction network theory](/ref/dynamics/reaction-network-theory)
- [Likelihood-free inference](/ref/bayes/likelihood-free)

**References:**

- Gillespie, "Exact stochastic simulation of coupled chemical reactions", J. Phys. Chem. (1977).
- Gillespie, "Stochastic simulation of chemical kinetics", Annu. Rev. Phys. Chem. (2007).
- Anderson & Kurtz, _Stochastic Analysis of Biochemical Systems_.
