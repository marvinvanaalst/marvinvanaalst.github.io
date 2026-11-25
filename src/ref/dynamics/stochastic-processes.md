---
title: 'Stochastic processes'
description: 'Discrete- and continuous-time Markov chains, Poisson processes, the chemical master equation, Fokker–Planck, stationary distributions, first passage and mixing times.'
aliases:
  [
    Markov chain,
    DTMC,
    CTMC,
    transition matrix,
    generator matrix,
    Q-matrix,
    stationary distribution,
    detailed balance,
    ergodicity,
    mixing time,
    spectral gap,
    Poisson process,
    birth-death process,
    master equation,
    chemical master equation,
    Fokker-Planck,
    Kolmogorov equations,
    first passage time,
    random walk,
    Brownian motion,
    renewal process,
    martingale
  ]
tags: [stochastic, probability, Markov chains]
order: 7
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Markov chain:* future depends only on the present state. Discrete time: transition matrix $`P`$; continuous time: generator $`Q`$ (rates), $`p(t) = p(0)e^{tQ}`$.
- *Stationary distribution* $`\pi`$: $`\pi P = \pi`$ or $`\pi Q = 0`$. Compute as a null vector (sparse direct / iterative solve), not by long simulation, when the state space is enumerable.
- *Detailed balance* ($`\pi_i q_{ij} = \pi_j q_{ji}`$) means reversibility: real spectrum, potential-like structure, the basis of MCMC.
- *Mixing time* $`\sim 1/\text{spectral gap}`$. First-passage times solve linear systems.
- *Large or continuous state spaces:* master equation (discrete counts) or Fokker–Planck (continuous); solve directly in low dimensions, simulate sample paths (SSA, SDE) otherwise.

## Decision table

| Process / object | Description | Key computation |
| --- | --- | --- |
| DTMC | $`p_{n+1} = p_nP`$, rows of $`P`$ sum to 1 | $`\pi`$: left eigenvector for eigenvalue 1 |
| CTMC | $`p' = pQ`$, rows of $`Q`$ sum to 0 | $`p(t) = p(0)e^{tQ}`$, $`\pi Q = 0`$ |
| Poisson process | events at rate $`\lambda`$, independent increments | inter-event times $`\sim\operatorname{Exp}(\lambda)`$, counts $`\sim\operatorname{Poisson}(\lambda t)`$ |
| Birth–death process | CTMC on $`\mathbb N`$ with up/down rates | $`\pi`$ in closed form via detailed balance |
| Chemical master equation | CTMC on molecule counts | SSA, finite state projection, moment closure |
| Brownian motion / diffusion | continuous paths, Gaussian increments | SDE simulation, Fokker–Planck PDE |
| Fokker–Planck | PDE for the density of an SDE | grid/spectral solve in ≲ 3–4 dimensions |
| Renewal process | i.i.d. non-exponential waiting times | non-Markovian; augment state or simulate |

| Question | Linear-algebra formulation |
| --- | --- |
| Stationary distribution | $`\pi Q = 0`$, $`\pi\mathbf 1 = 1`$ |
| Mean first-passage time to set $`A`$ | $`Q_{\bar A\bar A}\,\tau = -\mathbf 1`$ |
| Absorption probabilities | $`Q_{\bar A\bar A}\,h = -Q_{\bar A A}\mathbf 1_{\text{target}}`$ |
| Transient distribution | $`p(t) = p(0)e^{tQ}`$ (Krylov / uniformisation) |
| Relaxation / mixing rate | spectral gap of $`Q`$ (second eigenvalue) |

## Variants

### Discrete-time Markov chains

- *Irreducible + aperiodic* (finite state space) ⇒ unique $`\pi`$ and convergence $`p_n \to \pi`$ from any start (ergodic theorem).
- *Convergence rate:* $`\lVert p_n - \pi\rVert \lesssim C\lvert\lambda_2\rvert^n`$, second-largest eigenvalue modulus of $`P`$.
- *Absorbing chains:* fundamental matrix $`(I - Q_T)^{-1}`$ gives expected visits and absorption times.

### Continuous-time Markov chains

- Waiting time in state $`i`$ is exponential with rate $`-q_{ii} = \sum_{j \ne i}q_{ij}`$; the next state is $`j`$ with probability $`q_{ij}/(-q_{ii})`$ (embedded jump chain). This is exactly the Gillespie algorithm.
- *Uniformisation:* $`e^{tQ} = \sum_k e^{-\Lambda t}\frac{(\Lambda t)^k}{k!}P^k`$ with $`P = I + Q/\Lambda`$, $`\Lambda \ge \max_i\lvert q_{ii}\rvert`$. Numerically stable transient solutions.
- *Stationary distribution* of large sparse $`Q`$: solve $`Q^\top\pi = 0`$ with a normalisation row (sparse LU), or GMRES/power iteration on the uniformised chain.

### Poisson processes

- Counting process with independent, stationary increments: $`N(t) \sim \operatorname{Poisson}(\lambda t)`$.
- *Superposition* of independent Poisson processes is Poisson (rates add); *thinning* with probability $`p`$ gives rate $`p\lambda`$.
- *Inhomogeneous rate* $`\lambda(t)`$: simulate by thinning (Lewis–Shedler) or time change $`\Lambda(t) = \int_0^t\lambda`$.

### Master equation and moment methods

- Chemical master equation: CTMC on $`\mathbb N^S`$; infinite or combinatorially large state space.
- *Finite state projection:* truncate to a finite set, solve exactly with error bound given by the lost probability mass.
- *Moment equations:* linear for linear propensities (exact); nonlinear systems need moment closure (normal, log-normal), uncontrolled approximation.
- *Linear noise approximation:* Gaussian fluctuations around the deterministic solution (see [stochastic simulation](/ref/integrate/stochastic-simulation)).

### Diffusions and Fokker–Planck

- SDE $`dX = f\,dt + g\,dW`$ ⇔ density obeys $`\partial_t p = -\nabla\cdot(fp) + \tfrac12\sum_{ij}\partial_i\partial_j\big((gg^\top)_{ij}p\big)`$ (forward Kolmogorov).
- *Backward Kolmogorov equation* governs expectations $`u(x, t) = \mathbb E[\phi(X_T)\mid X_t = x]`$ and first-passage quantities.
- Stationary density for gradient drift $`f = -\nabla U`$ with $`g = \sqrt{2D}`$: $`p \propto e^{-U/D}`$ (Boltzmann).
- *Kramers' rate* for escape over a barrier $`\Delta U`$: $`\propto e^{-\Delta U/D}`$.

## Other lenses

- *Linear algebra.* Everything finite is linear algebra on $`P`$ or $`Q`$: Perron–Frobenius for $`\pi`$, spectral gaps for mixing, sparse solves for first passage, matrix exponentials for transients (see [eigenproblems](/ref/solve/eigenproblems), [matrix functions](/ref/solve/matrix-functions)).
- *Sampling.* MCMC constructs a chain with prescribed $`\pi`$ via detailed balance; convergence diagnostics are empirical estimates of mixing (see [MCMC](/ref/bayes/mcmc)).
- *Graphs.* Random walks on graphs: PageRank is a stationary distribution, commute times relate to effective resistance, Laplacian eigenvectors to metastable sets (spectral clustering, Markov state models).
- *Thermodynamics.* Detailed balance ⇔ equilibrium (no probability currents). Broken detailed balance means entropy production and non-equilibrium steady states, as in driven biochemical cycles.

## Pitfalls & diagnostics

- *Stiff generators:* rates spanning many orders of magnitude make $`e^{tQ}`$ and stationary solves ill-conditioned; use uniformisation, aggregation of fast states, or stiff solvers.
- *Non-Markovian data:* non-exponential waiting times (e.g. multi-step processes) need augmented states (phase-type distributions) to stay Markov.
- *Simulation for stationary statistics* of slow-mixing chains needs very long runs; compute $`\pi`$ directly when possible.
- *Truncated state spaces:* always monitor the probability leaving the truncation.

## See also

- [Stochastic simulation](/ref/integrate/stochastic-simulation)
- [SDEs](/ref/integrate/sdes)
- [MCMC](/ref/bayes/mcmc)
- [Matrix functions](/ref/solve/matrix-functions)
- [Graph algorithms](/ref/graphs/graph-algorithms)

*References:*

- Norris, _Markov Chains_.
- van Kampen, _Stochastic Processes in Physics and Chemistry_ (3rd ed.).
- Levin & Peres, _Markov Chains and Mixing Times_ (2nd ed.).
