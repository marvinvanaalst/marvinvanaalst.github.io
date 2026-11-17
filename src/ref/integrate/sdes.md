---
title: 'Stochastic differential equations'
description: 'Itô vs Stratonovich, Euler–Maruyama, Milstein and stochastic Runge–Kutta methods, strong vs weak convergence, stiffness and noise types.'
aliases:
  [
    SDE,
    Ito,
    Stratonovich,
    Euler-Maruyama,
    Milstein,
    stochastic Runge-Kutta,
    SRI,
    SRA,
    strong convergence,
    weak convergence,
    Wiener process,
    Brownian motion,
    Langevin equation,
    Ornstein-Uhlenbeck,
    additive noise,
    multiplicative noise,
    Fokker-Planck
  ]
tags: [SDE, stochastic, time stepping]
order: 6
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- $dX = f(X, t)\,dt + g(X, t)\,dW$. Decide **Itô or Stratonovich** first; they give different solutions for multiplicative noise.
- **Strong** convergence (pathwise error) matters for individual trajectories, coupling, MLMC; **weak** convergence (error in expectations) matters for statistics. Weak is cheaper.
- **Euler–Maruyama:** strong order 0.5, weak order 1. Fine for weak statistics with additive noise; poor for pathwise accuracy.
- **Milstein:** strong order 1 (needs $g'$; Lévy areas for non-commutative multi-dimensional noise).
- Adaptive higher-order SRK methods: **SRA** for additive noise, **SRI** for diagonal/scalar multiplicative noise (Rößler). Stiff drift: implicit or split methods (drift-implicit EM, SKenCarp).

## Decision table

| Method                         | Strong order              | Weak order | Needs                                                   | Use when                                      |
| ------------------------------ | ------------------------- | ---------- | ------------------------------------------------------- | --------------------------------------------- |
| Euler–Maruyama                 | 0.5 (1 if additive noise) | 1          | $f$, $g$                                                | simple, weak statistics, many cheap paths     |
| Milstein                       | 1                         | 1          | $g\,\partial g/\partial x$ (or derivative-free variant) | pathwise accuracy, scalar/commutative noise   |
| SRA (additive noise)           | 1.5                       | 2          | $f$, $g$                                                | additive noise, adaptive                      |
| SRI (diagonal noise)           | 1.5                       | 2          | $f$, $g$, iterated integrals                            | diagonal multiplicative noise, adaptive       |
| Weak 2nd-order (Talay, Platen) | —                         | 2          | moment-matched increments                               | expectations only                             |
| Drift-implicit EM / θ-methods  | 0.5                       | 1          | Jacobian of $f$                                         | stiff drift                                   |
| Exact / exponential schemes    | exact                     | exact      | closed form                                             | Ornstein–Uhlenbeck, geometric Brownian motion |

## Variants

### Itô vs Stratonovich

- **Itô:** integrand evaluated at the left point; martingale property; natural for finance, population genetics, chemical Langevin.
- **Stratonovich:** midpoint evaluation; ordinary chain rule holds; natural as the limit of smooth (coloured) noise in physics.
- **Conversion:** a Stratonovich SDE with drift $f$ equals an Itô SDE with drift $f + \tfrac12 g\,\partial g/\partial x$ (scalar case).
- With **additive noise** ($g$ independent of $X$) both coincide.

### Euler–Maruyama

- $X_{n+1} = X_n + f(X_n)\,h + g(X_n)\,\Delta W_n$, with $\Delta W_n \sim \mathcal N(0, h)$.
- Converges to the Itô solution. For a Stratonovich SDE, apply it to the converted Itô drift or use a midpoint (Heun) scheme.

### Milstein

- $X_{n+1} = X_n + f h + g\,\Delta W + \tfrac12 g g'\big(\Delta W^2 - h\big)$ (scalar Itô).
- Multi-dimensional noise needs the iterated integrals $I_{ij} = \int\int dW_i\,dW_j$; when noise is non-commutative these include Lévy areas, which are expensive to simulate. That is the usual barrier to strong order 1.

### Stochastic Runge–Kutta and adaptivity

- Derivative-free schemes matching Itô–Taylor expansions to a given order.
- **Adaptive time stepping** for SDEs must sample future Brownian increments consistently with those already drawn: when rejecting a step, split $\Delta W$ using a Brownian bridge (rejection sampling with memory, RSwM) so the path stays the same Brownian motion.

### Stiffness

- Explicit SDE methods inherit the stability limits of their drift part, plus mean-square stability constraints.
- **Drift-implicit** schemes: $X_{n+1} = X_n + f(X_{n+1})h + g(X_n)\Delta W$; implicit in $f$ only (implicit diffusion terms can blow up due to unbounded $\Delta W$).
- IMEX / split-step methods treat stiff drift implicitly and noise explicitly.

## Other lenses

- **PDEs.** The density of $X_t$ obeys the Fokker–Planck equation $\partial_t p = -\nabla\cdot(fp) + \tfrac12\nabla^2 : (gg^\top p)$. For low dimensions, solving Fokker–Planck directly can beat Monte Carlo; for high dimensions, simulate paths (see [stochastic processes](/ref/dynamics/stochastic-processes)).
- **Chemical kinetics.** The chemical Langevin equation is an SDE approximation of the master equation, valid for large molecule counts (see [stochastic simulation](/ref/integrate/stochastic-simulation)).
- **Sampling and ML.** Overdamped Langevin $dX = \nabla\log\pi(X)dt + \sqrt2\,dW$ samples $\pi$ (MALA = Euler–Maruyama + Metropolis correction). Diffusion models learn the score to reverse a noising SDE (see [generative models](/ref/learn/generative-models)).
- **Variance reduction.** Multilevel Monte Carlo couples coarse and fine EM paths with the same Brownian increments; strong order determines its efficiency (see [variance reduction](/ref/random/variance-reduction)).

## Pitfalls & diagnostics

- **Itô/Stratonovich mismatch** is the classic modelling error: same equation, different interpretation, different dynamics (e.g. noise-induced drift).
- **Positivity:** EM can take concentrations or populations negative; use reflecting/truncated schemes, transforms (log-coordinates), or exact schemes where available.
- **Strong vs weak testing:** check strong order by comparing paths at different $h$ driven by the same Brownian path; weak order by comparing moments over many paths (watch Monte Carlo error).
- **Random numbers:** each path needs independent increments; reproducibility across threads requires counter-based RNGs or careful stream splitting (see [PRNGs](/ref/random/prngs)).
- **Monte Carlo error** ($\sigma/\sqrt N$) usually dominates discretisation error for weak quantities; balance them.

## See also

- [Stochastic processes](/ref/dynamics/stochastic-processes)
- [Stochastic simulation](/ref/integrate/stochastic-simulation)
- [Variance reduction](/ref/random/variance-reduction)
- [Integrate ODEs](/ref/integrate/ode-initial-value)

**References:**

- Kloeden & Platen, _Numerical Solution of Stochastic Differential Equations_.
- Higham, "An algorithmic introduction to numerical simulation of SDEs", SIAM Review (2001).
- Rößler, "Runge–Kutta methods for the strong approximation of solutions of SDEs", SIAM J. Numer. Anal. (2010).
