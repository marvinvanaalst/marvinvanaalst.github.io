---
title: 'Steady states & stability'
description: 'Finding steady states by root solving vs time stepping vs pseudo-transient continuation, linear stability from Jacobian eigenvalues, and Lyapunov functions.'
aliases:
  [
    steady state,
    equilibrium,
    fixed point,
    linear stability,
    Jacobian eigenvalues,
    Hartman-Grobman,
    Lyapunov function,
    Lyapunov stability,
    pseudo-transient continuation,
    Routh-Hurwitz,
    phase portrait,
    nullclines,
    saddle,
    node,
    focus,
    center,
    multistability,
    basins of attraction
  ]
tags: [dynamical systems, stability, steady state]
order: 1
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Finding steady states* of $`x' = f(x)`$:
  - Good initial guess: Newton (with line search / trust region) on $`f(x) = 0`$.
  - Robust but slower: integrate to steady state with a stiff solver; or *pseudo-transient continuation* (implicit Euler with growing steps, ends in Newton).
  - All steady states of polynomial (mass-action) systems: homotopy continuation or algebraic methods.
- *Local stability:* eigenvalues of the Jacobian $`J = \partial f/\partial x`$ at the steady state. All $`\operatorname{Re}\lambda \lt 0`$: stable; any $`\operatorname{Re}\lambda \gt 0`$: unstable; zero real parts: linearisation is inconclusive (bifurcation point).
- *Global stability:* find a Lyapunov function, or map basins numerically.
- *Conservation laws* make $`J`$ singular; reduce to independent variables first.

## Decision table

| Method | Finds | Robustness | Cost | Use when |
| --- | --- | --- | --- | --- |
| Newton / trust region on $`f = 0`$ | stable and unstable states | local; needs good guess | few Jacobians + solves | continuation steps, good guesses |
| Time integration to steady state | stable states only | robust (follows dynamics) | long stiff integration | physically relevant attractor, bad guesses |
| Pseudo-transient continuation | stable states (usually) | robust, converges quadratically at the end | moderate | large PDE / network problems |
| Homotopy / algebraic methods | all isolated states | very robust | high | polynomial systems, multistationarity |
| Continuation in a parameter | branches of states | robust along a path | per point: Newton | how states change with parameters |
| Many random starts + Newton | many states | heuristic | embarrassingly parallel | exploring multistability |

| Eigenvalues of $`J`$ (2D) | Type |
| --- | --- |
| real, both negative | stable node |
| real, opposite signs | saddle |
| complex, negative real part | stable focus (spiral) |
| complex, zero real part | centre (linearisation inconclusive) |
| $`\operatorname{tr}J \lt 0`$, $`\det J \gt 0`$ | stable (2D criterion) |

## Variants

### Newton for steady states

- Solve $`f(x) = 0`$ with $`J`$ from AD or analytic expressions; use a trust-region method for robustness (see [nonlinear equations](/ref/solve/nonlinear-roots)).
- Converges equally to stable and unstable states; classify afterwards.
- *Reduced systems:* with conservation laws $`Lx = c`$, replace dependent variables or append the constraints and drop redundant equations; otherwise $`J`$ is singular and Newton fails.

### Pseudo-transient continuation

1. Take an implicit Euler step: solve $`(I/\Delta t - J)\,\delta = f(x)`$, update $`x \leftarrow x + \delta`$.
2. Grow $`\Delta t`$ as the residual shrinks (switched evolution relaxation: $`\Delta t_{k+1} = \Delta t_k\lVert f(x_{k-1})\rVert/\lVert f(x_k)\rVert`$).
3. As $`\Delta t \to \infty`$ the step becomes Newton, giving quadratic convergence at the end.

### Linear stability

- Linearise: $`\delta x' = J\,\delta x`$. By Hartman–Grobman, the nonlinear dynamics near a *hyperbolic* steady state (no eigenvalues on the imaginary axis) look like the linear ones.
- *Timescales:* $`1/\lvert\operatorname{Re}\lambda\rvert`$ are relaxation times; the slowest stable eigenvalue sets the approach to steady state; the spread is the stiffness ratio.
- *Routh–Hurwitz:* conditions on the coefficients of the characteristic polynomial for all roots to have negative real parts (useful symbolically for small systems).
- *Large systems:* compute only the rightmost eigenvalues (shift-invert Arnoldi; see [eigenproblems](/ref/solve/eigenproblems)).

### Lyapunov functions

- $`V(x) \gt 0`$ for $`x \ne x^*`$, $`V(x^*) = 0`$, and $`\dot V = \nabla V\cdot f \le 0`$: stable; $`\lt 0`$: asymptotically stable. Radially unbounded $`V`$ gives global stability.
- *Linear systems:* $`V = x^\top Px`$ with $`J^\top P + PJ = -Q`$ (Lyapunov equation).
- *Chemical networks:* the pseudo-Helmholtz free energy is a Lyapunov function for complex-balanced mass-action systems (see [reaction network theory](/ref/dynamics/reaction-network-theory)).
- *Finding them:* sum-of-squares (SOS) programming for polynomial systems.
- *LaSalle's invariance principle:* with $`\dot V \le 0`$, trajectories approach the largest invariant set where $`\dot V = 0`$.

### Multistability and basins

- Several stable steady states: map basins of attraction by integrating from a grid or random sample of initial conditions.
- Boundaries between basins are stable manifolds of saddles; continuation of saddles shows how basins change.

## Other lenses

- *Optimisation.* Gradient systems $`x' = -\nabla V`$ have only steady states that are critical points of $`V`$, stable ones = local minima. Newton for steady states is Newton for optimisation when $`f`$ is a gradient.
- *Linear algebra.* Stability is a spectral question; non-normal Jacobians allow large transient amplification even when stable (pseudospectra), relevant for excitable and fluid systems.
- *Control.* Steady-state sensitivity $`\partial x^*/\partial p = -J^{-1}\partial f/\partial p`$ (implicit function theorem) is the basis of metabolic control analysis and steady-state gradients without simulation (see [global sensitivity](/ref/dynamics/global-sensitivity)).

## Pitfalls & diagnostics

- *Singular Jacobian at steady state:* conservation laws or genuine bifurcation. Check the rank of the stoichiometric matrix.
- *"Steady state" reached by integration* may be a slow transient or a stable limit cycle with small amplitude; check $`\lVert f(x)\rVert`$ and look at eigenvalues.
- *Scaling:* concentrations spanning orders of magnitude make residual norms meaningless; use scaled or relative residuals.
- *Negative or unphysical solutions:* Newton happily converges to steady states with negative concentrations; constrain or check.
- *Eigenvalues near zero* relative to $`\lVert J\rVert`$: stability classification is numerically uncertain; compute with care or treat as a bifurcation.

## See also

- [Nonlinear equations](/ref/solve/nonlinear-roots)
- [Bifurcation & continuation](/ref/dynamics/bifurcation-continuation)
- [Eigenproblems](/ref/solve/eigenproblems)
- [Stoichiometric analysis](/ref/dynamics/stoichiometric-analysis)
- [Reaction network theory](/ref/dynamics/reaction-network-theory)

*References:*

- Strogatz, _Nonlinear Dynamics and Chaos_ (2nd ed.).
- Kuznetsov, _Elements of Applied Bifurcation Theory_ (3rd ed.), ch. 2.
- Kelley & Keyes, "Convergence analysis of pseudo-transient continuation", SIAM J. Numer. Anal. (1998).
