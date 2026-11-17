---
title: 'Periodic orbits & chaos'
description: 'Computing limit cycles by shooting and collocation, Floquet multipliers, Poincaré maps, Lyapunov exponents and practical chaos diagnostics.'
aliases:
  [
    limit cycle,
    periodic orbit,
    single shooting,
    multiple shooting,
    orthogonal collocation,
    Floquet theory,
    Floquet multipliers,
    monodromy matrix,
    Poincare map,
    Poincare section,
    Lyapunov exponents,
    chaos,
    strange attractor,
    period doubling,
    phase response curve,
    isochrons,
    phase reduction,
    recurrence plot,
    correlation dimension
  ]
tags: [dynamical systems, oscillations, chaos]
order: 6
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Computing a limit cycle:** solve the boundary value problem $x(T) = x(0)$ with unknown period $T$ plus a phase condition. **Shooting** (with Newton, multiple shooting for unstable/stiff systems) or **orthogonal collocation** (AUTO, robust, best for continuation).
- **Stability:** Floquet multipliers = eigenvalues of the monodromy matrix $\partial x(T)/\partial x(0)$. One is always 1; the cycle is stable if all others have $\lvert\mu\rvert \lt 1$.
- **Poincaré maps** reduce a flow to a discrete map on a section; fixed points = periodic orbits.
- **Chaos:** positive largest Lyapunov exponent, computed by integrating the variational equations with periodic re-orthonormalisation (Benettin).
- **Phase reduction:** near a stable cycle, dynamics collapse to a phase variable; phase response curves describe how perturbations shift timing (circadian clocks, neurons).

## Decision table

| Method                           | Computes                          | Robustness               | Use when                                   |
| -------------------------------- | --------------------------------- | ------------------------ | ------------------------------------------ |
| Long simulation + peak detection | stable cycles, period             | simple                   | quick look, stable oscillators             |
| Single shooting                  | cycle, period, monodromy          | fails for unstable/stiff | small, mildly unstable systems             |
| Multiple shooting                | as above                          | good                     | stiff or strongly unstable cycles          |
| Orthogonal collocation           | cycle as a piecewise polynomial   | very good                | continuation, unstable cycles, homoclinics |
| Harmonic balance / Fourier       | cycle as truncated Fourier series | good for smooth cycles   | weakly nonlinear oscillations              |
| Floquet analysis                 | stability of a computed cycle     | —                        | after any of the above                     |
| Benettin / QR method             | Lyapunov spectrum                 | standard                 | chaos detection, attractor dimension       |
| 0–1 test, recurrence plots       | chaos from time series            | heuristic                | data, no model                             |

## Variants

### Shooting

- Unknowns: initial point $x_0$ and period $T$.
- Equations: $\Phi_T(x_0) - x_0 = 0$ ($n$ equations) plus a **phase condition**, e.g. $f(x_0^{\text{ref}})^\top(x_0 - x_0^{\text{ref}}) = 0$ (fix the point on a hyperplane) or a Poincaré-type condition $x_{0,k} = c$.
- Newton's Jacobian needs the monodromy matrix $M = \partial\Phi_T/\partial x_0$ from the variational equations $M' = J(x(t))M$, $M(0) = I$.
- **Multiple shooting:** split $[0, T]$ into segments with continuity conditions; keeps the Newton system well conditioned when trajectories diverge quickly.

### Orthogonal collocation

- Represent the cycle as piecewise polynomials on a mesh of $[0, 1]$ (time scaled by $T$), enforce the ODE at Gauss points, plus periodicity and an integral phase condition $\int_0^1 x(t)^\top\dot x_{\text{ref}}(t)\,dt = 0$.
- Large sparse nonlinear system, solved by Newton; adaptive mesh. Floquet multipliers come from the condensed Jacobian.

### Floquet theory

- Linearise around the cycle: $\delta x' = J(x(t))\,\delta x$ with $T$-periodic $J$. Monodromy matrix $M = \Phi(T)$.
- Eigenvalues $\mu_i$ of $M$: one trivial $\mu = 1$ (along the orbit). Stable if all others satisfy $\lvert\mu_i\rvert \lt 1$. Floquet exponents $\lambda_i = \log(\mu_i)/T$.
- Bifurcations of cycles: $\mu = 1$ (fold of cycles), $\mu = -1$ (period doubling), complex pair on the unit circle (Neimark–Sacker, torus).

### Poincaré maps

- Choose a section $\Sigma$ transversal to the flow (e.g. $x_k = c$ with $\dot x_k \gt 0$). The return map $P : \Sigma \to \Sigma$ turns the flow into a discrete map.
- Compute crossings with event location on the dense output. Fixed points of $P$ = periodic orbits; eigenvalues of $DP$ = nontrivial Floquet multipliers.

### Lyapunov exponents

1. Integrate the trajectory and $k$ tangent vectors with the variational equation.
2. Every $\Delta t$, re-orthonormalise the tangent vectors with a QR decomposition, accumulating $\log\lvert R_{ii}\rvert$.
3. $\lambda_i \approx \frac1t\sum\log\lvert R_{ii}\rvert$ after discarding the transient.

- Largest exponent $\gt 0$: sensitive dependence (chaos). Sum of exponents = average phase-space contraction. Kaplan–Yorke dimension from the spectrum.

### Phase reduction

- Near a stable cycle, $\dot\theta = \omega + Z(\theta)\cdot p(t)$ for weak perturbations $p$, with phase response curve $Z$ (adjoint of the linearisation, computed by integrating the adjoint equation backwards).
- Basis for synchronisation, entrainment and coupled oscillator analysis (Kuramoto).

## Other lenses

- **Boundary value problems.** Limit cycles are periodic BVPs; the same shooting/collocation machinery solves general BVPs and optimal control problems (see [optimal control](/ref/optimise/optimal-control-mpc)).
- **Linear algebra.** Floquet multipliers are eigenvalues of a product of transition matrices; computing them accurately for very unstable cycles needs periodic Schur decompositions rather than forming $M$.
- **Time series.** Spectral analysis gives frequencies but cannot distinguish chaos from noise; Lyapunov exponents and recurrence quantification from embeddings (Takens) try to (see [spectral estimation](/ref/signals/spectral-estimation)).

## Pitfalls & diagnostics

- **Simulation-based period estimates** are biased by transients and slow convergence near bifurcations; compute the cycle as a BVP.
- **Missing phase condition** makes the Newton system singular (any shifted cycle is a solution).
- **Stiff relaxation oscillators** (fast jumps, slow drifts) need adaptive collocation meshes or multiple shooting.
- **Lyapunov exponent estimates** converge slowly; check convergence over time and with different initial points; small positive values may be numerical artefacts.
- **Chaos vs long transients vs noise:** confirm with several diagnostics.

## See also

- [Bifurcation & continuation](/ref/dynamics/bifurcation-continuation)
- [Steady states & stability](/ref/dynamics/steady-states-stability)
- [Step control & events](/ref/integrate/step-control-events)
- [Spectral estimation](/ref/signals/spectral-estimation)

**References:**

- Seydel, _Practical Bifurcation and Stability Analysis_ (3rd ed.).
- Parker & Chua, _Practical Numerical Algorithms for Chaotic Systems_.
- Ermentrout & Terman, _Mathematical Foundations of Neuroscience_ (phase reduction).
