---
title: 'Integrate ODEs (initial value problems)'
description: 'Explicit vs implicit, one-step vs multistep, and which integrator wins for non-stiff, stiff and high-accuracy problems.'
aliases:
  [
    Runge-Kutta,
    RK4,
    Dormand-Prince,
    DP5,
    ode45,
    Tsit5,
    Verner,
    DOP853,
    Adams,
    BDF,
    Radau,
    Rosenbrock,
    SDIRK,
    TR-BDF2,
    LSODA,
    stiffness,
    implicit Euler,
    A-stable,
    L-stable
  ]
tags: [ODE, time stepping, stiff, integrator]
order: 2
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

Solve $`y' = f(t, y)`$, $`y(t_0) = y_0`$.

- *Non-stiff:* adaptive explicit RK.
  - 5th order (Tsit5, DP5) at default tolerances.
  - 7th–9th order (Vern7/9, DOP853) for tolerances ≲ $`10^{-8}`$.
- *Stiff:*
  - Rosenbrock (Rodas) for small to medium systems at moderate tolerances.
  - BDF for large systems at loose tolerances.
  - Radau IIA for tight tolerances or DAEs.
- *Unknown stiffness:* an auto-switching solver (LSODA, composite RK/Rosenbrock).
- *Fixed-step RK4:* only when you really need a fixed grid.

## Stiffness

A problem is stiff when the step size is set by *stability*, not accuracy.

- *Mechanism:* fast, strongly damped modes ($`\operatorname{Re}\lambda \ll 0`$ for eigenvalues $`\lambda`$ of $`J = \partial f / \partial y`$) have long since decayed, but an explicit method still needs $`h\lvert\lambda_{\max}\rvert \lesssim C`$ to stay stable.
- *Symptoms:* an explicit solver takes thousands of tiny steps through a smooth solution, with many rejected steps.
- *Typical sources:*
  - chemical kinetics and metabolism (rate constants spanning orders of magnitude)
  - diffusion discretisations ($`\lambda \sim -1/\Delta x^2`$)
  - circuits
  - control loops with fast feedback

## Decision table

| Method | Type | Order | Cost per step | Use when |
| --- | --- | --- | --- | --- |
| Explicit Euler | explicit 1-step | 1 | 1 $`f`$ | teaching; inner loops of other schemes |
| RK4 (classic) | explicit 1-step, fixed $`h`$ | 4 | 4 $`f`$ | fixed grid needed, cheap non-stiff |
| DP5 / Tsit5 | explicit embedded 5(4) | 5 | 6 $`f`$ (FSAL: last stage reused) | *default non-stiff*, rtol \~ $`10^{-3}`$–$`10^{-8}`$ |
| Vern7 / Vern9 / DOP853 | explicit embedded | 7 / 9 / 8 | 10 / 16 / 12 $`f`$ | non-stiff, tight tolerances |
| Adams–Bashforth–Moulton (VO) | explicit multistep, PECE | 1–12 | \~2 $`f`$ | expensive $`f`$, smooth solution, few events |
| BDF 1–5 (VO, e.g. CVODE) | implicit multistep | 1–5 | Newton on $`n \times n`$, \~1–3 iterations | *large stiff systems*, loose–moderate tol |
| Radau IIA (5) | fully implicit RK, 3 stages | 5 | Newton on $`3n`$ → one real + one complex $`n\times n`$ solve | stiff + tight tol, DAEs up to index 3 |
| Rosenbrock (Rodas4/5, ROS3P) | linearly implicit RK | 3–5 | $`s`$ linear solves, one LU, *no Newton* | small–medium stiff, moderate tol |
| SDIRK / ESDIRK (TR-BDF2, Kvaerno) | diagonally implicit RK | 2–5 | $`s`$ sequential Newton solves of size $`n`$ | stiff with inexact Jacobian; IMEX splits |
| Trapezoid / implicit midpoint | implicit 1-step | 2 | 1 Newton solve | A-stable but not L-stable: ringing on stiff modes |
| LSODA | auto Adams ↔ BDF | 1–12 / 1–5 | adaptive | stiffness unknown or changing |

*Rules of thumb:*

- High order pays off at tight tolerances: fewer steps beat more stages.
- Rosenbrock beats BDF for small systems: the LU is cheap and there is no Newton iteration.
- BDF beats one-step methods for big systems: one nonlinear solve per step.
- Radau beats BDF at tight tolerances or for very stiff oscillatory problems, since BDF orders above 2 are not A-stable near the imaginary axis.

## Variants

### Explicit Runge–Kutta (embedded)

- *Stages:* $`k_i = f\big(t + c_i h,\; y + h\sum_{j<i} a_{ij} k_j\big)`$.
- *Update:* $`y_{n+1} = y_n + h\sum_i b_i k_i`$.
- *Error estimate:* a second weight vector $`\hat b`$ gives a lower-order solution, so $`\mathrm{err} = h\sum_i (b_i - \hat b_i)\,k_i`$ is almost free.
- *Step control* (details on [their own page](/ref/integrate/step-control-events)):
  1. Scale per component, $`\mathrm{sc}_i = \mathrm{atol} + \mathrm{rtol}\cdot\max(\lvert y_i\rvert, \lvert y_{n+1,i}\rvert)`$, and take the RMS norm $`e = \lVert \mathrm{err}/\mathrm{sc}\rVert_{\mathrm{RMS}}`$.
  2. Accept the step if $`e \le 1`$.
  3. Next step size: $`h_{\text{new}} = h \cdot \operatorname{clamp}\big(0.9\, e^{-1/(q+1)},\, 0.2,\, 5\big)`$, where $`q`$ is the lower order.
- *FSAL* (first same as last): the last stage of step $`n`$ is the first stage of step $`n+1`$.
- *Dense output:* extra interpolation weights give the solution between steps, so you don't need to shrink $`h`$ just to hit output times.

### Implicit one-step: implicit Euler, Radau, SDIRK

- *Implicit Euler:* $`y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})`$, a nonlinear system per step.
- *Simplified Newton* on $`z = y_n + h f(t_{n+1}, z)`$:
  1. Factor $`W = I - h\gamma J`$ once (implicit Euler: $`\gamma = 1`$).
  2. Iterate $`z \leftarrow z + W^{-1}\big(y_n + h f(t_{n+1}, z) - z\big)`$ until the update is below tolerance.
  3. No convergence within a few iterations → reject the step and reduce $`h`$.
  4. Reuse $`W`$ across iterations and across steps until $`h`$ or $`J`$ change noticeably.
- *Radau IIA:*
  - Collocation at Radau points, $`s`$ coupled stages: a $`3n \times 3n`$ system for $`s = 3`$.
  - Diagonalising the RK matrix reduces this to one real and one complex $`n \times n`$ factorisation.
  - Order $`2s-1`$, L-stable, handles DAEs of index 1 to 3.
- *SDIRK:* lower-triangular RK matrix with equal diagonals, so the stages are solved one after another and all share the same $`W`$.

### Rosenbrock (linearly implicit)

Rosenbrock methods replace Newton with one linear solve per stage using the exact $`J`$:

```math
(I - h\gamma J)\, k_i = h f\Big(y_n + \sum_{j<i}\alpha_{ij}k_j\Big) + h J \sum_{j<i}\gamma_{ij}k_j
```

- One LU per step and no convergence failures.
- An *inaccurate Jacobian ruins the order* (W-methods relax this), so use AD or an analytic $`J`$.

### Multistep: Adams and BDF

Multistep methods reuse past values instead of extra stages.

- *Adams:* integrate an interpolating polynomial through past $`f`$ values. Explicit Adams–Bashforth predicts, implicit Adams–Moulton corrects.
- *BDF:* fit a polynomial through past $`y`$ values and enforce the ODE at the new point, e.g.

```math
\text{BDF2:}\quad y_{n+1} - \tfrac43 y_n + \tfrac13 y_{n-1} = \tfrac23 h\, f(t_{n+1}, y_{n+1})
```

- Both run variable step and variable order (VSVO), and are cheap per step.
- Restarts after discontinuities and events are costly: the method drops back to order 1.

### Stability

Apply the method to $`y' = \lambda y`$ to get $`y_{n+1} = R(z)\,y_n`$ with $`z = h\lambda`$.

| Method | $`R(z)`$ | Property |
| --- | --- | --- |
| explicit Euler | $`1 + z`$ | stable only in a disc around $`-1`$ |
| RK4 | $`\sum_{k=0}^{4} z^k / k!`$ | bounded region, roughly to $`-2.8`$ |
| implicit Euler | $`1 / (1 - z)`$ | *L-stable*: $`R(\infty) = 0`$ |
| trapezoid | $`(1 + z/2)/(1 - z/2)`$ | A-stable, $`\lvert R(\infty)\rvert = 1`$ |

- *A-stable:* $`\lvert R(z)\rvert \le 1`$ for all $`\operatorname{Re} z \le 0`$.
- *L-stable:* also $`R(z) \to 0`$ as $`z \to \infty`$, so stiff modes are damped, not just kept bounded.
- *Dahlquist barriers:*
  - No explicit method is A-stable.
  - A-stable linear multistep methods have order ≤ 2.

### Cost model

Total cost ≈ steps × (stages × cost of $`f`$ + linear algebra). For implicit methods, the linear algebra is:

- *Jacobian:* $`n`$ extra $`f`$-evaluations by finite differences, fewer with sparsity colouring or AD.
- *LU:* $`n^3`$ dense, much less when sparse.
- *Solves:* $`n^2`$ each.

At large $`n`$ the factorisation dominates, so implicit solvers live or die by Jacobian reuse, *sparsity* and, for huge systems, Jacobian-free Newton–Krylov.

## Other lenses

- *Linear algebra.*
  - For $`y' = Ay`$ the exact step is $`e^{hA}`$. Every one-step method approximates it with a rational function $`R(hA)`$: polynomials (Taylor-like) for explicit methods, Padé-like for implicit ones.
  - Stability means $`\lvert R\rvert \le 1`$ on the spectrum of $`hA`$. Stiffness means that spectrum is spread wide.
- *Optimisation.* For a gradient flow $`y' = -\nabla F(y)`$:
  - Explicit Euler is gradient descent. ResNets are explicit Euler too.
  - Implicit Euler is a proximal step, $`y_{n+1} = \arg\min_z F(z) + \tfrac1{2h}\lVert z - y_n\rVert^2`$, which is why it is unconditionally stable on convex $`F`$.
- *Interpolation.*
  - Collocation RK methods (Gauss, Radau, Lobatto) fit a polynomial through the stages.
  - Multistep methods are polynomial interpolation through past values.
- *Geometry.* Generic integrators don't preserve invariants. For Hamiltonian systems, symplectic methods keep the energy error bounded over long times (see [geometric integrators](/ref/integrate/geometric-integrators)).

## Pitfalls & diagnostics

- *Tolerances:*
  - `atol` matters wherever $`\lvert y\rvert`$ approaches 0. With concentrations it is the usual cause of small negative values and blow-ups. Set it per component, below the scale you care about.
  - Loose tolerances on stiff problems can give a stable but *qualitatively wrong* answer. Check convergence by tightening the tolerance 10× and comparing.
- *Discontinuities* in $`f`$ (dosing, switches, `max`, lookup tables) kill error control. Stop exactly at them (`tstops`, events) and restart.
- *Output:* saving densely ≠ stepping densely. Use the interpolant for output instead of forcing a small $`h`$.
- *Benchmarking:* compare solvers with work-precision diagrams (error vs runtime over a tolerance sweep), not runtime at a single tolerance.
- *Slow implicit solver?* Check that the Jacobian is reused and sparse, and that no dense $`n \times n`$ matrix is refactored every step.
- *Energy drift* in long-time conservative dynamics (orbits, molecular dynamics) is an artefact of non-symplectic methods, not physics.

## See also

- [Step control & events](/ref/integrate/step-control-events)
- [DAEs](/ref/integrate/daes)
- [Geometric integrators](/ref/integrate/geometric-integrators)
- [SDEs](/ref/integrate/sdes)
- [Method of lines](/ref/integrate/pdes-method-of-lines)
- [ODE sensitivities](/ref/diff/ode-sensitivities)
- [Solve Ax = b](/ref/solve/linear-systems)
- [Steady states & stability](/ref/dynamics/steady-states-stability)

*References:*

- Hairer, Nørsett & Wanner, _Solving ODEs I: Nonstiff Problems_.
- Hairer & Wanner, _Solving ODEs II: Stiff and DAE Problems_.
- Ascher & Petzold, _Computer Methods for ODEs and DAEs_.
