---
title: 'Nonlinear equations'
description: 'Bracketing vs open methods in 1D, Newton and quasi-Newton for systems, globalisation, Newton–Krylov for large problems and homotopy continuation.'
aliases:
  [
    root finding,
    bisection,
    Brent,
    regula falsi,
    Illinois,
    secant,
    Newton-Raphson,
    Halley,
    Broyden,
    Newton-Krylov,
    JFNK,
    line search,
    trust region,
    dogleg,
    homotopy continuation,
    fixed-point iteration,
    Anderson acceleration
  ]
tags: [root finding, Newton, nonlinear systems]
order: 5
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Scalar with a bracket:* Brent (or ITP). Guaranteed convergence, superlinear in practice. Bisection if you only trust the sign.
- *Scalar, smooth, good start, cheap derivative:* Newton (quadratic). No derivative: secant (order 1.618).
- *Systems:* Newton with a *line search or trust region* for globalisation. Jacobian too expensive: Broyden. Too large to factor: *Newton–Krylov* (Jacobian-free, GMRES inner solve).
- *Fixed-point form* $`x = g(x)`$: plain iteration converges linearly if $`\lVert g'\rVert \lt 1`$; accelerate with *Anderson*.
- *Hard problems / all solutions:* homotopy continuation.

## Decision table

| Method | Convergence | Cost per iteration | Robustness | Use when |
| --- | --- | --- | --- | --- |
| Bisection | linear (1 bit/step) | 1 $`f`$ | guaranteed with a sign change | last resort, discontinuous $`f`$ |
| Illinois / regula falsi | superlinear (\~1.44) | 1 $`f`$ | guaranteed | simple bracketing |
| Brent / ITP | superlinear, never worse than bisection | 1 $`f`$ | guaranteed | *default scalar with a bracket* |
| Secant | order 1.618 | 1 $`f`$ | local only | scalar, no derivative |
| Newton | quadratic | $`f`$, $`f'`$ | local only | scalar or system with a cheap Jacobian |
| Halley / Householder | cubic | $`f, f', f''`$ | local only | cheap higher derivatives (special functions) |
| Newton + line search / trust region | quadratic near root | $`f`$, $`J`$, linear solve | global in practice | *default for systems* |
| Broyden (good) | superlinear | $`f`$ + rank-1 update | local | expensive Jacobian, moderate $`n`$ |
| Newton–Krylov (JFNK) | quadratic-ish (inexact) | GMRES with $`Jv`$ products | needs preconditioner | large sparse systems, PDEs |
| Anderson acceleration | superlinear-ish | $`g`$ + small least squares | moderate | slow fixed-point iterations (SCF, Picard) |
| Homotopy continuation | follows a path | many Newton corrections | very robust | polynomial systems, all solutions, bad starts |

## Variants

### Bracketing in 1D

- *Bisection:* halve $`[a, b]`$ keeping the sign change. Error $`\le (b - a)/2^k`$.
- *Brent:* combines inverse quadratic interpolation, secant and bisection; falls back to bisection when interpolation misbehaves. Standard in every library.
- *ITP* (interpolate–truncate–project): newer, provably never worse than bisection in the worst case and superlinear on smooth functions.

### Newton's method

- *Scalar:* $`x_{k+1} = x_k - f(x_k)/f'(x_k)`$.
- *System:* solve $`J(x_k)\,\Delta = -F(x_k)`$, update $`x_{k+1} = x_k + \Delta`$.
- Quadratic convergence near a simple root: correct digits double each step. Linear at multiple roots (use $`x - m f/f'`$ with multiplicity $`m`$, or deflation).
- *Cost:* a Jacobian ($`n`$ $`F`$-evaluations by finite differences, colouring/AD for sparse) and a factorisation per iteration. *Simplified Newton* reuses the factorisation for several steps.

### Globalisation

- *Line search* on the merit function $`\phi(x) = \tfrac12\lVert F(x)\rVert^2`$:
  1. Compute the Newton step $`\Delta`$.
  2. Start with $`\alpha = 1`$; backtrack $`\alpha \leftarrow \alpha/2`$ (or by interpolation) until $`\phi(x + \alpha\Delta) \le (1 - c\alpha)\,\phi(x)`$ (Armijo).
  3. Update $`x \leftarrow x + \alpha\Delta`$.
- *Trust region (dogleg, Levenberg–Marquardt style):* minimise the linear model within $`\lVert\Delta\rVert \le r`$; grow or shrink $`r`$ depending on the ratio of actual to predicted reduction. More robust near singular Jacobians.
- Both can still converge to a local minimum of $`\phi`$ that is not a root.

### Quasi-Newton: Broyden

- Maintain $`B_k \approx J`$ with the secant condition $`B_{k+1}s_k = y_k`$ ($`s = \Delta x`$, $`y = \Delta F`$):
  $`B_{k+1} = B_k + \dfrac{(y_k - B_k s_k)s_k^\top}{s_k^\top s_k}`$.
- Update the inverse (Sherman–Morrison) or a factorisation; $`O(n^2)`$ per step instead of $`O(n^3)`$.

### Newton–Krylov

- Solve $`J\Delta = -F`$ with GMRES, needing only products $`Jv \approx [F(x + \epsilon v) - F(x)]/\epsilon`$ (or forward-mode AD). No Jacobian ever stored.
- *Inexact Newton:* stop GMRES at relative residual $`\eta_k`$ (Eisenstat–Walker choice), tightening as the outer iteration converges.
- Needs a preconditioner for the inner linear solves (often from a simplified physics Jacobian).

### Fixed-point iteration and Anderson acceleration

- $`x_{k+1} = g(x_k)`$ converges linearly with rate $`\rho(g'(x^*))`$.
- *Anderson($`m`$):* take the combination of the last $`m`$ iterates that minimises the combined residual $`g(x) - x`$ in least squares; often turns a slow iteration into a fast one. Equivalent to a multisecant quasi-Newton method.

### Homotopy continuation

- Deform an easy system $`G(x) = 0`$ into the target: $`H(x, t) = (1 - t)\,G(x) + t\,F(x)`$.
- Track the solution path from $`t = 0`$ to $`t = 1`$ with predictor-corrector steps (see [bifurcation & continuation](/ref/dynamics/bifurcation-continuation)).
- For polynomial systems with a random complex $`\gamma`$ in the start system, all isolated solutions are found with probability 1 (numerical algebraic geometry).

## Other lenses

- *Optimisation.* Root finding of $`\nabla f`$ is unconstrained optimisation; Newton on $`\nabla f`$ is Newton's optimisation method. Conversely, $`\min\tfrac12\lVert F\rVert^2`$ is nonlinear least squares, and Gauss–Newton on it is Newton for $`F`$.
- *Dynamics.* Newton is forward Euler on $`\dot x = -J^{-1}F(x)`$ with step 1 (Newton flow). Damped Newton takes smaller steps along the same flow; pseudo-transient continuation integrates $`\dot x = F(x)`$ to steady state.
- *Fixed points.* Any root problem can be written as $`x = g(x)`$; Newton is the choice $`g(x) = x - J^{-1}F`$, which makes $`g'(x^*) = 0`$.

## Pitfalls & diagnostics

- *Bad initial guesses* cause divergence or chaos (Newton fractals). Use continuation in a parameter or a physically motivated start.
- *Scaling:* variables of very different magnitude break both the line search criterion and the finite-difference Jacobian. Scale variables and residuals to $`O(1)`$.
- *Wrong Jacobians* turn quadratic into linear convergence. Check against finite differences or AD on a test point.
- *Singular Jacobian at the root* (bifurcation, conservation laws): convergence slows to linear; remove redundant equations (conserved moieties) or use a trust-region / least-squares solver.
- *Convergence test:* check both $`\lVert F\rVert`$ and the step size $`\lVert\Delta\rVert`$, scaled.

## See also

- [Unconstrained local optimisation](/ref/optimise/unconstrained-local)
- [Nonlinear least squares](/ref/optimise/nonlinear-least-squares)
- [Steady states & stability](/ref/dynamics/steady-states-stability)
- [Bifurcation & continuation](/ref/dynamics/bifurcation-continuation)
- [Krylov methods & preconditioning](/ref/solve/krylov-preconditioning)

*References:*

- Kelley, _Solving Nonlinear Equations with Newton's Method_.
- Dennis & Schnabel, _Numerical Methods for Unconstrained Optimization and Nonlinear Equations_.
- Knoll & Keyes, "Jacobian-free Newton–Krylov methods: a survey", J. Comput. Phys. (2004).
