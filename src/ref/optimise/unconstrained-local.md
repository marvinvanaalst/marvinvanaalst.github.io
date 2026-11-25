---
title: 'Unconstrained local optimisation'
description: 'Gradient descent, Newton, BFGS and L-BFGS, nonlinear CG, line searches with Wolfe conditions vs trust regions: convergence rates and costs.'
aliases:
  [
    gradient descent,
    steepest descent,
    Newton's method,
    quasi-Newton,
    BFGS,
    L-BFGS,
    SR1,
    nonlinear conjugate gradient,
    line search,
    Wolfe conditions,
    Armijo,
    backtracking,
    trust region,
    dogleg,
    Steihaug-CG,
    Newton-CG,
    truncated Newton,
    convergence rate,
    condition number
  ]
tags: [optimisation, gradient methods, Newton]
order: 1
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Smooth, gradients available, moderate or large $`n`$:* *L-BFGS* with a Wolfe line search. The default for deterministic smooth problems.
- *Small $`n`$ (≲ a few hundred), cheap Hessian:* Newton with a trust region (or modified Cholesky + line search). Quadratic convergence.
- *Large $`n`$, Hessian-vector products available:* Newton–CG (truncated Newton) with a trust region.
- *Gradient descent:* only with good conditioning or as the stochastic version in ML; converges at a rate set by $`\kappa`$.
- *Line search vs trust region:* both globalise; trust regions handle indefinite Hessians and near-singularity more gracefully.
- No gradients? See [global & derivative-free](/ref/optimise/global-derivative-free). Noisy minibatch gradients? See [stochastic gradient](/ref/optimise/stochastic-gradient).

## Decision table

| Method | Needs | Convergence (local) | Cost per iteration | Memory | Use when |
| --- | --- | --- | --- | --- | --- |
| Gradient descent | $`\nabla f`$ | linear, rate $`\frac{\kappa - 1}{\kappa + 1}`$ | 1 gradient | $`O(n)`$ | well-conditioned, huge $`n`$ |
| Nesterov accelerated GD | $`\nabla f`$ | linear, rate \~$`1 - 1/\sqrt\kappa`$ (convex) | 1 gradient | $`O(n)`$ | convex, Lipschitz gradient known |
| Nonlinear CG (PR+, HZ) | $`\nabla f`$ | superlinear-ish on quadratics | 1 gradient + line search | $`O(n)`$ | large $`n`$, low memory |
| L-BFGS ($`m`$ ≈ 5–20) | $`\nabla f`$ | superlinear (in practice) | gradient + $`O(mn)`$ | $`O(mn)`$ | *default smooth* |
| BFGS | $`\nabla f`$ | superlinear | gradient + $`O(n^2)`$ | $`O(n^2)`$ | small/medium $`n`$ |
| Newton (+ TR / LS) | $`\nabla f`$, $`\nabla^2 f`$ | quadratic | Hessian + $`O(n^3)`$ solve | $`O(n^2)`$ | small $`n`$, accurate solutions |
| Newton–CG / truncated Newton | $`\nabla f`$, $`Hv`$ | superlinear–quadratic | several HVPs | $`O(n)`$ | large $`n`$, HVPs via AD |
| Gauss–Newton / LM | residual Jacobian | linear–quadratic | Jacobian + least squares | $`O(mn)`$ | least-squares objectives (see [NLLS](/ref/optimise/nonlinear-least-squares)) |

## Variants

### Line searches

Given a descent direction $`d`$ ($`\nabla f^\top d \lt 0`$), choose a step $`\alpha`$:

- *Armijo (sufficient decrease):* $`f(x + \alpha d) \le f(x) + c_1\alpha\nabla f^\top d`$, $`c_1 \approx 10^{-4}`$.
- *Curvature (Wolfe):* $`\nabla f(x + \alpha d)^\top d \ge c_2\nabla f^\top d`$, $`c_2 = 0.9`$ for quasi-Newton, $`0.1`$ for CG.
- *Backtracking:* start at $`\alpha = 1`$, shrink until Armijo holds. Enough for Newton.
- *Strong Wolfe / Moré–Thuente / Hager–Zhang:* bracketing + interpolation; needed for BFGS/CG so that $`s^\top y \gt 0`$ keeps the quasi-Newton matrix positive definite.

### Trust regions

1. Model $`m(p) = f + g^\top p + \tfrac12p^\top Bp`$ within $`\lVert p\rVert \le \Delta`$.
2. Solve approximately: dogleg (combine Cauchy and Newton steps), Steihaug–CG (CG stopped at the boundary or at negative curvature), or exact (Moré–Sorensen).
3. Ratio $`\rho = (f(x) - f(x + p))/(m(0) - m(p))`$: accept if $`\rho \gt \eta`$; grow $`\Delta`$ if $`\rho \gt 0.75`$ and the step hit the boundary, shrink if $`\rho \lt 0.25`$.

### Newton's method

- Step: solve $`\nabla^2 f\,p = -\nabla f`$. Quadratic convergence near a minimiser with positive definite Hessian.
- Indefinite Hessian: modify (add $`\tau I`$, modified Cholesky) or use a trust region, which exploits negative curvature directions.
- Affine invariant: unaffected by linear rescaling of variables (unlike gradient descent).

### Quasi-Newton: BFGS and L-BFGS

- With $`s_k = x_{k+1} - x_k`$, $`y_k = \nabla f_{k+1} - \nabla f_k`$, $`\rho_k = 1/y_k^\top s_k`$, update the inverse Hessian approximation:
  $`H_{k+1} = (I - \rho_ks_ky_k^\top)H_k(I - \rho_ky_ks_k^\top) + \rho_ks_ks_k^\top`$.
- *L-BFGS* stores only the last $`m`$ pairs $`(s, y)`$ and applies $`H_k`$ to the gradient with the two-loop recursion in $`O(mn)`$, initial scaling $`H_0 = \frac{s^\top y}{y^\top y}I`$.
- *SR1:* symmetric rank-1 update, can capture indefinite curvature; use with trust regions.

### Nonlinear conjugate gradient

- $`d_{k+1} = -\nabla f_{k+1} + \beta_kd_k`$ with Polak–Ribière+ $`\beta = \max\big(0, \nabla f_{k+1}^\top(\nabla f_{k+1} - \nabla f_k)/\lVert\nabla f_k\rVert^2\big)`$ or Hager–Zhang.
- Low memory like GD, usually much faster; restart periodically. L-BFGS is usually better if memory allows.

## Other lenses

- *Dynamics.* Gradient descent is forward Euler on the gradient flow $`x' = -\nabla f`$; Newton is a variable-metric flow; momentum methods discretise damped Hamiltonian dynamics. Ill-conditioning = stiffness of the gradient flow (see [integrate ODEs](/ref/integrate/ode-initial-value)).
- *Linear algebra.* On quadratics, CG is optimal among Krylov methods, BFGS with exact line search reproduces CG, and the convergence rate is all about the spectrum of the Hessian (see [Krylov methods](/ref/solve/krylov-preconditioning)).
- *Root finding.* Minimising $`f`$ is solving $`\nabla f = 0`$; quasi-Newton methods are secant methods for the gradient (see [nonlinear equations](/ref/solve/nonlinear-roots)).

## Pitfalls & diagnostics

- *Scaling:* variables of very different magnitudes destroy GD/CG and slow quasi-Newton; rescale or optimise in log space (positive parameters).
- *Wrong gradients* are the most common failure: line searches fail or convergence is linear. Check with finite differences (see [finite differences](/ref/diff/finite-differences)).
- *Stopping criteria:* combine $`\lVert\nabla f\rVert_\infty \le \text{gtol}`$ (scaled) with relative function change and step size; a small step alone may just mean a failed line search.
- *Noisy objectives* (simulation with tolerances, Monte Carlo) break line searches and quasi-Newton updates; tighten inner tolerances or switch to noise-tolerant / derivative-free methods.
- *Local minima:* all these methods are local; use multistart or global methods when the landscape is multimodal.

## See also

- [Nonlinear least squares](/ref/optimise/nonlinear-least-squares)
- [Constrained optimisation](/ref/optimise/constrained)
- [Stochastic gradient methods](/ref/optimise/stochastic-gradient)
- [Automatic differentiation](/ref/diff/automatic-differentiation)
- [Nonlinear equations](/ref/solve/nonlinear-roots)

*References:*

- Nocedal & Wright, _Numerical Optimization_ (2nd ed.).
- Dennis & Schnabel, _Numerical Methods for Unconstrained Optimization and Nonlinear Equations_.
- Hager & Zhang, "A survey of nonlinear conjugate gradient methods", Pac. J. Optim. (2006).
