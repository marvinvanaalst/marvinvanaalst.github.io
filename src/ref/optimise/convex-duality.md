---
title: 'Convex duality & proximal methods'
description: 'Convexity, Lagrangian duality and KKT, duality gaps as certificates, proximal operators, proximal gradient and ADMM for composite and structured problems.'
aliases:
  [
    convex optimization,
    convexity,
    Lagrangian duality,
    dual problem,
    strong duality,
    Slater condition,
    KKT,
    duality gap,
    conjugate function,
    Fenchel duality,
    proximal operator,
    proximal gradient,
    ISTA,
    FISTA,
    ADMM,
    Douglas-Rachford,
    primal-dual,
    Chambolle-Pock,
    mirror descent,
    subgradient
  ]
tags: [optimisation, convex, duality, proximal]
order: 4
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Convex problems:** every local minimum is global; duality gives certificates. Recognise convexity and you get reliable solvers.
- **Lagrangian duality:** the dual function $g(\lambda) = \inf_x\mathcal L(x, \lambda)$ is concave and lower-bounds the primal. Strong duality (zero gap) holds under Slater's condition. The **duality gap** is the stopping criterion.
- **Composite problems** $\min f(x) + g(x)$ with smooth $f$ and simple nonsmooth $g$ (L1, indicator of a set, nuclear norm): **proximal gradient / FISTA**.
- **Coupled or split problems** $\min f(x) + g(z)$ s.t. $Ax + Bz = c$: **ADMM**. Robust, parallelisable, modest accuracy.
- First-order methods: cheap iterations, $10^{-3}$–$10^{-6}$ accuracy; interior point: expensive iterations, high accuracy.

## Decision table

| Method                       | Problem form                              | Rate (convex)                       | Per iteration             | Use when                                     |
| ---------------------------- | ----------------------------------------- | ----------------------------------- | ------------------------- | -------------------------------------------- |
| Subgradient method           | nonsmooth convex                          | $O(1/\sqrt k)$                      | 1 subgradient             | last resort; huge-scale, very simple         |
| Proximal gradient (ISTA)     | $f + g$, $f$ $L$-smooth, prox of $g$ easy | $O(1/k)$                            | 1 gradient + 1 prox       | LASSO, constrained least squares             |
| FISTA / accelerated          | same                                      | $O(1/k^2)$                          | same                      | default for composite problems               |
| ADMM                         | $f(x) + g(z)$, $Ax + Bz = c$              | $O(1/k)$ (ergodic)                  | two subproblem solves     | splitting, consensus, distributed, QP (OSQP) |
| Douglas–Rachford             | $f + g$, both with easy prox              | $O(1/k)$                            | two proxes                | feasibility, sums of nonsmooth terms         |
| Primal-dual (Chambolle–Pock) | $f(Kx) + g(x)$                            | $O(1/k)$                            | $K$, $K^\top$, two proxes | imaging (TV), saddle problems                |
| Mirror descent               | geometry other than Euclidean (simplex)   | $O(1/\sqrt k)$                      | 1 gradient + mirror map   | probability simplex, online learning         |
| Interior point               | conic (LP/QP/SOCP/SDP)                    | $O(\log(1/\varepsilon))$ iterations | Newton step               | high accuracy, moderate size                 |
| Dual decomposition           | separable objective, coupling constraints | as subgradient                      | parallel subproblems      | distributed resource allocation              |

## Variants

### Convexity checklist

- Convex functions: affine, norms, $\max$ of convex, $-\log$, $e^x$, quadratic forms with $Q \succeq 0$, log-sum-exp, $-\log\det$ on SPD.
- Preserved by: non-negative sums, composition with affine maps, pointwise maximum, partial minimisation over convex sets.
- **Disciplined convex programming** (CVX, CVXPY, Convex.jl): build problems from these rules and the tool converts them to conic form.

### Lagrangian duality

- Primal $\min f(x)$ s.t. $h_i(x) \le 0$, $Ax = b$. Lagrangian $\mathcal L = f + \sum\lambda_ih_i + \nu^\top(Ax - b)$.
- **Weak duality:** $g(\lambda, \nu) \le p^*$ for $\lambda \ge 0$. **Strong duality** $d^* = p^*$ for convex problems with a strictly feasible point (Slater).
- **Duality gap** $f(x) - g(\lambda, \nu)$ bounds suboptimality: a certificate that a solution is $\varepsilon$-optimal.
- **Conjugate** $f^*(y) = \sup_x(y^\top x - f(x))$: duals of norms are dual norms; the dual of LASSO is a projection problem.

### Proximal operators

- $\operatorname{prox}_{tg}(v) = \arg\min_x\big(g(x) + \tfrac1{2t}\lVert x - v\rVert^2\big)$: a generalised projection.
- **Closed forms:**
  - L1 norm: soft thresholding.
  - Indicator of a convex set: Euclidean projection.
  - L2 norm: block soft thresholding (group LASSO).
  - Nuclear norm: soft thresholding of singular values.
  - $\tfrac12\lVert Ax - b\rVert^2$: a linear solve.
- **Proximal gradient:** $x_{k+1} = \operatorname{prox}_{g/L}\big(x_k - \tfrac1L\nabla f(x_k)\big)$; FISTA adds momentum. Backtracking estimates $L$.

### ADMM

For $\min f(x) + g(z)$ s.t. $Ax + Bz = c$, with penalty $\rho$ and scaled dual $u$:

1. $x \leftarrow \arg\min_x f(x) + \tfrac\rho2\lVert Ax + Bz - c + u\rVert^2$
2. $z \leftarrow \arg\min_z g(z) + \tfrac\rho2\lVert Ax + Bz - c + u\rVert^2$
3. $u \leftarrow u + Ax + Bz - c$
4. Stop when primal residual $\lVert Ax + Bz - c\rVert$ and dual residual $\rho\lVert A^\top B(z - z_{\text{prev}})\rVert$ are small.

- Converges for any $\rho \gt 0$ (convex case), but speed depends heavily on $\rho$ (adapt it to balance residuals).
- Reuse factorisations in step 1 when $A$ and $\rho$ are fixed.

## Other lenses

- **Economics.** Multipliers are prices; dual decomposition is a market where subproblems react to prices and prices adjust to clear constraints.
- **Dynamics.** Proximal steps are implicit Euler steps on gradient flows; gradient steps are explicit Euler. Proximal gradient is forward-backward splitting (an IMEX scheme for optimisation).
- **Statistics.** Regularised estimators (LASSO, ridge, group LASSO, nuclear norm completion) are composite problems; MAP estimation with log-concave priors is convex (see [sparse recovery](/ref/approx/sparse-recovery)).
- **Physics.** Duality gives variational principles in pairs (minimum potential vs complementary energy); optimal transport duality (Kantorovich) underlies Wasserstein distances (see [distribution distances](/ref/info/distribution-distances)).

## Pitfalls & diagnostics

- **Assumed convexity:** composition rules matter; $x/y$ or bilinear terms are not convex. Let a DCP tool verify.
- **ADMM slow tail:** fast to moderate accuracy, slow beyond; polish with a few Newton/active-set steps or switch to interior point.
- **Step sizes:** proximal gradient needs $t \le 1/L$; too large diverges, too small crawls; use backtracking.
- **Report the duality gap** (or residuals) instead of iteration counts as evidence of convergence.
- **Nonconvex problems** with ADMM/proximal methods have no global guarantees; results depend on initialisation.

## See also

- [Constrained optimisation](/ref/optimise/constrained)
- [Sparse recovery](/ref/approx/sparse-recovery)
- [Stochastic gradient methods](/ref/optimise/stochastic-gradient)
- [Distribution distances](/ref/info/distribution-distances)

**References:**

- Boyd & Vandenberghe, _Convex Optimization_.
- Parikh & Boyd, "Proximal algorithms", Found. Trends Optim. (2014).
- Boyd et al., "Distributed optimization and statistical learning via ADMM", Found. Trends Mach. Learn. (2011).
