---
title: 'Constrained optimisation'
description: 'Linear programming (simplex vs interior point), quadratic programming, SQP, augmented Lagrangian, barrier methods, projections and variable transforms for bounds.'
aliases:
  [
    linear programming,
    LP,
    simplex,
    interior point,
    IPM,
    quadratic programming,
    QP,
    active set,
    SQP,
    sequential quadratic programming,
    augmented Lagrangian,
    method of multipliers,
    barrier method,
    penalty method,
    projected gradient,
    bound constraints,
    L-BFGS-B,
    IPOPT,
    conic programming,
    SOCP,
    SDP
  ]
tags: [optimisation, constraints, linear programming]
order: 3
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Bounds only:* projected methods (L-BFGS-B, TRF) or a transform ($`\theta = e^\phi`$, logistic). Cheap and robust.
- *LP:* simplex (warm starts, exact vertices, small-medium) or interior point (large sparse, polynomial time). Modern solvers try both.
- *Convex QP / conic (SOCP, SDP):* interior point or ADMM (OSQP); formulate with a modelling layer (disciplined convex programming).
- *General smooth nonlinear constraints:* *SQP* (few variables, accurate, warm starts, e.g. SNOPT, SLSQP) or *interior point* (large sparse, IPOPT). Augmented Lagrangian for very large or structured problems.
- Write the problem so the solver sees structure: linear constraints as linear, sparsity patterns, exact derivatives.

## Decision table

| Problem class | Method | Scale | Notes |
| --- | --- | --- | --- |
| Bounds | L-BFGS-B, projected Newton, TRF | large | or transform variables |
| LP | dual simplex | up to \~$`10^6`$ (sparse) | warm starts (branch and bound, FBA knockouts) |
| LP | interior point (barrier) | very large | crossover to get a vertex |
| Convex QP | active set | small–medium | warm starts (MPC) |
| Convex QP | interior point / ADMM (OSQP) | large | ADMM: moderate accuracy, very fast |
| SOCP / SDP | primal-dual interior point | moderate (SDP: small) | robust optimisation, relaxations, control |
| Smooth NLP, small–medium | SQP (active set QP subproblems) | ≲ $`10^3`$–$`10^4`$ | few iterations, warm starts |
| Smooth NLP, large sparse | interior point (IPOPT, KNITRO) | $`10^5`$+ | needs sparse Hessians/Jacobians |
| Smooth NLP, structured | augmented Lagrangian (LANCELOT, ALM) | large | easy subproblems, distributed |
| Nonsmooth / composite | proximal / ADMM | large | see [convex duality](/ref/optimise/convex-duality) |

## Variants

### Optimality: KKT conditions

For $`\min f(x)`$ s.t. $`c_E(x) = 0`$, $`c_I(x) \ge 0`$, with Lagrangian $`\mathcal L = f - \lambda^\top c`$:

- Stationarity: $`\nabla f = \sum_i\lambda_i\nabla c_i`$.
- Feasibility: $`c_E = 0`$, $`c_I \ge 0`$.
- Dual feasibility: $`\lambda_I \ge 0`$.
- Complementarity: $`\lambda_ic_i = 0`$ for inequalities.
- Multipliers = sensitivity of the optimal value to the constraint ("shadow prices").

### Linear programming

- *Simplex:* walk along vertices of the polytope, pivoting one basis variable at a time. Exponential worst case, excellent in practice; dual simplex re-optimises quickly after bounds change.
- *Interior point:* follow the central path of the barrier problem with Newton steps on the perturbed KKT system; $`O(\sqrt n\log(1/\varepsilon))`$ iterations, each a sparse symmetric indefinite solve.
- *Duality:* every LP has a dual; strong duality holds; dual values give shadow prices (FBA, see [stoichiometric analysis](/ref/dynamics/stoichiometric-analysis)).

### Quadratic programming

- Convex QP ($`Q \succeq 0`$): active-set methods (exact, warm-startable), interior point, or first-order (ADMM/OSQP).
- KKT system for equality-constrained QP: $`\begin{pmatrix} Q & A^\top \\ A & 0\end{pmatrix}\begin{pmatrix} x \\ -\lambda\end{pmatrix} = \begin{pmatrix} -c \\ b\end{pmatrix}`$; solve with $`LDL^\top`$ or null-space methods.

### SQP

1. At $`x_k`$, solve the QP: $`\min_p\nabla f^\top p + \tfrac12p^\top B_kp`$ s.t. linearised constraints $`c + \nabla c^\top p \;(=, \ge)\; 0`$.
2. Line search on a merit function (L1 penalty) or filter, to balance objective and constraint violation.
3. Update $`B_k`$ (BFGS on the Lagrangian Hessian, damped) and multipliers from the QP.

- Few iterations, superlinear convergence; each iteration solves a QP (expensive for many inequalities).

### Interior point (barrier) for NLP

- Replace inequalities by a log barrier: $`\min f(x) - \mu\sum\log s_i`$ s.t. $`c(x) - s = 0`$; drive $`\mu \to 0`$.
- Newton on the primal-dual KKT system each iteration; scales to huge sparse problems if the linear algebra (sparse $`LDL^\top`$, inertia correction) is good.

### Augmented Lagrangian and penalties

- *Quadratic penalty* $`f + \tfrac\mu2\lVert c\rVert^2`$: simple, but needs $`\mu \to \infty`$ (ill-conditioning).
- *Augmented Lagrangian* $`f - \lambda^\top c + \tfrac\mu2\lVert c\rVert^2`$ with multiplier updates $`\lambda \leftarrow \lambda - \mu c`$: exact solutions with finite $`\mu`$; subproblems are bound-constrained (solved by L-BFGS-B).

### Projections and transforms

- *Projected gradient:* step, then project onto the feasible set; efficient when projection is cheap (boxes, simplices, balls).
- *Transforms:* $`\theta = \exp(\phi)`$ (positivity), $`\theta = l + (u - l)\,\sigma(\phi)`$ (intervals), softmax (simplex). Unconstrained solvers then apply directly, but optima on the boundary become unreachable limits.

## Other lenses

- *Duality.* Lagrange multipliers turn constraints into prices; the dual problem gives lower bounds and certificates of optimality (see [convex duality](/ref/optimise/convex-duality)).
- *Linear algebra.* Every method reduces to KKT (saddle-point) systems; their sparse symmetric indefinite factorisation or block preconditioning is the computational core.
- *Geometry.* Constraints define a manifold (equalities) or a region with a boundary (inequalities); manifold optimisation handles nice equality constraints intrinsically (see [manifold optimisation](/ref/optimise/manifold-optimisation)).

## Pitfalls & diagnostics

- *Infeasibility:* check feasibility with a zero objective first; many solvers return a point minimising infeasibility.
- *Degenerate LPs* stall simplex (cycling, many zero-length pivots); modern solvers perturb, but degeneracy also makes dual values non-unique.
- *Scaling* matters even more than unconstrained: rows and columns of constraints with wildly different magnitudes confuse tolerances.
- *Constraint qualification failure* (e.g. redundant equality constraints, $`\nabla c`$ rank-deficient) makes multipliers non-unique and SQP/IPM struggle; remove redundancy.
- *Strict inequalities* don't exist numerically; use small margins.

## See also

- [Convex duality & proximal methods](/ref/optimise/convex-duality)
- [Mixed-integer & SAT/SMT](/ref/optimise/mixed-integer-sat-smt)
- [Optimal control & MPC](/ref/optimise/optimal-control-mpc)
- [Stoichiometric analysis](/ref/dynamics/stoichiometric-analysis)

*References:*

- Nocedal & Wright, _Numerical Optimization_, ch. 12–19.
- Boyd & Vandenberghe, _Convex Optimization_.
- Wächter & Biegler, "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming", Math. Program. (2006).
