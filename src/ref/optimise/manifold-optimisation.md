---
title: 'Optimisation on manifolds'
description: 'Riemannian gradients and Hessians, retractions and vector transport, Stiefel, Grassmann, SPD and SO(3) manifolds, and when to reparametrise instead.'
aliases:
  [
    manifold optimization,
    Riemannian optimization,
    Riemannian gradient,
    retraction,
    vector transport,
    exponential map,
    Stiefel manifold,
    Grassmann manifold,
    SPD manifold,
    orthogonal group,
    SO(3),
    Lie groups,
    sphere constraint,
    Riemannian trust region,
    Manopt,
    natural gradient,
    constrained parameters,
    Cayley transform
  ]
tags: [optimisation, geometry, Lie groups]
order: 11
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- When constraints define a smooth manifold (unit norm, orthogonality, rotations, SPD matrices, fixed rank), optimise **on the manifold**: feasible iterates, unconstrained-style algorithms, no penalties or multipliers.
- **Recipe:** project the Euclidean gradient onto the tangent space (Riemannian gradient), take a step, map back to the manifold with a **retraction** (cheap approximation of the exponential map).
- Riemannian versions of gradient descent, CG, L-BFGS and trust-region Newton exist with the usual convergence properties (Manopt, Pymanopt, Manifolds.jl/Manopt.jl).
- **Alternative:** reparametrise to unconstrained coordinates (Cholesky factor for SPD, exp map for rotations, softmax for the simplex). Simpler, but may introduce singularities or distort conditioning.

## Decision table

| Manifold                      | Points                       | Tangent projection of $G$ at $X$                                                                    | Retraction                                                                      | Applications                                   |
| ----------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------- |
| Sphere $S^{n-1}$              | $\lVert x\rVert = 1$         | $(I - xx^\top)g$                                                                                    | $(x + \xi)/\lVert x + \xi\rVert$                                                | Rayleigh quotient, normalised directions       |
| Stiefel $\mathrm{St}(n, p)$   | $X^\top X = I_p$             | $G - X\operatorname{sym}(X^\top G)$                                                                 | QR or polar decomposition of $X + \xi$                                          | orthonormal frames, PCA, orthogonal layers     |
| Grassmann $\mathrm{Gr}(n, p)$ | subspaces (span of $X$)      | $(I - XX^\top)G$                                                                                    | QR of $X + \xi$                                                                 | subspace tracking, invariant subspaces         |
| Orthogonal / $SO(3)$          | $R^\top R = I$, $\det R = 1$ | $R\operatorname{skew}(R^\top G)$                                                                    | $R\exp(\Omega)$ or Cayley                                                       | rotations, pose estimation, rigid registration |
| SPD $\mathcal P_n$            | $X \succ 0$                  | $\operatorname{sym}(G)$ (affine-invariant metric changes the gradient to $X\operatorname{sym}(G)X$) | $X^{1/2}\exp(X^{-1/2}\xi X^{-1/2})X^{1/2}$ or $X + \xi + \tfrac12\xi X^{-1}\xi$ | covariance estimation, metric learning         |
| Fixed-rank matrices           | rank-$r$ $n\times m$         | via factorisation $USV^\top$                                                                        | truncated SVD of $X + \xi$                                                      | matrix completion, low-rank models             |
| Simplex (interior)            | $x \gt 0$, $\sum x = 1$      | Fisher–Rao metric                                                                                   | normalised multiplicative update                                                | probabilities, mixture weights                 |

## Variants

### Riemannian gradient descent

1. Compute the Euclidean gradient $\nabla f(X)$ (AD works as usual on the embedding).
2. Riemannian gradient: $\operatorname{grad}f(X) = P_X(\nabla f(X))$ (projection onto the tangent space; for non-embedded metrics, also apply the metric's inverse).
3. Step: $X_{k+1} = R_{X_k}(-\alpha\operatorname{grad}f(X_k))$ with retraction $R$ and Armijo line search along the retraction curve.

### Second-order and quasi-Newton

- **Riemannian Hessian:** projected directional derivative of the Riemannian gradient, plus a curvature (Weingarten) term for embedded manifolds.
- **Riemannian trust region** (RTR) with truncated CG: robust, superlinear, standard in Manopt.
- **CG and L-BFGS** need **vector transport** to move previous directions/gradients to the current tangent space (often just projection).

### Lie groups

- Parametrise updates in the Lie algebra: $R \leftarrow R\exp(\widehat\omega)$ with $\omega \in \mathbb R^3$ for $SO(3)$; closed-form Rodrigues exponential.
- Same idea for $SE(3)$ (poses), unitary groups, and for Lie-group ODE integrators (see [geometric integrators](/ref/integrate/geometric-integrators)).

### Reparametrisation instead

- **SPD:** $X = LL^\top$ with positive diagonal (log-diagonal), or $X = \exp(S)$ with symmetric $S$.
- **Rotations:** quaternions with normalisation, or local exp-map updates around the current estimate.
- **Orthogonal matrices:** Cayley transform $Q = (I - A)(I + A)^{-1}$ with skew $A$, or Householder products.
- **Simplex:** softmax / stick-breaking.
- Works with any unconstrained optimiser and AD, but global parametrisations may be singular somewhere (Euler angles: gimbal lock) and can make the landscape harder.

## Other lenses

- **Constrained optimisation.** Manifold methods are what SQP would do with exact feasibility restoration; Lagrange multipliers become implicit in the tangent projection (see [constrained optimisation](/ref/optimise/constrained)).
- **Information geometry.** The natural gradient is the Riemannian gradient under the Fisher metric; mirror descent is its dual-coordinates cousin (see [Fisher information](/ref/info/fisher-information-geometry)).
- **Eigenproblems.** Extreme eigenvectors minimise the Rayleigh quotient on the sphere / Grassmannian; LOBPCG is a Riemannian method in disguise (see [eigenproblems](/ref/solve/eigenproblems)).
- **Graphics and robotics.** Bundle adjustment and SLAM optimise over products of $SE(3)$ with Gauss–Newton on Lie-algebra increments.

## Pitfalls & diagnostics

- **Using Euclidean steps then normalising** works but loses convergence guarantees and can zig-zag; it is a crude retraction without the projected gradient.
- **Metric choice** changes the gradient direction and conditioning (e.g. affine-invariant vs Euclidean on SPD).
- **Gradient check** on manifolds: compare $\langle\operatorname{grad}f, \xi\rangle$ with finite differences of $f(R_X(t\xi))$ for tangent $\xi$.
- **Numerical drift** off the manifold over many iterations; re-orthonormalise occasionally (QR) for Stiefel/rotations.
- **Singularities** of global parametrisations (Euler angles, polar coordinates) cause slow convergence near them; prefer local charts.

## See also

- [Constrained optimisation](/ref/optimise/constrained)
- [Rotations](/ref/space/rotations)
- [Geometric integrators](/ref/integrate/geometric-integrators)
- [Fisher information & geometry](/ref/info/fisher-information-geometry)

**References:**

- Absil, Mahony & Sepulchre, _Optimization Algorithms on Matrix Manifolds_.
- Boumal, _An Introduction to Optimization on Smooth Manifolds_.
- Solà, Deray & Atchuthan, "A micro Lie theory for state estimation in robotics" (2018).
