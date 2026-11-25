---
title: 'Matrix functions'
description: 'Computing exp, log, square roots and φ-functions of matrices: scaling-and-squaring, Schur methods, and Krylov approximations of f(A)v for large problems.'
aliases:
  [
    matrix exponential,
    expm,
    scaling and squaring,
    Pade approximation,
    Krylov expmv,
    expmv,
    phi functions,
    sqrtm,
    logm,
    Schur-Parlett,
    Denman-Beavers,
    Newton-Schulz,
    matrix sign function,
    f(A)b
  ]
tags: [linear algebra, matrix exponential, Krylov]
order: 7
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Dense $`e^A`$:* scaling and squaring with Padé approximants (Higham's `expm`), \~$`20`$–$`30`$ matrix products' worth, backward stable in practice.
- *Large sparse, you need $`e^{tA}v`$, not $`e^{tA}`$:* Krylov (Arnoldi/Lanczos) or truncated Taylor with scaling (Al-Mohy–Higham `expmv`). Never form $`e^{tA}`$ for a sparse $`A`$; it is dense.
- *$`\sqrt{A}`$, $`\log A`$:* Schur-based methods (Björck–Hammarling square root, inverse scaling-and-squaring for log). Newton iterations (Denman–Beavers) are an alternative, less stable.
- *Diagonalisation* $`V f(\Lambda) V^{-1}`$ is fine only for symmetric/normal matrices.
- For time stepping, $`\varphi`$-functions $`\varphi_k(hA)v`$ are the workhorse of exponential integrators.

## Decision table

| Method | Computes | Cost | Use when | Pitfalls |
| --- | --- | --- | --- | --- |
| Eigendecomposition $`V f(\Lambda)V^{-1}`$ | $`f(A)`$ | $`O(n^3)`$ | symmetric / normal $`A`$ | unstable for non-normal $`A`$ ($`\kappa(V)`$ large) |
| Scaling & squaring + Padé | $`e^A`$ | $`O(n^3)`$, \~$`(6 + s)`$ products + 1 solve | dense $`A`$, default | large $`\lVert A\rVert`$ needs many squarings |
| Schur–Parlett | general $`f(A)`$ | $`O(n^3)`$ | general analytic $`f`$, dense | close eigenvalues need blocking |
| Björck–Hammarling (Schur) | $`\sqrt{A}`$ | $`\approx 28 n^3`$ | dense square root | eigenvalues on $`(-\infty, 0]`$ are a problem |
| Inverse scaling & squaring | $`\log A`$ | $`O(n^3)`$ | dense logarithm |  |
| Krylov (Arnoldi/Lanczos) | $`f(A)v`$ | $`m`$ matvecs + small dense $`f(H_m)`$ | large sparse, smooth $`f`$ | many steps for large $`\lVert tA\rVert`$; restart |
| Truncated Taylor + scaling (`expmv`) | $`e^{tA}v`$ | matvecs only | large sparse, moderate norm, many $`t`$ | cost grows with $`\lVert tA\rVert`$ |
| Rational Krylov / contour integrals | $`f(A)v`$ | shifted solves | stiff (large negative spectrum) | needs sparse factorisations |
| Chebyshev expansion | $`f(A)v`$ | matvecs | Hermitian with known spectral interval | needs eigenvalue bounds |

## Variants

### Scaling and squaring

1. Choose $`s`$ so that $`\lVert A/2^s\rVert`$ is small (bound chosen from backward-error analysis, using norm estimates of $`A^k`$).
2. Approximate $`e^{A/2^s} \approx r_m(A/2^s)`$ with the diagonal Padé approximant $`r_m = p_m/q_m`$, $`m \in \{3, 5, 7, 9, 13\}`$: a few matrix products plus one linear solve.
3. Square $`s`$ times: $`e^A = (e^{A/2^s})^{2^s}`$.

- Taylor-based variants (Bader–Blanes–Casas) use only products and are competitive on GPUs.

### Krylov approximation of $`f(A)v`$

1. Run $`m`$ steps of Arnoldi from $`v`$: $`AV_m = V_m H_m + h_{m+1,m}v_{m+1}e_m^\top`$.
2. Approximate $`f(A)v \approx \lVert v\rVert\, V_m f(H_m) e_1`$, computing $`f(H_m)`$ densely ($`m \ll n`$).
3. Monitor the a-posteriori estimate $`\lVert v\rVert\, h_{m+1,m}\,\lvert e_m^\top f(H_m)e_1\rvert`$; split $`[0, t]`$ into substeps if convergence is slow.

- Symmetric $`A`$: Lanczos, convergence governed by how well a polynomial approximates $`f`$ on the spectral interval.
- For stiff $`A`$ (large negative eigenvalues), polynomial Krylov needs many steps; *rational Krylov* (shift-invert) converges much faster.

### φ-functions

- $`\varphi_0(z) = e^z`$, $`\varphi_{k+1}(z) = (\varphi_k(z) - 1/k!)/z`$, e.g. $`\varphi_1(z) = (e^z - 1)/z`$.
- Exact solution of $`y' = Ay + b`$: $`y(h) = e^{hA}y_0 + h\varphi_1(hA)b`$.
- Compute $`\sum_k \varphi_k(hA)v_k`$ in one go via an augmented matrix exponential (Al-Mohy–Higham, `phipm`) or Krylov. See [parallel-in-time & exponential integrators](/ref/integrate/parallel-in-time-exponential).

### Square roots, logarithms, sign

- *Schur method for $`\sqrt{A}`$:* $`A = QTQ^*`$, compute upper triangular $`U`$ with $`U^2 = T`$ column by column, $`\sqrt{A} = QUQ^*`$.
- *Newton-type iterations:* Denman–Beavers for $`\sqrt{A}`$; Newton–Schulz $`X_{k+1} = \tfrac12 X_k(3I - X_k^2)`$ for the sign function and polar decomposition (matmul-only, popular on GPUs, e.g. orthogonalising updates in optimisers).
- *Logarithm:* take repeated square roots until $`A^{1/2^k}`$ is close to $`I`$, apply Padé to $`\log`$, scale back by $`2^k`$.

## Other lenses

- *ODEs.* $`e^{tA}y_0`$ is the solution of $`y' = Ay`$; every one-step ODE method is a rational approximation of $`e^{hA}`$, and Padé approximants of the exponential are exactly the stability functions of implicit RK methods. See [integrate ODEs](/ref/integrate/ode-initial-value).
- *Markov chains.* Transition probabilities of a continuous-time Markov chain are $`e^{tQ}`$; the chemical master equation solution is $`e^{tA}p_0`$ with a huge sparse $`A`$ (finite state projection + Krylov). See [stochastic processes](/ref/dynamics/stochastic-processes).
- *Lie groups.* $`\exp`$ maps a Lie algebra to its group (e.g. skew-symmetric → rotations); Rodrigues' formula is a closed form for $`3\times 3`$. See [rotations](/ref/space/rotations).
- *Graphs.* Communicability and network centrality use $`e^A`$ or $`(I - \alpha A)^{-1}`$ of adjacency matrices.

## Pitfalls & diagnostics

- *Non-normality:* $`\lVert e^{tA}\rVert`$ can grow transiently by orders of magnitude even if all eigenvalues have negative real part, and eigendecomposition-based formulas lose accuracy.
- *Large norms:* $`\lVert A\rVert`$ in the hundreds means many squarings and loss of accuracy for some problems; balancing ($`D^{-1}AD`$) and shifting ($`e^A = e^\mu e^{A - \mu I}`$) help.
- *Sparse in, dense out:* $`e^A`$, $`A^{-1}`$, $`\sqrt{A}`$ are dense even for sparse $`A`$; work with actions $`f(A)v`$.
- *Branch choices:* $`\log`$ and $`\sqrt{\ }`$ of matrices with eigenvalues on the negative real axis have no principal real value.
- *Check:* compare with a different method or higher precision on a small case; for $`e^A`$, check $`e^{A}e^{-A} \approx I`$ only as a sanity test, not as an accuracy guarantee.

## See also

- [Eigenproblems](/ref/solve/eigenproblems)
- [Parallel-in-time & exponential integrators](/ref/integrate/parallel-in-time-exponential)
- [Stochastic processes](/ref/dynamics/stochastic-processes)
- [Rotations](/ref/space/rotations)

*References:*

- Higham, _Functions of Matrices: Theory and Computation_.
- Moler & Van Loan, "Nineteen dubious ways to compute the exponential of a matrix, twenty-five years later", SIAM Review (2003).
- Al-Mohy & Higham, "Computing the action of the matrix exponential", SIAM J. Sci. Comput. (2011).
