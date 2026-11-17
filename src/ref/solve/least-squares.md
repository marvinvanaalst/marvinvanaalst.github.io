---
title: 'Least squares'
description: 'Normal equations vs QR vs SVD, rank deficiency, weighted and generalised least squares, Tikhonov regularisation, total least squares and NNLS.'
aliases:
  [
    linear regression,
    normal equations,
    QR least squares,
    SVD,
    pseudoinverse,
    Moore-Penrose,
    Tikhonov,
    ridge,
    weighted least squares,
    generalised least squares,
    total least squares,
    NNLS,
    L-curve,
    rank deficiency
  ]
tags: [linear algebra, regression, regularisation]
order: 4
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- Solve $\min_x \lVert Ax - b\rVert_2$ for tall $A \in \mathbb R^{m\times n}$.
- **Default: Householder QR**, $Rx = Q^\top b$. Backward stable, cost $2mn^2 - \tfrac23 n^3$.
- **Normal equations** ($A^\top A x = A^\top b$, Cholesky) are twice as fast for $m \gg n$ but square the condition number. Only for well-conditioned problems.
- **SVD** for rank-deficient or nearly rank-deficient problems, and when you want to inspect the spectrum. ~2–10× the cost of QR.
- Ill-posed (noisy, ill-conditioned)? Regularise: **Tikhonov/ridge** or truncated SVD, with the parameter from the L-curve, GCV or cross-validation.

## Decision table

| Method                      | Cost (flops, $m \gg n$)      | Accuracy                                              | Rank-deficient            | Use when                                      |
| --------------------------- | ---------------------------- | ----------------------------------------------------- | ------------------------- | --------------------------------------------- |
| Normal equations + Cholesky | $mn^2 + \tfrac13 n^3$        | error $\propto \kappa^2 u$                            | fails                     | well-conditioned, huge $m$, streaming updates |
| Householder QR              | $2mn^2 - \tfrac23 n^3$       | error $\propto \kappa u + \kappa^2 u\,\lVert r\rVert$ | no                        | **default**                                   |
| Column-pivoted QR           | as QR (+ pivoting)           | as QR                                                 | yes (rank-revealing)      | numerical rank, basic solution                |
| SVD                         | $\approx 2mn^2 + 11 n^3$     | best                                                  | yes (minimum norm)        | rank deficiency, diagnostics, TSVD            |
| LSQR / LSMR (iterative)     | matvecs with $A$, $A^\top$   | converges like CG on $A^\top A$                       | yes (minimum norm from 0) | huge sparse or matrix-free                    |
| Tikhonov via augmented QR   | QR of $\binom{A}{\lambda I}$ | stable                                                | regularised               | ill-posed problems                            |

## Variants

### Normal equations

- $A^\top A x = A^\top b$, solve by Cholesky.
- $\kappa(A^\top A) = \kappa(A)^2$: with $\kappa(A) = 10^8$ the result has no correct digits in Float64.
- Advantage: $A^\top A$ and $A^\top b$ can be accumulated in a stream (rows arriving one at a time), $O(n^2)$ memory.

### QR

1. Factor $A = QR$ (Householder), $Q \in \mathbb R^{m\times n}$ orthonormal columns, $R$ upper triangular $n\times n$.
2. Compute $c = Q^\top b$ by applying the reflectors.
3. Solve $Rx = c$. The residual norm is $\lVert (I - QQ^\top)b\rVert$, available for free.

- **Updating:** adding or removing rows/columns can be done in $O(mn)$ with Givens rotations (recursive least squares, active-set methods).

### SVD and the pseudoinverse

- $A = U\Sigma V^\top$, solution $x = \sum_{\sigma_i \gt \tau} \frac{u_i^\top b}{\sigma_i} v_i = A^+ b$.
- Small $\sigma_i$ amplify noise in $u_i^\top b$. **Truncated SVD** drops them (threshold $\tau$); the Picard plot ($\lvert u_i^\top b\rvert$ vs $\sigma_i$) shows where noise dominates.
- For rank-deficient $A$, $A^+b$ is the minimum-norm solution among all minimisers.

### Weighted and generalised least squares

- **Weighted:** $\min\lVert W^{1/2}(Ax - b)\rVert$ with $W = \operatorname{diag}(1/\sigma_i^2)$. Just scale the rows.
- **Generalised:** correlated errors with covariance $\Sigma$: whiten with its Cholesky factor, $L^{-1}A x \approx L^{-1}b$. This is the maximum-likelihood estimator for Gaussian noise.
- **Parameter covariance:** $\operatorname{Cov}(\hat x) = \sigma^2 (A^\top A)^{-1} = \sigma^2 R^{-1}R^{-\top}$, computed from $R$ without forming $A^\top A$.

### Regularisation

- **Tikhonov / ridge:** $\min\lVert Ax - b\rVert^2 + \lambda^2\lVert Lx\rVert^2$. With $L = I$, filter factors $\sigma_i^2/(\sigma_i^2 + \lambda^2)$ smoothly damp small singular values.
- **Choosing $\lambda$:**
  - L-curve: corner of $\log\lVert Lx_\lambda\rVert$ vs $\log\lVert Ax_\lambda - b\rVert$.
  - GCV (generalised cross-validation): minimise $\lVert Ax_\lambda - b\rVert^2 / \operatorname{tr}(I - A A_\lambda^+)^2$.
  - Discrepancy principle: residual ≈ known noise level.
  - Cross-validation (see [resampling](/ref/freq/resampling)).
- **L1 (LASSO)** for sparse solutions: see [sparse recovery](/ref/approx/sparse-recovery).

### Constrained and other variants

- **NNLS** ($x \ge 0$): Lawson–Hanson active-set method; exact and fast for moderate $n$.
- **Equality-constrained:** null-space method or the KKT system.
- **Total least squares** (errors in $A$ too): SVD of $[A\ b]$, solution from the right singular vector of the smallest singular value.
- **Robust regression** (outliers): IRLS with Huber or bisquare weights, see [regression & GLMs](/ref/freq/regression-glms).

## Other lenses

- **Geometry.** $A\hat x$ is the orthogonal projection of $b$ onto $\operatorname{range}(A)$; the residual is orthogonal to the columns ($A^\top r = 0$, which _is_ the normal equations).
- **Statistics.** Under Gaussian noise, least squares is maximum likelihood; Gauss–Markov says it is the best linear unbiased estimator even without Gaussianity. Ridge is the MAP estimate under a Gaussian prior; the posterior covariance is $(A^\top A + \lambda^2 I)^{-1}\sigma^2$.
- **Optimisation.** A convex quadratic. Gradient descent on it converges like $\kappa(A)^2$, CG on the normal equations (CGLS/LSQR) like $\kappa(A)$.

## Pitfalls & diagnostics

- **Column scaling matters** for the conditioning you observe: scale columns to unit norm before judging rank.
- **Polynomial fits in the monomial basis** are badly conditioned. Use orthogonal polynomials or centred, scaled variables.
- **Forming $A^\top A$ explicitly** loses half the digits; avoid unless you know $\kappa$ is small.
- **Residual diagnostics:** plot residuals vs fitted values and vs each regressor; structure means a model error, not noise.
- **Rank deficiency** shows as a drop in the diagonal of pivoted $R$ or in $\sigma_i$; decide the numerical rank with a threshold relative to $\sigma_1$ and the noise level.

## See also

- [Solve Ax = b](/ref/solve/linear-systems)
- [Nonlinear least squares](/ref/optimise/nonlinear-least-squares)
- [Low-rank approximation](/ref/approx/low-rank)
- [Regression & GLMs](/ref/freq/regression-glms)
- [Sparse recovery](/ref/approx/sparse-recovery)

**References:**

- Björck, _Numerical Methods for Least Squares Problems_.
- Golub & Van Loan, _Matrix Computations_, ch. 5.
- Hansen, _Discrete Inverse Problems: Insight and Algorithms_.
