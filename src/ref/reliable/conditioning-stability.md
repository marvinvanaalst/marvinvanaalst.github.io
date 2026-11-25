---
title: 'Conditioning & stability'
description: 'Condition numbers, forward vs backward error, stable vs unstable algorithms and scaling: how to tell whether a bad answer is the problem or the method.'
aliases:
  [
    condition number,
    backward error,
    forward error,
    backward stability,
    numerical stability,
    equilibration,
    scaling,
    perturbation theory
  ]
tags: [error analysis, stability, conditioning]
order: 3
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Conditioning* is a property of the _problem_: how much the exact answer changes when the input is perturbed.
- *Stability* is a property of the _algorithm_: whether it computes the exact answer to a nearby problem.
- Rule of thumb: *forward error ≲ condition number × backward error*. A backward-stable algorithm loses about $`\log_{10}\kappa`$ digits and no more.
- Bad answer? Estimate $`\kappa`$. If it is huge, reformulate the problem (rescale, reparametrise, regularise). If it is modest, the algorithm is unstable: change the algorithm.

## Decision table

| Quantity | Definition | Typical value / use |
| --- | --- | --- |
| Relative condition number | $`\kappa(x) = \dfrac{\lVert J(x)\rVert\,\lVert x\rVert}{\lVert f(x)\rVert}`$ | digits lost $`\approx \log_{10}\kappa`$ |
| Matrix condition number | $`\kappa(A) = \lVert A\rVert\,\lVert A^{-1}\rVert = \sigma_{\max}/\sigma_{\min}`$ | linear solves, least squares |
| Backward error | smallest $`\lVert\Delta x\rVert/\lVert x\rVert`$ with $`\hat y = f(x + \Delta x)`$ | $`O(u)`$ for stable algorithms |
| Forward error | $`\lVert\hat y - y\rVert / \lVert y\rVert`$ | what you actually care about |
| Residual (linear systems) | $`\lVert b - A\hat x\rVert / (\lVert A\rVert\,\lVert\hat x\rVert)`$ | computable backward error |
| Mixed / componentwise condition | uses $`\lvert A\rvert`$, $`\lvert x\rvert`$ entrywise (Skeel) | sparse or badly scaled problems, sharper than normwise |

## Variants

### Condition numbers of common problems

- *Sum* $`\sum x_i`$: $`\kappa = \sum\lvert x_i\rvert / \lvert\sum x_i\rvert`$. Cancellation means ill-conditioning.
- *Linear system* $`Ax = b`$: $`\kappa(A)`$. Least squares: $`\kappa(A)`$ plus a $`\kappa(A)^2 \cdot`$ residual term.
- *Eigenvalues:* symmetric matrices are perfectly conditioned (Weyl: shifts ≤ $`\lVert\Delta A\rVert`$). Non-normal matrices can be terribly conditioned; look at the pseudospectrum.
- *Polynomial roots* from coefficients: often horribly conditioned (Wilkinson's polynomial). Prefer eigenvalues of a companion matrix in a good basis, or avoid the monomial basis.
- *Function evaluation:* $`\kappa(x) = \lvert x f'(x)/f(x)\rvert`$. E.g. $`\log x`$ near $`x = 1`$ is ill-conditioned in this sense.

### Stability notions

- *Backward stable:* $`\hat y = f(x + \Delta x)`$ with $`\lVert\Delta x\rVert/\lVert x\rVert = O(u)`$. Householder QR, LU with partial pivoting (in practice), Cholesky.
- *Mixed (forward-backward) stable:* $`\hat y + \Delta y = f(x + \Delta x)`$, both small. Most library functions.
- *Forward stable:* forward error comparable to that of a backward-stable method, without a backward-error argument.
- *Unstable examples:* Gram–Schmidt (classical) for orthogonalisation, normal equations for least squares, Gaussian elimination without pivoting, explicit inverse for solving, the monomial basis for interpolation at many points.

### Scaling and equilibration

- *Row/column scaling* $`D_r A D_c`$ so that rows and columns have similar norms. Reduces $`\kappa`$ for badly scaled problems; LAPACK `geequ`.
- *Variable scaling* in optimisation and ODEs: choose units so variables are $`O(1)`$. Concentrations in nM next to mM, parameters spanning $`10^{-6}`$ to $`10^{6}`$: work in log space or rescale.
- Scaling cannot fix intrinsic ill-conditioning (e.g. nearly collinear columns), but it removes the artificial kind.

### Estimating condition numbers cheaply

- From an existing LU/Cholesky factorisation: LAPACK `gecon`/`pocon` (Hager–Higham estimator), $`O(n^2)`$.
- Via SVD: exact but $`O(n^3)`$.
- Experimentally: perturb inputs by $`\sim\sqrt{u}`$ or $`10^{-8}`$ relative and watch outputs; or rerun in higher precision.

## Other lenses

- *Calculus.* The condition number is the norm of the derivative, scaled to relative quantities. First-order perturbation theory is just the chain rule.
- *Geometry.* For linear systems, $`1/\kappa(A)`$ is the relative distance from $`A`$ to the nearest singular matrix (Eckart–Young). Ill-conditioned means "close to degenerate".
- *Statistics.* Ill-conditioned design matrices are collinear regressors: parameter estimates have huge variances. Practical non-identifiability is ill-conditioning of the Fisher information (see [identifiability](/ref/freq/identifiability)).

## Pitfalls & diagnostics

- *Small residual ≠ accurate solution* when $`\kappa`$ is large. Report both.
- *Don't blame the algorithm first.* Check $`\kappa`$; a backward-stable method cannot do better than $`\kappa u`$.
- *Hidden ill-conditioning from parametrisation:* the same physical problem can be well or ill-conditioned depending on coordinates (monomials vs Chebyshev, rate constants vs log rate constants).
- *Tolerances below $`\kappa u`$ are meaningless:* asking an iterative solver for $`10^{-14}`$ on a problem with $`\kappa = 10^{10}`$ just burns iterations.

## See also

- [Floating-point arithmetic](/ref/reliable/floating-point)
- [Solve Ax = b](/ref/solve/linear-systems)
- [Least squares](/ref/solve/least-squares)
- [Identifiability](/ref/freq/identifiability)

*References:*

- Higham, _Accuracy and Stability of Numerical Algorithms_, ch. 1, 7.
- Trefethen & Bau, _Numerical Linear Algebra_, lectures 12–15.
