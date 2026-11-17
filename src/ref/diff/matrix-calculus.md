---
title: 'Matrix calculus'
description: 'Layout conventions, the differential trick, and the identities for traces, determinants, inverses, quadratic forms and common losses that come up again and again.'
aliases:
  [
    matrix derivative,
    gradient of trace,
    log determinant derivative,
    derivative of inverse,
    numerator layout,
    denominator layout,
    Jacobian layout,
    differential,
    Frechet derivative,
    quadratic form gradient,
    softmax gradient,
    chain rule matrix
  ]
tags: [differentiation, linear algebra, identities]
order: 4
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- Use **differentials**: write $df$ in terms of $dX$, bring it to the form $df = \operatorname{tr}(G^\top dX)$, then $\nabla_X f = G$. This avoids layout confusion entirely.
- Gradients of scalar functions have the **shape of the argument** (that is the convention to keep).
- The handful of identities below cover most of ML, statistics (Gaussian likelihoods) and optimisation.
- In code, you rarely need these by hand: reverse-mode AD computes them. They matter for deriving closed forms, efficient custom adjoints and sanity checks.

## Decision table

| Function $f$                     | Differential $df$                         | Gradient / derivative                       |
| -------------------------------- | ----------------------------------------- | ------------------------------------------- |
| $a^\top x$                       | $a^\top dx$                               | $a$                                         |
| $x^\top A x$                     | $x^\top(A + A^\top)dx$                    | $(A + A^\top)x$ ($2Ax$ if symmetric)        |
| $\lVert Ax - b\rVert^2$          | $2(Ax - b)^\top A\,dx$                    | $2A^\top(Ax - b)$                           |
| $\operatorname{tr}(AX)$          | $\operatorname{tr}(A\,dX)$                | $A^\top$                                    |
| $\operatorname{tr}(X^\top A X)$  | $\operatorname{tr}(X^\top(A + A^\top)dX)$ | $(A + A^\top)X$                             |
| $\log\det X$                     | $\operatorname{tr}(X^{-1}dX)$             | $X^{-\top}$                                 |
| $\det X$                         | $\det X\,\operatorname{tr}(X^{-1}dX)$     | $\det X\cdot X^{-\top}$                     |
| $X^{-1}$                         | $-X^{-1}\,dX\,X^{-1}$                     | (matrix-valued)                             |
| $a^\top X^{-1} b$                | $-a^\top X^{-1}dX\,X^{-1}b$               | $-X^{-\top}ab^\top X^{-\top}$               |
| $\lVert X\rVert_F^2$             | $2\operatorname{tr}(X^\top dX)$           | $2X$                                        |
| $\sigma(x)$ elementwise          | $\sigma'(x)\odot dx$                      | diagonal Jacobian                           |
| $\operatorname{softmax}(z) = s$  | $(\operatorname{diag}(s) - ss^\top)dz$    | Jacobian $\operatorname{diag}(s) - ss^\top$ |
| cross-entropy $-y^\top\log s(z)$ | $(s - y)^\top dz$ (if $\sum y = 1$)       | $s - y$                                     |

## Variants

### Layout conventions

- **Numerator layout:** $\partial y/\partial x$ for $y \in \mathbb R^m$, $x \in \mathbb R^n$ is $m\times n$ (the Jacobian). Gradients of scalars are row vectors.
- **Denominator layout:** the transpose ($n\times m$); gradients are column vectors.
- Papers mix them freely. Keeping "Jacobian = $m\times n$" and "gradient = same shape as the argument" and deriving via differentials avoids errors.

### The differential method

1. Write the function and take the differential using the rules $d(XY) = dX\,Y + X\,dY$, $d(X^{-1}) = -X^{-1}dX\,X^{-1}$, $d\operatorname{tr}(X) = \operatorname{tr}(dX)$, $d\log\det X = \operatorname{tr}(X^{-1}dX)$.
2. Use trace tricks to move $dX$ to the end: $\operatorname{tr}(ABC) = \operatorname{tr}(CAB)$, $a^\top b = \operatorname{tr}(ba^\top)$, $\operatorname{tr}(A^\top) = \operatorname{tr}(A)$.
3. Read off: $df = \operatorname{tr}(G^\top dX) \Rightarrow \nabla_X f = G$.

### Worked example: Gaussian log-likelihood

For $\ell(\mu, \Sigma) = -\tfrac12\log\det\Sigma - \tfrac12(x - \mu)^\top\Sigma^{-1}(x - \mu)$:

- $\nabla_\mu\ell = \Sigma^{-1}(x - \mu)$.
- $\nabla_\Sigma\ell = -\tfrac12\Sigma^{-1} + \tfrac12\Sigma^{-1}(x - \mu)(x - \mu)^\top\Sigma^{-1}$.
- Setting the sum over samples to zero gives the familiar MLEs (sample mean and covariance).
- GP marginal likelihood gradient w.r.t. a kernel hyperparameter $\theta$: $\tfrac12\operatorname{tr}\big((\alpha\alpha^\top - K^{-1})\,\partial K/\partial\theta\big)$ with $\alpha = K^{-1}y$.

### Vectorisation and Kronecker products

- $\operatorname{vec}(AXB) = (B^\top\otimes A)\operatorname{vec}(X)$ turns matrix-valued derivatives into ordinary Jacobians.
- Useful for second derivatives (Hessians of matrix functions) and for solving Sylvester/Lyapunov equations formally; never form the Kronecker product numerically for large matrices.

### Chain rule for compositions

- Scalar loss through matrices: if $Y = g(X)$ and $L = L(Y)$, then $dL = \operatorname{tr}(\bar Y^\top dY)$ and substituting $dY$ in terms of $dX$ gives $\bar X$. This is exactly the reverse-mode rule ("adjoint" or pullback) for $g$.
- Example, matrix product $Y = XW$: $\bar X = \bar Y W^\top$, $\bar W = X^\top\bar Y$.

## Other lenses

- **AD.** Every identity in the table is a custom VJP rule. Writing the rule for $\log\det$ or a linear solve by hand (using a factorisation) beats differentiating through the factorisation algorithm.
- **Functional analysis.** The differential is the Fréchet derivative; the gradient is its Riesz representative under the Frobenius inner product $\langle A, B\rangle = \operatorname{tr}(A^\top B)$. Changing the inner product changes the gradient (natural gradient, see [Fisher information](/ref/info/fisher-information-geometry)).
- **Statistics.** Score functions and Fisher information of multivariate Gaussians are built entirely from these identities.

## Pitfalls & diagnostics

- **Symmetric matrix arguments:** if $X$ is constrained symmetric, the gradient with respect to its free entries differs from the unconstrained formula (off-diagonals counted twice). Symmetrise: $\tfrac12(G + G^\top)$ for the gradient on the symmetric subspace.
- **Complex matrices:** use Wirtinger derivatives and conjugate transposes; the formulas above need $(\cdot)^\top \to (\cdot)^*$ and care with conjugation.
- **Don't invert to compute gradients:** $X^{-1}b$ via a factorisation; $\operatorname{tr}(X^{-1}A)$ via solves or stochastic trace estimation (Hutchinson) for large $X$.
- **Check shapes** at every step; a gradient with the wrong shape is the quickest bug to catch.

## See also

- [Automatic differentiation](/ref/diff/automatic-differentiation)
- [Least squares](/ref/solve/least-squares)
- [Gaussian processes](/ref/approx/gaussian-processes)
- [Estimation & MLE](/ref/freq/estimation-mle)

**References:**

- Magnus & Neudecker, _Matrix Differential Calculus with Applications in Statistics and Econometrics_.
- Petersen & Pedersen, _The Matrix Cookbook_.
- Giles, "An extended collection of matrix derivative results for forward and reverse mode AD" (2008).
