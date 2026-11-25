---
title: 'Chebyshev & spectral methods'
description: 'Chebyshev series and interpolants, Clenshaw evaluation, spectral convergence, differentiation matrices and Chebfun-style workflows for smooth functions and ODE/PDE solves.'
aliases:
  [
    Chebyshev polynomials,
    Chebyshev series,
    Chebyshev points,
    Clenshaw algorithm,
    spectral convergence,
    spectral methods,
    collocation,
    pseudospectral,
    differentiation matrix,
    Chebfun,
    Bernstein ellipse,
    Legendre polynomials,
    ultraspherical method,
    rootfinding colleague matrix,
    minimax approximation,
    Remez
  ]
tags: [approximation, spectral, polynomials]
order: 2
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- For smooth functions on an interval, *Chebyshev interpolation* is near-optimal and converges exponentially (analytic $`f`$) or algebraically at a rate set by smoothness.
- Represent $`f(x) \approx \sum_{k=0}^n c_kT_k(x)`$; get coefficients from samples at Chebyshev points with one *FFT/DCT*, $`O(n\log n)`$.
- Evaluate with *Clenshaw's recurrence* (stable, $`O(n)`$); choose $`n`$ adaptively by looking at the coefficient decay (Chebfun's strategy).
- Calculus becomes easy: derivatives, integrals, roots (eigenvalues of a colleague matrix), max/min, all to near machine precision.
- *Spectral collocation* solves smooth BVPs/PDEs with very few unknowns; ultraspherical methods give sparse, well-conditioned systems.

## Decision table

| Task | Chebyshev approach | Cost | Notes |
| --- | --- | --- | --- |
| Approximate smooth $`f`$ on $`[a, b]`$ | sample at $`n + 1`$ Chebyshev points, DCT → coefficients | $`O(n\log n)`$ | choose $`n`$ by coefficient decay |
| Evaluate | Clenshaw (coefficients) or barycentric (values) | $`O(n)`$ | both stable |
| Integrate | Clenshaw–Curtis weights / coefficient formula | $`O(n)`$ | near-Gauss accuracy |
| Differentiate | coefficient recurrence or differentiation matrix | $`O(n)`$ / $`O(n^2)`$ | differentiation amplifies noise |
| All roots on the interval | eigenvalues of colleague matrix (recursive subdivision) | $`O(n^3)`$, $`O(n^2)`$ with subdivision | finds every root robustly |
| Best (minimax) approximation | Remez algorithm | iterative | only slightly better than interpolation |
| ODE/PDE BVP, smooth solution | spectral collocation or ultraspherical method | dense $`O(n^3)`$ / sparse | exponential convergence |
| Non-smooth or singular $`f`$ | piecewise Chebyshev (domain splitting) | — | split at kinks / singularities |

## Variants

### Chebyshev polynomials and points

- $`T_k(\cos\theta) = \cos(k\theta)`$; recurrence $`T_{k+1} = 2xT_k - T_{k-1}`$.
- *Chebyshev points (2nd kind):* $`x_j = \cos(j\pi/n)`$, $`j = 0..n`$. Interpolation at these points ⇔ cosine transform of $`f(\cos\theta)`$, so FFT-based.
- Map a general interval $`[a, b]`$ affinely to $`[-1, 1]`$.

### Convergence

- If $`f`$ is analytic inside the Bernstein ellipse $`E_\rho`$ (foci $`\pm 1`$, semi-axes sum $`\rho`$), $`\lvert c_k\rvert \le 2M\rho^{-k}`$: exponential convergence.
- If $`f`$ has $`\nu`$ derivatives with bounded variation, $`\lvert c_k\rvert = O(k^{-\nu - 1})`$: algebraic convergence.
- *Adaptive truncation:* double $`n`$ until trailing coefficients fall below $`\varepsilon\cdot\max\lvert c_k\rvert`$ and plateau; that plateau shows the achievable accuracy.

### Clenshaw's algorithm

To evaluate $`\sum_{k=0}^n c_kT_k(x)`$:

1. $`b_{n+1} = b_{n+2} = 0`$.
2. For $`k = n, \dots, 1`$: $`b_k = c_k + 2xb_{k+1} - b_{k+2}`$.
3. Result: $`c_0 + xb_1 - b_2`$.

### Calculus on Chebyshev expansions

- *Derivative coefficients* from the recurrence $`c'_{k-1} = c'_{k+1} + 2kc_k`$.
- *Indefinite integral* coefficients by the inverse relation; definite integral via Clenshaw–Curtis.
- *Roots:* colleague matrix (Chebyshev analogue of the companion matrix) eigenvalues; Chebfun subdivides intervals so matrices stay small.

### Spectral collocation for ODEs/PDEs

- Unknowns: values at Chebyshev points; derivatives via the dense differentiation matrix $`D`$ ($`D^2`$ for second derivatives). Replace boundary rows with boundary conditions.
- Exponential convergence for smooth solutions; $`D`$ is dense and ill-conditioned ($`\kappa(D^2) \sim n^4`$), fine up to a few hundred points.
- *Ultraspherical (Olver–Townsend) method:* work with coefficients in ultraspherical bases where differentiation is sparse: banded systems, well-conditioned, thousands of unknowns.
- Time-dependent PDEs: spectral in space + ODE solver in time (method of lines); see [method of lines](/ref/integrate/pdes-method-of-lines).

## Other lenses

- *Fourier analysis.* Chebyshev on $`[-1, 1]`$ is Fourier on the circle under $`x = \cos\theta`$: non-periodic smooth functions become even periodic ones, which is why FFT-based transforms and exponential convergence carry over (see [Fourier & FFT](/ref/approx/fourier-fft)).
- *Orthogonal polynomials.* Chebyshev is one member of the Jacobi family; Legendre is natural for Galerkin methods ($`L^2`$ orthogonality), Gauss quadrature nodes are their roots.
- *Linear algebra.* Polynomial rootfinding = eigenvalue problem; interpolation in the right basis = well-conditioned linear algebra.

## Pitfalls & diagnostics

- *Non-smooth functions* (kinks, discontinuities) converge slowly with Gibbs oscillations; split the domain at the singularity.
- *Infinite or semi-infinite domains* need mappings (algebraic, exponential) or other bases (Laguerre, Hermite, rational Chebyshev).
- *Noisy samples:* coefficients plateau at the noise level; truncate there, don't increase $`n`$.
- *Dense collocation* gets expensive and ill-conditioned beyond \~1000 points; use ultraspherical or domain decomposition.
- *High dimensions:* tensor-product Chebyshev grows as $`n^d`$; see [high-dimensional numerics](/ref/approx/high-dimensional-numerics).

## See also

- [Interpolation](/ref/approx/interpolation)
- [Fourier & FFT](/ref/approx/fourier-fft)
- [Quadrature](/ref/integrate/quadrature)
- [Method of lines](/ref/integrate/pdes-method-of-lines)

*References:*

- Trefethen, _Approximation Theory and Approximation Practice_ (extended ed.).
- Trefethen, _Spectral Methods in MATLAB_.
- Boyd, _Chebyshev and Fourier Spectral Methods_ (2nd ed.).
