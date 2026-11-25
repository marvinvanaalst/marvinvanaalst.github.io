---
title: 'Quadrature'
description: 'Newton–Cotes, Gauss and Clenshaw–Curtis rules, adaptive Gauss–Kronrod, singular and infinite integrals, and when to switch to Monte Carlo in high dimensions.'
aliases:
  [
    numerical integration,
    trapezoidal rule,
    Simpson,
    Newton-Cotes,
    Gauss-Legendre,
    Gauss-Kronrod,
    Clenshaw-Curtis,
    Gauss-Hermite,
    Gauss-Laguerre,
    adaptive quadrature,
    QUADPACK,
    Romberg,
    tanh-sinh,
    double exponential,
    cubature,
    quadrature rules
  ]
tags: [integration, quadrature]
order: 1
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *1D, smooth, black-box:* adaptive Gauss–Kronrod (G7–K15, QUADPACK `qags`). The default everywhere.
- *Smooth and you choose the nodes:* Gauss–Legendre ($`n`$ points, exact for degree $`2n-1`$) or Clenshaw–Curtis (nested, FFT-friendly, nearly as accurate).
- *Periodic over a full period, or rapidly decaying on $`\mathbb R`$:* the *trapezoidal rule* converges exponentially. Don't use anything fancier.
- *Endpoint singularities / infinite intervals:* tanh-sinh (double exponential) substitution, or Gauss–Jacobi / Gauss–Laguerre / Gauss–Hermite with the weight built in.
- *Dimension ≳ 4–6:* tensor rules die (curse of dimensionality). Sparse grids for smooth integrands up to \~10–20 dimensions, otherwise (quasi-)Monte Carlo.

## Decision table

| Rule | Error (smooth $`f`$) | Nested | Use when | Pitfalls |
| --- | --- | --- | --- | --- |
| Composite trapezoid | $`O(h^2)`$; exponential if periodic | yes | periodic, data on a uniform grid | slow for generic non-periodic $`f`$ |
| Composite Simpson | $`O(h^4)`$ | yes | tabulated data, simple codes | Newton–Cotes of high order is unstable |
| Romberg | Richardson on trapezoid | yes | smooth $`f`$, cheap evaluations | fails for singularities |
| Gauss–Legendre ($`n`$ pts) | exact to degree $`2n - 1`$, \~$`\rho^{-2n}`$ | no | smooth $`f`$, fixed budget | not nested (no free error estimate) |
| Gauss–Kronrod ($`n`$, $`2n+1`$) | Gauss accuracy + error estimate | partly | *adaptive default* | can be fooled by narrow peaks |
| Clenshaw–Curtis | \~Gauss in practice | yes | nested refinement, FFT, Chebyshev workflows | none worth mentioning |
| Gauss–Jacobi / Laguerre / Hermite | exact for weight × polynomial | no | known weight: $`(1 - x)^\alpha(1 + x)^\beta`$, $`e^{-x}`$, $`e^{-x^2}`$ | must factor the weight out |
| Tanh-sinh | exponential, even with endpoint singularities | yes | integrable endpoint singularities, high precision | needs care near endpoints in floating point |
| Monte Carlo / QMC | $`O(N^{-1/2})`$ / $`\approx O(N^{-1})`$ | yes | high dimensions | slow in low dimensions |

## Variants

### Newton–Cotes (equispaced)

- Trapezoid $`\int_a^b f \approx \tfrac h2\big(f_0 + 2f_1 + \dots + 2f_{n-1} + f_n\big)`$, error $`-\tfrac{(b - a)h^2}{12}f''`$.
- Simpson: weights $`\tfrac h3(1, 4, 2, 4, \dots, 4, 1)`$, error $`O(h^4)`$.
- High-order closed Newton–Cotes rules get negative weights and become unstable (Runge phenomenon); use composite low-order rules or Gauss instead.
- *Euler–Maclaurin:* the trapezoid error is a series in $`h^2`$ with coefficients involving endpoint derivatives $`f^{(2k-1)}(b) - f^{(2k-1)}(a)`$. These vanish for periodic functions, hence exponential convergence.

### Gaussian quadrature

- Nodes = roots of the degree-$`n`$ orthogonal polynomial for the weight $`w(x)`$, weights positive. Exact for all polynomials up to degree $`2n - 1`$.
- *Computing nodes and weights:* Golub–Welsch (eigenvalues of the Jacobi matrix, $`O(n^2)`$) for moderate $`n`$; Newton with asymptotic initial guesses (Hale–Townsend, Bogaert) in $`O(n)`$ for huge $`n`$.
- Convergence for analytic $`f`$ is geometric, with rate set by the largest Bernstein ellipse where $`f`$ is analytic.

### Clenshaw–Curtis

- Interpolate at Chebyshev points $`x_k = \cos(k\pi/n)`$ and integrate the interpolant exactly. Weights via FFT in $`O(n\log n)`$.
- Exact only to degree $`n`$, but for most functions as accurate as Gauss with the same number of points, and nested (doubling $`n`$ reuses all evaluations).

### Adaptive quadrature

1. Apply a rule pair (e.g. G7/K15) on $`[a, b]`$: the difference estimates the error.
2. Keep a priority queue of subintervals by error estimate.
3. Bisect the worst one, re-estimate, until the total estimated error is below $`\max(\mathrm{atol}, \mathrm{rtol}\cdot\lvert I\rvert)`$.
4. Optional: extrapolate (Wynn's epsilon algorithm) to accelerate convergence at singularities (QUADPACK `qags`).

### Singularities and infinite ranges

- *Variable substitution:* $`x = \tanh(\tfrac\pi2\sinh t)`$ (tanh-sinh) clusters points doubly exponentially at endpoints; integrable singularities become harmless.
- *Infinite intervals:* map to a finite one ($`x = t/(1 - t^2)`$) or use sinh-sinh / exp-sinh variants; or Gauss–Laguerre / Hermite if the decay matches the weight.
- *Known singularities:* subtract them analytically, or split the interval at interior singular points and kinks.
- *Oscillatory integrals* ($`\int f(x)e^{i\omega x}`$, large $`\omega`$): Filon / Levin methods or contour deformation; standard rules need points per wavelength.

### Multiple dimensions

- *Tensor products* of 1D rules: $`n^d`$ points. Fine for $`d \le 3`$–4.
- *Sparse grids* (Smolyak): combinations of nested 1D rules, $`O(n\log^{d-1}n)`$ points for mixed-smooth integrands (see [high-dimensional numerics](/ref/approx/high-dimensional-numerics)).
- *Monte Carlo* $`O(N^{-1/2})`$ independent of $`d`$; *QMC* (Sobol, lattice rules) close to $`O(N^{-1})`$ for smooth integrands of low effective dimension (see [quasi-Monte Carlo](/ref/random/quasi-monte-carlo)).
- *Complex domains:* triangulate and use simplex rules (Dunavant, Grundmann–Möller).

## Other lenses

- *Interpolation.* Interpolatory rules integrate the polynomial interpolant of $`f`$; quadrature error is integrated interpolation error. Gauss picks nodes to double the exactness.
- *Linear algebra.* Gauss nodes and weights are eigenvalues and (squared first components of) eigenvectors of the tridiagonal Jacobi matrix of the three-term recurrence. Lanczos on a matrix produces exactly this, which is why Lanczos estimates $`v^\top f(A)v`$ (Gauss quadrature for spectral measures).
- *ODEs.* $`\int_a^b f(t)dt`$ is the ODE $`y' = f(t)`$; adaptive ODE solvers work but are overkill. Collocation RK methods (Gauss, Radau, Lobatto) are the quadrature rules of the same names.

## Pitfalls & diagnostics

- *Narrow peaks or discontinuities* between initial nodes can be missed entirely by adaptive schemes. Split the interval at known features.
- *Error estimates are estimates:* check by halving tolerances or comparing two different rules.
- *Noisy integrands* (from simulations): high-order rules amplify noise; reduce the noise or use low-order rules / smoothing.
- *Don't use Newton–Cotes of high order* on equispaced data; fit splines or use composite Simpson.
- *Integrate data, not functions:* with tabulated samples, use trapezoid/Simpson on the grid or integrate a spline; you cannot choose nodes.

## See also

- [Interpolation](/ref/approx/interpolation)
- [Chebyshev & spectral methods](/ref/approx/chebyshev-spectral)
- [Quasi-Monte Carlo](/ref/random/quasi-monte-carlo)
- [High-dimensional numerics](/ref/approx/high-dimensional-numerics)

*References:*

- Davis & Rabinowitz, _Methods of Numerical Integration_ (2nd ed.).
- Trefethen, "Is Gauss quadrature better than Clenshaw–Curtis?", SIAM Review (2008).
- Trefethen & Weideman, "The exponentially convergent trapezoidal rule", SIAM Review (2014).
