---
title: 'Interpolation'
description: 'Barycentric polynomial interpolation, Runge phenomenon and Chebyshev nodes, cubic and B-splines, and shape-preserving PCHIP and Akima.'
aliases:
  [
    polynomial interpolation,
    Lagrange interpolation,
    barycentric formula,
    Newton divided differences,
    Runge phenomenon,
    Chebyshev nodes,
    Lebesgue constant,
    cubic spline,
    natural spline,
    not-a-knot,
    B-spline,
    de Boor,
    PCHIP,
    monotone interpolation,
    Akima,
    Hermite interpolation,
    linear interpolation,
    multivariate interpolation,
    scattered data,
    RBF
  ]
tags: [approximation, interpolation, splines]
order: 1
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Tabulated data, need something smooth:** cubic spline ($C^2$). Data must stay monotone/positive (concentrations, CDFs): **PCHIP** or Akima (shape-preserving, $C^1$).
- **You choose the nodes and $f$ is smooth:** polynomial interpolation at **Chebyshev points** with the barycentric formula. Converges exponentially for analytic $f$ (see [Chebyshev & spectral](/ref/approx/chebyshev-spectral)).
- **Equispaced nodes + high-degree polynomial = Runge phenomenon.** Don't.
- **Scattered data in 2D/3D:** piecewise linear on a triangulation, natural neighbour, or radial basis functions (see [surrogates](/ref/approx/surrogates)).
- **Noisy data:** don't interpolate; fit (smoothing splines, least squares).

## Decision table

| Method                                      | Smoothness | Cost (build / eval)                    | Shape preserving | Use when                                 |
| ------------------------------------------- | ---------- | -------------------------------------- | ---------------- | ---------------------------------------- |
| Piecewise linear                            | $C^0$      | $O(n)$ / $O(\log n)$                   | yes              | robust default for data, lookup tables   |
| Cubic spline (natural, not-a-knot, clamped) | $C^2$      | tridiagonal solve $O(n)$ / $O(\log n)$ | no (overshoots)  | smooth data, derivatives needed          |
| PCHIP (Fritsch–Carlson)                     | $C^1$      | $O(n)$ / $O(\log n)$                   | monotone         | monotone or positive data                |
| Akima / makima                              | $C^1$      | $O(n)$ / $O(\log n)$                   | fewer wiggles    | data with abrupt changes                 |
| B-splines (degree $k$)                      | $C^{k-1}$  | banded solve                           | no               | flexible bases, fitting, CAD, regression |
| Polynomial, Chebyshev nodes (barycentric)   | $C^\infty$ | $O(n)$ per eval                        | no               | smooth functions you can sample          |
| Polynomial, equispaced                      | $C^\infty$ | —                                      | no               | only for low degree ($\lesssim 5$)       |
| Rational (AAA, Floater–Hormann)             | $C^\infty$ | $O(nm^2)$ / cheap                      | no               | functions with poles, equispaced data    |
| Trigonometric (FFT)                         | $C^\infty$ | $O(n\log n)$                           | no               | periodic data on uniform grids           |

## Variants

### Polynomial interpolation: barycentric form

- **Second (true) barycentric formula:**
  $p(x) = \dfrac{\sum_j \frac{w_j}{x - x_j}f_j}{\sum_j\frac{w_j}{x - x_j}}$, with weights $w_j = 1/\prod_{k \ne j}(x_j - x_k)$.
- $O(n^2)$ to compute weights once (closed form for Chebyshev points: $w_j = (-1)^j$, halved at the ends), $O(n)$ per evaluation, numerically stable.
- Avoid the monomial basis (Vandermonde matrices are exponentially ill-conditioned) and naive Lagrange evaluation.

### Runge phenomenon and node choice

- Interpolation error: $f(x) - p(x) = \dfrac{f^{(n+1)}(\xi)}{(n+1)!}\prod_j(x - x_j)$.
- Equispaced nodes make $\prod(x - x_j)$ huge near the ends; the Lebesgue constant grows like $2^n$.
- Chebyshev points $x_j = \cos(j\pi/n)$ cluster at the ends; Lebesgue constant $O(\log n)$: near-best approximation.

### Cubic splines

- Piecewise cubics, $C^2$ at the knots. Unknowns: second derivatives at knots, from a **tridiagonal** system (Thomas algorithm).
- **End conditions:**
  - natural ($s'' = 0$): simple but $O(h^2)$ error near the ends
  - not-a-knot (third derivative continuous at the second and second-last knots): default in most libraries, $O(h^4)$
  - clamped (prescribed end slopes)
  - periodic
- Minimises $\int s''^2$ among interpolants (minimal "bending energy").

### Shape-preserving: PCHIP and Akima

- **PCHIP:** piecewise cubic Hermite with slopes chosen from neighbouring secants (harmonic mean, zero at local extrema). Monotone data stay monotone; no overshoot.
- **Akima / modified Akima (makima):** slopes from weighted local secants; less wiggly than cubic splines near sharp changes, not strictly monotone.

### B-splines

- Basis functions with local support (degree $k$ spans $k + 1$ knot intervals), evaluated stably with the de Boor / Cox–de Boor recursion.
- Any spline is a combination of B-splines: interpolation and least-squares fitting become banded linear systems.
- **Smoothing splines / P-splines:** penalise roughness $\lambda\int s''^2$ (or coefficient differences); choose $\lambda$ by GCV.

### Multivariate

- **Tensor product** splines or polynomials on grids; cost grows as $n^d$.
- **Scattered data:** Delaunay triangulation + linear (or Clough–Tocher cubic) interpolation; natural neighbour; RBF interpolation (thin plate splines, Gaussians, polyharmonic) solving a dense $n\times n$ system; kriging = GP regression without noise (see [Gaussian processes](/ref/approx/gaussian-processes)).

## Other lenses

- **Linear algebra.** Interpolation = solving a linear system in some basis; the basis choice controls conditioning (Vandermonde terrible, Chebyshev/Lagrange fine, B-splines banded).
- **Statistics.** Splines are linear smoothers; smoothing splines are posterior means of a GP with a specific (integrated Wiener) prior. Regression splines are linear models in a B-spline basis.
- **Numerical methods.** Quadrature, finite differences and collocation ODE methods all integrate or differentiate interpolants (see [quadrature](/ref/integrate/quadrature), [finite differences](/ref/diff/finite-differences)).

## Pitfalls & diagnostics

- **Extrapolation** with polynomials or splines diverges quickly; clamp or use physically motivated extrapolation.
- **Overshoot** of cubic splines creates negative concentrations or non-monotone CDFs; use PCHIP.
- **Noisy data** interpolated exactly gives wild derivatives; smooth or fit instead.
- **Unsorted or duplicate abscissae** break most implementations.
- **Lookup tables in ODE right-hand sides:** linear interpolation has derivative jumps at knots that upset adaptive solvers; use $C^1$/$C^2$ interpolants or put knots in `tstops`.

## See also

- [Chebyshev & spectral methods](/ref/approx/chebyshev-spectral)
- [Fourier & FFT](/ref/approx/fourier-fft)
- [Surrogate models](/ref/approx/surrogates)
- [Quadrature](/ref/integrate/quadrature)

**References:**

- Trefethen, _Approximation Theory and Approximation Practice_ (extended ed.).
- de Boor, _A Practical Guide to Splines_ (rev. ed.).
- Berrut & Trefethen, "Barycentric Lagrange interpolation", SIAM Review (2004).
