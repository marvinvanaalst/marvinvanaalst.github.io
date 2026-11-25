---
title: 'PDEs via the method of lines'
description: 'Discretise space (finite differences, volumes, elements, spectral), integrate in time with an ODE solver: CFL limits, upwinding, IMEX and operator splitting.'
aliases:
  [
    method of lines,
    MOL,
    finite differences,
    finite volumes,
    finite elements,
    spectral methods,
    discontinuous Galerkin,
    CFL condition,
    upwinding,
    flux limiter,
    TVD,
    WENO,
    reaction-diffusion,
    advection-diffusion,
    IMEX,
    operator splitting,
    Crank-Nicolson,
    von Neumann stability
  ]
tags: [PDE, discretisation, time stepping]
order: 7
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Method of lines:* discretise space → large ODE system $`u' = F(u)`$ → hand it to an ODE solver. Space and time accuracy can be chosen and controlled separately.
- *Spatial discretisation by problem type:*
  - Simple geometry, smooth solutions: finite differences or spectral (highest accuracy per unknown).
  - Conservation laws, shocks: finite volumes with upwinding/limiters (or DG).
  - Complex geometry, elliptic/parabolic: finite elements.
- *Time integration by operator type:*
  - Diffusion is stiff ($`\Delta t \propto \Delta x^2`$): implicit (BDF, Rosenbrock with sparse Jacobians) or IMEX.
  - Advection is non-stiff but has a CFL limit ($`\Delta t \propto \Delta x`$): explicit SSP Runge–Kutta.
  - Mixed: IMEX (implicit diffusion/reaction, explicit advection) or operator splitting.

## Decision table

| Spatial method | Accuracy | Geometry | Conservation | Use when |
| --- | --- | --- | --- | --- |
| Finite differences | order of stencil (2, 4, …) | structured grids | not automatic | simple domains, prototyping |
| Finite volumes | 1–2 (higher with WENO/MUSCL) | unstructured OK | exact (fluxes) | conservation laws, shocks, transport |
| Finite elements (CG) | polynomial degree $`p+1`$ | arbitrary meshes | weak form | elliptic/parabolic, elasticity, complex geometry |
| Discontinuous Galerkin | $`p + 1`$, local | arbitrary meshes | local | hyperbolic, high order, $`hp`$-adaptivity |
| Spectral / pseudospectral | exponential (smooth) | periodic, boxes | via formulation | smooth solutions, turbulence, waves |

| Time integrator | Stability | Use for |
| --- | --- | --- |
| Explicit SSP-RK (RK2/RK3) | CFL: $`\lvert a\rvert\Delta t/\Delta x \le C`$ | advection, hyperbolic, with limiters |
| Explicit RK4 / Tsit5 | diffusion: $`D\Delta t/\Delta x^2 \le C`$ | mildly diffusive, small meshes |
| Crank–Nicolson | A-stable (not L-stable) | smooth diffusion; oscillates on sharp data |
| BDF / Rosenbrock (sparse J) | L-stable / stiffly accurate | reaction–diffusion, parabolic, stiff kinetics |
| IMEX RK (ARK, KenCarp) | implicit part unconstrained | advection–diffusion–reaction |
| Exponential integrators | exact for linear part | stiff linear + mild nonlinear |

## Variants

### Finite differences

- Second derivative: $`u_{xx} \approx (u_{i+1} - 2u_i + u_{i-1})/\Delta x^2`$; first derivative central $`(u_{i+1} - u_{i-1})/2\Delta x`$.
- *Boundary conditions:* Dirichlet replace the boundary unknown; Neumann via ghost points or one-sided stencils.
- The semi-discrete 1D diffusion operator has eigenvalues down to $`\approx -4D/\Delta x^2`$, which sets the stiffness.

### Finite volumes and upwinding

- Unknowns are cell averages; update by fluxes through faces: $`\bar u_i' = -(F_{i+1/2} - F_{i-1/2})/\Delta x`$. Conservation is exact by construction.
- *Upwinding:* take the flux from the side the information comes from ($`a \gt 0`$: use $`u_i`$ for face $`i + \tfrac12`$). Central differencing for advection is unstable with explicit Euler and oscillatory in general.
- *High resolution:* MUSCL reconstruction with flux limiters (minmod, van Leer) or WENO; 2nd–5th order in smooth regions without oscillations at shocks (Godunov's theorem: linear schemes above 1st order oscillate).
- Riemann solvers (Roe, HLL, HLLC) for systems like the Euler equations.

### Finite elements

- Weak form: find $`u_h \in V_h`$ with $`a(u_h, v) = \ell(v)`$ for all test functions $`v \in V_h`$. Galerkin = same trial and test space.
- Semi-discretisation gives $`Mu' + Ku = f`$ with mass matrix $`M`$ and stiffness matrix $`K`$: a mass-matrix ODE (lump $`M`$ to a diagonal for explicit methods).
- Advection-dominated problems need stabilisation (SUPG, streamline diffusion) or DG.

### Spectral methods

- Expand in Fourier (periodic) or Chebyshev (non-periodic) series; differentiate exactly in coefficient space or with differentiation matrices.
- Exponential convergence for smooth solutions; FFT makes it $`O(n\log n)`$. Nonlinear terms by pseudospectral evaluation on the grid, with dealiasing (2/3 rule).
- See [Chebyshev & spectral methods](/ref/approx/chebyshev-spectral) and [Fourier & FFT](/ref/approx/fourier-fft).

### Stability: CFL and von Neumann

- *CFL condition:* the numerical domain of dependence must contain the physical one: $`\lvert a\rvert\Delta t/\Delta x \le C`$ (explicit advection).
- *Von Neumann analysis:* insert $`u_j = \hat u\,e^{ikj\Delta x}`$, get an amplification factor $`G(k)`$, require $`\lvert G\rvert \le 1`$ for all $`k`$. Equivalent to the ODE stability region containing $`\Delta t\times`$ the spectrum of the semi-discrete operator.

### Splitting and IMEX

- *Operator splitting:* for $`u_t = A(u) + B(u)`$ (e.g. diffusion + reaction), alternate solving each part. Strang splitting is 2nd order; each substep can use a tailored solver (reaction: stiff ODE per grid point; diffusion: implicit linear solve or FFT).
- *IMEX Runge–Kutta:* one method with paired tableaus treating stiff terms implicitly and the rest explicitly; avoids splitting error and is easier to make high order.

## Other lenses

- *Linear algebra.* Implicit time stepping of parabolic problems is a sequence of sparse linear solves with $`M + \gamma\Delta t K`$; multigrid or sparse direct solvers are the engine (see [multigrid](/ref/solve/multigrid-domain-decomposition)).
- *ODEs.* The semi-discrete system is a stiff ODE whose stiffness grows with resolution; ODE theory (A-stability, L-stability, order reduction at boundaries) applies directly (see [integrate ODEs](/ref/integrate/ode-initial-value)).
- *Machine learning.* Convolutions are finite-difference stencils with learned weights; neural operators and PINNs are alternative PDE solvers with different trade-offs (see [physics-informed ML](/ref/learn/physics-informed-ml)).

## Pitfalls & diagnostics

- *Convergence study:* refine the mesh by 2× and check the error ratio matches the expected order. Without this, results are anecdotes.
- *Oscillations near steep fronts* with central schemes or Crank–Nicolson: use upwinding/limiters or L-stable time stepping.
- *Negative concentrations* from high-order schemes: positivity-preserving limiters or appropriate clipping with conservation fixes.
- *Jacobian sparsity:* pass the banded/sparse structure to the implicit solver or it will form dense Jacobians with $`n`$ evaluations.
- *Boundary conditions* are the most common source of order loss and instability; test them separately (manufactured solutions).

## See also

- [Integrate ODEs](/ref/integrate/ode-initial-value)
- [Finite differences](/ref/diff/finite-differences)
- [Multigrid & domain decomposition](/ref/solve/multigrid-domain-decomposition)
- [Chebyshev & spectral methods](/ref/approx/chebyshev-spectral)
- [Physics-informed ML](/ref/learn/physics-informed-ml)

*References:*

- LeVeque, _Finite Difference Methods for Ordinary and Partial Differential Equations_.
- LeVeque, _Finite Volume Methods for Hyperbolic Problems_.
- Hundsdorfer & Verwer, _Numerical Solution of Time-Dependent Advection-Diffusion-Reaction Equations_.
