---
title: 'Parallel-in-time & exponential integrators'
description: 'Parareal and PFASST for parallelism across time steps, and exponential integrators (ETD, Rosenbrock-exponential, Krylov φ-functions) for stiff semilinear problems.'
aliases:
  [
    parareal,
    PFASST,
    MGRIT,
    parallel in time,
    time parallel,
    exponential integrators,
    exponential time differencing,
    ETD,
    ETDRK4,
    exponential Rosenbrock,
    Lawson methods,
    integrating factor,
    phi functions,
    IMEX Runge-Kutta,
    additive Runge-Kutta
  ]
tags: [time stepping, parallel, stiff, exponential integrators]
order: 9
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Parallel in time:** when spatial parallelism is exhausted, iterate over the whole time interval at once. **Parareal** (cheap coarse propagator + parallel fine propagators) gives modest speedups (typically 3–10×) for diffusive problems; struggles on hyperbolic/oscillatory ones. **MGRIT** and **PFASST** are multilevel generalisations.
- **Exponential integrators:** for $u' = Lu + N(u)$ with stiff linear $L$, integrate $L$ exactly via $\varphi$-functions and treat $N$ explicitly. Large stable steps without Newton iterations. ETDRK4 is the classic for stiff PDEs (Kuramoto–Sivashinsky, reaction–diffusion with spectral $L$).
- **IMEX Runge–Kutta** (ARK, KenCarp) is the main competitor: implicit for the stiff part, explicit for the rest, needs linear solves instead of $\varphi$-functions.
- Pick exponential methods when $e^{hL}$ or $\varphi_k(hL)v$ is cheap (diagonal in Fourier space, small $L$, good Krylov convergence).

## Decision table

| Method                        | Parallel across time | Stiff handling            | Cost driver                           | Use when                                        |
| ----------------------------- | -------------------- | ------------------------- | ------------------------------------- | ----------------------------------------------- |
| Parareal                      | yes                  | via fine/coarse solvers   | coarse sequential sweeps × iterations | diffusive problems, many idle cores             |
| MGRIT                         | yes                  | via relaxation solvers    | multigrid cycles in time              | long time intervals, parabolic problems         |
| PFASST                        | yes (+ space)        | via SDC sweeps            | spectral deferred correction          | combined space-time parallelism at scale        |
| Integrating factor (Lawson)   | no                   | exact linear part         | $e^{hL}$                              | simple, but poor for stiff $N$ and fixed points |
| ETD / ETDRK2–4                | no                   | exact linear part         | $\varphi_k(hL)v$                      | stiff linear + nonstiff nonlinear, spectral $L$ |
| Exponential Rosenbrock        | no                   | linearise each step ($J$) | $\varphi_k(hJ)v$ via Krylov           | general stiff nonlinear, large sparse           |
| IMEX RK (ARK, KenCarp)        | no                   | implicit stiff part       | linear/nonlinear solves               | stiff + nonstiff splits, robust default         |
| Standard implicit (BDF/Radau) | no                   | everything implicit       | Newton + linear solves                | when no useful splitting exists                 |

## Variants

### Parareal

With a cheap coarse propagator $G$ (large steps, low accuracy) and an accurate fine propagator $F$ over each time slice:

1. Initial guess by a sequential coarse sweep: $U^0_{n+1} = G(U^0_n)$.
2. In parallel over slices, compute $F(U^k_n)$.
3. Sequential correction: $U^{k+1}_{n+1} = G(U^{k+1}_n) + F(U^k_n) - G(U^k_n)$.
4. Repeat until converged (after $k$ iterations the first $k$ slices are exact).

- Speedup ≈ $N/(k\,(1 + N\,c_G/c_F))$ for $N$ slices, $k$ iterations and cost ratio $c_G/c_F$: needs few iterations and a very cheap $G$.
- Convergence is good for dissipative problems, poor for advection-dominated or oscillatory ones.

### MGRIT and PFASST

- **MGRIT:** treat all time steps as unknowns of one big system and solve it with multigrid in time (F/C-relaxation, coarse time grids). Non-intrusive wrapping of existing time steppers.
- **PFASST:** parallel full approximation scheme in space and time, built on spectral deferred correction sweeps; designed for extreme-scale combined parallelism.

### Exponential time differencing (ETD)

- Variation of constants: $u(t_n + h) = e^{hL}u_n + \int_0^h e^{(h - s)L}N(u(t_n + s))\,ds$.
- **ETD1 (exponential Euler):** $u_{n+1} = e^{hL}u_n + h\varphi_1(hL)N(u_n)$.
- **ETDRK4** (Cox–Matthews, with Kassam–Trefethen's contour-integral evaluation of $\varphi$-functions): 4th order, 4 nonlinear evaluations per step.
- **Exponential Rosenbrock:** linearise around the current state, $L = J(u_n)$, $N$ = the remainder; works for general stiff nonlinear problems with Krylov $\varphi$-function evaluation.
- Computing $\varphi_k(hL)v$: diagonal $L$ (Fourier) → elementwise with care for small arguments; general sparse $L$ → Krylov or `phipm` (see [matrix functions](/ref/solve/matrix-functions)).

### IMEX Runge–Kutta

- Split $u' = f_E(u) + f_I(u)$; paired Butcher tableaus with explicit stages for $f_E$ and diagonally implicit stages for $f_I$.
- Each stage solves $(I - h\gamma J_I)k = \text{rhs}$: linear if $f_I$ is linear (diffusion), cheap with FFTs or multigrid.
- Higher-order IMEX (ARK4(3)6L, KenCarp4/5) with embedded error estimates are robust general-purpose choices.

## Other lenses

- **Linear algebra.** Exponential integrators move the work from solving $(I - h\gamma J)x = b$ to evaluating $\varphi(hJ)v$; Krylov makes both "a few matvecs times something", so the trade-off depends on spectra and preconditioners.
- **Multigrid.** MGRIT is literally multigrid applied to the time dimension; parareal is a two-level special case.
- **Optimisation.** Exponential Euler on a linear-quadratic gradient flow gives exact steps; proximal-gradient methods are the splitting analogue in optimisation (see [convex duality & proximal methods](/ref/optimise/convex-duality)).

## Pitfalls & diagnostics

- **Parareal on hyperbolic problems** often needs as many iterations as slices, giving no speedup. Check convergence per iteration before scaling up.
- **Small-argument cancellation:** naive $\varphi_1(z) = (e^z - 1)/z$ loses all accuracy near $z = 0$; use `expm1`, Padé or contour-integral evaluation.
- **Order reduction:** exponential and IMEX methods can lose order for stiff problems with non-trivial boundary conditions; check observed order in convergence tests.
- **Splitting choice matters:** putting a stiff nonlinear reaction term into the "explicit" part reintroduces the stability limit.
- **Wall-clock vs efficiency:** time-parallel methods spend more total work to reduce wall-clock time; only worth it with otherwise idle hardware.

## See also

- [Integrate ODEs](/ref/integrate/ode-initial-value)
- [Matrix functions](/ref/solve/matrix-functions)
- [Method of lines](/ref/integrate/pdes-method-of-lines)
- [Multigrid & domain decomposition](/ref/solve/multigrid-domain-decomposition)

**References:**

- Gander, "50 years of time parallel time integration" (2015).
- Hochbruck & Ostermann, "Exponential integrators", Acta Numerica (2010).
- Kassam & Trefethen, "Fourth-order time-stepping for stiff PDEs", SIAM J. Sci. Comput. (2005).
