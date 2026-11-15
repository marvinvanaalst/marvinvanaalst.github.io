---
title: 'Integrate ODEs (initial value problems)'
description: 'Explicit vs implicit, one-step vs multistep, and which integrator wins for non-stiff, stiff and high-accuracy problems.'
aliases:
  [
    Runge-Kutta,
    RK4,
    Dormand-Prince,
    DP5,
    ode45,
    Tsit5,
    Verner,
    DOP853,
    Adams,
    BDF,
    Radau,
    Rosenbrock,
    SDIRK,
    TR-BDF2,
    LSODA,
    stiffness,
    implicit Euler,
    A-stable,
    L-stable
  ]
tags: [ODE, time stepping, stiff, integrator]
order: 2
updated: '2026-11-15'
published: true
layout: ref
---

## TL;DR

Solve $y' = f(t, y)$, $y(t_0) = y_0$. **Non-stiff:** adaptive explicit RK, 5th order (Tsit5, DP5) at default tolerances, 7th–9th order (Vern7/9, DOP853) for tolerances ≲ $10^{-8}$. **Stiff:** Rosenbrock (Rodas) for small to medium systems at moderate tolerances, BDF for large systems at loose tolerances, Radau IIA when you need tight tolerances or a DAE. **Unknown stiffness:** an auto-switching solver (LSODA, composite RK/Rosenbrock). Fixed-step RK4 only when you really need a fixed grid.

## Stiffness in one paragraph

A problem is stiff when the step size is set by **stability**, not accuracy. Fast, strongly damped modes ($\operatorname{Re}\lambda \ll 0$ for eigenvalues $\lambda$ of $J = \partial f / \partial y$) have long since decayed, but an explicit method still needs $h\lvert\lambda_{\max}\rvert \lesssim C$ to stay stable. **Symptoms:** an explicit solver takes thousands of tiny steps through a smooth solution, with many rejected steps. **Typical sources:** chemical kinetics and metabolism (rate constants spanning orders of magnitude), diffusion discretisations ($\lambda \sim -1/\Delta x^2$), circuits, and control loops with fast feedback.

## Decision table

| Method                            | Type                        | Order      | Cost per step                                             | Use when                                          |
| --------------------------------- | --------------------------- | ---------- | --------------------------------------------------------- | ------------------------------------------------- |
| Explicit Euler                    | explicit 1-step             | 1          | 1 $f$                                                     | teaching; inner loops of other schemes            |
| RK4 (classic)                     | explicit 1-step, fixed $h$  | 4          | 4 $f$                                                     | fixed grid needed, cheap non-stiff                |
| DP5 / Tsit5                       | explicit embedded 5(4)      | 5          | 6 $f$ (FSAL: last stage reused)                           | **default non-stiff**, rtol ~ $10^{-3}$–$10^{-8}$ |
| Vern7 / Vern9 / DOP853            | explicit embedded           | 7 / 9 / 8  | 10 / 16 / 12 $f$                                          | non-stiff, tight tolerances                       |
| Adams–Bashforth–Moulton (VO)      | explicit multistep, PECE    | 1–12       | ~2 $f$                                                    | expensive $f$, smooth solution, few events        |
| BDF 1–5 (VO, e.g. CVODE)          | implicit multistep          | 1–5        | Newton on $n \times n$, ~1–3 iterations                   | **large stiff systems**, loose–moderate tol       |
| Radau IIA (5)                     | fully implicit RK, 3 stages | 5          | Newton on $3n$ → one real + one complex $n\times n$ solve | stiff + tight tol, DAEs up to index 3             |
| Rosenbrock (Rodas4/5, ROS3P)      | linearly implicit RK        | 3–5        | $s$ linear solves, one LU, **no Newton**                  | small–medium stiff, moderate tol                  |
| SDIRK / ESDIRK (TR-BDF2, Kvaerno) | diagonally implicit RK      | 2–5        | $s$ sequential Newton solves of size $n$                  | stiff with inexact Jacobian; IMEX splits          |
| Trapezoid / implicit midpoint     | implicit 1-step             | 2          | 1 Newton solve                                            | A-stable but not L-stable: ringing on stiff modes |
| LSODA                             | auto Adams ↔ BDF            | 1–12 / 1–5 | adaptive                                                  | stiffness unknown or changing                     |

**Rules of thumb.** High order pays off at tight tolerances (fewer steps beat more stages). Rosenbrock beats BDF for small systems because the LU is cheap and you skip Newton. BDF beats one-step methods for big systems because it does one nonlinear solve per step. Radau beats BDF when the tolerance is tight or the problem is very stiff and oscillatory, since BDF orders above 2 are not A-stable near the imaginary axis.

## Variants

### Explicit Runge–Kutta (embedded)

Stages $k_i = f\big(t + c_i h,\; y + h\sum_{j<i} a_{ij} k_j\big)$, update $y_{n+1} = y_n + h\sum_i b_i k_i$. A second weight vector $\hat b$ gives a lower-order solution, so the error estimate is almost free.

```julia
function rk_step(f, t, y, h, A, b, b̂, c)
    s = length(b); k = Vector{typeof(y)}(undef, s)
    for i in 1:s
        yi = y + h * sum((A[i, j] * k[j] for j in 1:i-1); init=zero(y))
        k[i] = f(t + c[i] * h, yi)
    end
    ynew = y + h * sum(b[i] * k[i] for i in 1:s)
    err = h * sum((b[i] - b̂[i]) * k[i] for i in 1:s)
    return ynew, err
end

function accept_and_next_h(h, err, y, ynew; atol=1e-6, rtol=1e-3, q=4)
    sc = atol .+ rtol .* max.(abs.(y), abs.(ynew))
    e = sqrt(sum(abs2, err ./ sc) / length(y))          # scaled RMS norm
    return e <= 1, h * clamp(0.9 * e^(-1 / (q + 1)), 0.2, 5.0)
end
```

FSAL (first same as last): the last stage of step $n$ is the first stage of step $n+1$. Dense output comes from extra interpolation weights, so you don't need to shrink $h$ just to hit output times. Step control details are on their own page.

### Implicit one-step: implicit Euler, Radau, SDIRK

Implicit Euler: $y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})$, a nonlinear system per step. Newton uses $W = I - h\gamma J$. Factor $W$ once and reuse it across Newton iterations and across steps until $h$ or $J$ change noticeably (simplified Newton).

```julia
function implicit_euler_step(f, J, t, y, h; tol=1e-10, maxit=10)
    W = lu(I - h * J(t + h, y))                 # reuse across iterations
    z = copy(y)
    for _ in 1:maxit
        Δ = W \ (y + h * f(t + h, z) - z)
        z += Δ
        norm(Δ) < tol && return z
    end
    error("Newton did not converge: reduce h")
end
```

**Radau IIA** is collocation at Radau points. Its $s$ stages are coupled, giving a $3n \times 3n$ system for $s=3$. Diagonalising the RK matrix reduces this to one real and one complex $n \times n$ factorisation. It is order $2s-1$, L-stable, and handles index-1 to 3 DAEs. **SDIRK** has a lower-triangular RK matrix with equal diagonals, so the stages are solved sequentially and all share the same $W$.

### Rosenbrock (linearly implicit)

Rosenbrock methods replace Newton with one linear solve per stage using the exact $J$:

$$
(I - h\gamma J)\, k_i = h f\Big(y_n + \sum_{j<i}\alpha_{ij}k_j\Big) + h J \sum_{j<i}\gamma_{ij}k_j
$$

There is one LU per step and no convergence failures. However, an **inaccurate Jacobian ruins the order** (W-methods relax this), so use AD or an analytic $J$.

### Multistep: Adams and BDF

Multistep methods reuse past values instead of extra stages. **Adams** methods integrate an interpolating polynomial through past $f$ values: explicit Adams–Bashforth predicts, implicit Adams–Moulton corrects. **BDF** fits a polynomial through past $y$ values and enforces the ODE at the new point:

$$
\text{BDF2:}\quad y_{n+1} - \tfrac43 y_n + \tfrac13 y_{n-1} = \tfrac23 h\, f(t_{n+1}, y_{n+1})
$$

Both run variable step and variable order (VSVO), and are cheap per step. Restarts after discontinuities and events are costly: the method drops back to order 1.

### Stability in one formula

Apply the method to $y' = \lambda y$ to get $y_{n+1} = R(z)\,y_n$ with $z = h\lambda$.

| Method         | $R(z)$                    | Property                               |
| -------------- | ------------------------- | -------------------------------------- |
| explicit Euler | $1 + z$                   | stable only in a disc around $-1$      |
| RK4            | $\sum_{k=0}^{4} z^k / k!$ | bounded region, roughly to $-2.8$      |
| implicit Euler | $1 / (1 - z)$             | **L-stable**: $R(\infty) = 0$          |
| trapezoid      | $(1 + z/2)/(1 - z/2)$     | A-stable, $\lvert R(\infty)\rvert = 1$ |

**A-stable:** $\lvert R(z)\rvert \le 1$ for all $\operatorname{Re} z \le 0$. **L-stable:** also $R(z) \to 0$ as $z \to \infty$, so stiff modes are damped, not just kept bounded. **Dahlquist barriers:** no explicit method is A-stable, and A-stable linear multistep methods have order ≤ 2.

### Cost model

Total cost ≈ steps × (stages × cost of $f$ + linear algebra). The linear algebra in implicit methods is a Jacobian ($n$ extra $f$-evaluations by finite differences, fewer with sparsity colouring or AD), an LU ($n^3$ dense, much less when sparse) and solves ($n^2$ each). At large $n$ the factorisation dominates, so implicit solvers live or die by Jacobian reuse, **sparsity** and, for huge systems, Jacobian-free Newton–Krylov.

## Other lenses

- **Linear algebra.** For $y' = Ay$ the exact step is $e^{hA}$. Every one-step method approximates it with a rational function $R(hA)$: polynomials (Taylor-like) for explicit methods, Padé-like for implicit ones. Stability means $\lvert R\rvert \le 1$ on the spectrum of $hA$, and stiffness means that spectrum is spread wide.
- **Optimisation.** For a gradient flow $y' = -\nabla F(y)$, explicit Euler is gradient descent and implicit Euler is a proximal step, $y_{n+1} = \arg\min_z F(z) + \tfrac1{2h}\lVert z - y_n\rVert^2$. That is why implicit steps are unconditionally stable on convex $F$. ResNets are explicit Euler too.
- **Interpolation.** Collocation RK methods (Gauss, Radau, Lobatto) fit a polynomial through the stages. Multistep methods are polynomial interpolation through past values.
- **Geometry.** Generic integrators don't preserve invariants. For Hamiltonian systems, symplectic methods keep the energy error bounded over long times (see geometric integrators).

## Pitfalls & diagnostics

- **Tolerances:** `atol` matters wherever $\lvert y\rvert$ approaches 0. With concentrations it is the usual cause of small negative values and blow-ups. Set it per component, below the scale you care about.
- Loose tolerances on stiff problems can give a stable but **qualitatively wrong** answer. Always check convergence by tightening the tolerance 10× and comparing.
- **Discontinuities** in $f$ (dosing, switches, `max`, lookup tables) kill error control. Stop exactly at them (`tstops`, events) and restart.
- Saving densely ≠ stepping densely: use the interpolant for output instead of forcing a small $h$.
- Compare solvers with **work-precision diagrams** (error vs runtime over a tolerance sweep), not runtime at a single tolerance.
- Implicit solver slow? Check that the Jacobian is reused and sparse, and that no dense $n \times n$ matrix is being refactored every step.
- For long-time conservative dynamics (orbits, molecular dynamics), energy drift from non-symplectic methods is a method artefact, not physics.

## See also

Step control & events · DAEs · Geometric integrators · SDEs · Method of lines · ODE sensitivities · Solve Ax = b · Steady states & stability

**References:** Hairer, Nørsett & Wanner, _Solving ODEs I: Nonstiff Problems_. Hairer & Wanner, _Solving ODEs II: Stiff and DAE Problems_. Ascher & Petzold, _Computer Methods for ODEs and DAEs_.
