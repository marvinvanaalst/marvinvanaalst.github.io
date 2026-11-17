---
title: 'Step control & events'
description: 'Error estimation, tolerance scaling, I and PI step-size controllers, initial step selection, dense output and event location in ODE solvers.'
aliases:
  [
    adaptive step size,
    error control,
    embedded methods,
    local error,
    PI controller,
    step size controller,
    atol,
    rtol,
    initial step size,
    dense output,
    continuous extension,
    event detection,
    root finding events,
    callbacks,
    tstops,
    discontinuities
  ]
tags: [ODE, adaptivity, events]
order: 3
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- Adaptive solvers estimate the **local error** each step (embedded pair or multistep difference), scale it by `atol + rtol·|y|` per component, and accept the step if the scaled RMS norm is ≤ 1.
- New step: $h_{\text{new}} = h\cdot\mathrm{safety}\cdot e^{-1/(q+1)}$ (I controller); **PI controllers** smooth the sequence and reduce rejections, especially on stiff problems.
- Tolerances control **local** error per step, not global error. Global error is roughly proportional to the tolerance; verify by tightening it.
- **Dense output** gives the solution between steps at the method's order; use it for output and events instead of forcing small steps.
- **Events:** monitor sign changes of $g(t, y)$, locate the root on the dense output, then stop or modify the state. Known discontinuity times go in `tstops`.

## Decision table

| Component            | Options                                                                                | Default / recommendation                     |
| -------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------- |
| Error estimate       | embedded RK pair; multistep predictor-corrector difference; Richardson (step doubling) | embedded pair (free)                         |
| Error norm           | RMS (scaled $L^2$), max norm                                                           | RMS                                          |
| Scaling              | $\mathrm{atol}_i + \mathrm{rtol}\cdot\max(\lvert y_i\rvert, \lvert\hat y_i\rvert)$     | per-component atol for variables near zero   |
| Controller           | I (elementary), PI, PID / digital filters (Söderlind)                                  | PI for most; I for explicit nonstiff is fine |
| Growth/shrink limits | factor in $[0.2, 5]$ or $[0.1, 10]$                                                    | limit growth after rejections                |
| Initial step         | user guess; Hairer's algorithm                                                         | automatic (one or two extra $f$ evaluations) |
| Output               | step points, dense interpolant, fixed `saveat`                                         | interpolant                                  |
| Discontinuities      | events (unknown times), `tstops` (known times)                                         | always declare them                          |

## Variants

### Error estimation and scaling

- **Embedded pair:** two solutions of orders $p$ and $q = p - 1$ from the same stages; $\mathrm{err} = y_{n+1} - \hat y_{n+1}$.
- **Local extrapolation:** propagate the higher-order solution (DP5, Tsit5 do this) even though the estimate is for the lower one.
- **Scaled norm:** $e = \sqrt{\tfrac1n\sum_i\big(\mathrm{err}_i / (\mathrm{atol}*i + \mathrm{rtol}\cdot\max(\lvert y*{n,i}\rvert, \lvert y_{n+1,i}\rvert))\big)^2}$.
  - `rtol` sets the number of correct digits for large components.
  - `atol` sets the noise floor below which a component is not resolved; it dominates when $\lvert y_i\rvert \lt \mathrm{atol}/\mathrm{rtol}$.

### Step-size controllers

- **I controller (elementary):** $h_{n+1} = h_n\cdot\mathrm{fac}\cdot e_n^{-1/(q+1)}$ with safety $\mathrm{fac} \approx 0.8$–$0.9$.
- **PI controller:** $h_{n+1} = h_n\, e_n^{-\beta_1/(q+1)}\,e_{n-1}^{\beta_2/(q+1)}$, e.g. $\beta_1 = 0.7$, $\beta_2 = 0.4$ (Gustafsson). Damps oscillating step sizes; fewer rejections when the step is stability-limited.
- **After a rejection:** don't increase $h$ on the next step.
- **Implicit methods:** step changes also cost Jacobian/factorisation updates; controllers often keep $h$ fixed when the change would be small (ratio within ~$[0.8, 1.2]$) to reuse the factorisation.

### Initial step size

Hairer's heuristic:

1. $d_0 = \lVert y_0\rVert$, $d_1 = \lVert f(t_0, y_0)\rVert$ (scaled norms); $h_0 = 0.01\,d_0/d_1$.
2. Explicit Euler step to $y_1 = y_0 + h_0 f_0$, estimate $d_2 = \lVert f(t_0 + h_0, y_1) - f_0\rVert / h_0$.
3. $h_1 = \big(0.01/\max(d_1, d_2)\big)^{1/(p+1)}$, take $h = \min(100h_0, h_1)$.

### Dense output

- Continuous extension: a polynomial $u(\theta)$, $\theta \in [0, 1]$, through the step using the stages (plus a few extra evaluations for high-order methods such as Verner's).
- Multistep methods interpolate their history polynomial for free.
- Use for: output at arbitrary times, event location, delay-equation history lookup.

### Events

1. After each accepted step, evaluate event functions $g_j(t, y)$ at both ends.
2. On a sign change (in the requested direction), locate the root on the dense output with a bracketing method (Brent/Illinois) to tolerance.
3. Apply the action: terminate, or modify $y$ / parameters (an impulse), then **restart** the integrator (multistep methods restart at order 1, FSAL caches are invalidated).

- **Missed events:** two sign changes inside one step cancel out. Limit the maximum step or check intermediate points of the interpolant.
- **Chattering:** repeated events at the same point (sliding modes); needs hysteresis or a Filippov treatment.

## Other lenses

- **Control theory.** Step-size selection is a feedback loop: the error estimate is the measured output, $\log h$ the control input. PI/PID controllers come from viewing it as a discrete-time control system (Gustafsson, Söderlind).
- **Root finding.** Events are 1D root finding in time along the solution curve (see [nonlinear equations](/ref/solve/nonlinear-roots)).
- **Hybrid systems.** ODEs with events are hybrid automata: continuous flows plus discrete jumps. Pharmacokinetic dosing, cell division and switching models all live here.

## Pitfalls & diagnostics

- **Tolerance proportionality check:** run at rtol $10^{-6}$ and $10^{-8}$; if the results differ more than ~100× the tolerance ratio suggests, something (event handling, discontinuity, stiffness) is wrong.
- **Discontinuous right-hand sides** without events or `tstops` cause step-size collapse and many rejections exactly where the switch happens.
- **Too-loose atol** for small components (concentrations near zero): they are effectively not controlled, may go negative and destabilise the rest.
- **Forcing output points** by stepping exactly to each `saveat` time degrades efficiency; prefer interpolation.
- **Counts to watch:** number of accepted/rejected steps, $f$ evaluations, Jacobians, LU factorisations. Many rejections → stiffness, discontinuities or a bad controller.

## See also

- [Integrate ODEs](/ref/integrate/ode-initial-value)
- [DAEs](/ref/integrate/daes)
- [Nonlinear equations](/ref/solve/nonlinear-roots)
- [Stochastic simulation](/ref/integrate/stochastic-simulation)

**References:**

- Hairer, Nørsett & Wanner, _Solving ODEs I_, sections II.4–II.6.
- Söderlind, "Digital filters in adaptive time-stepping", ACM TOMS (2003).
- Shampine & Thompson, "Event location for ordinary differential equations", Comput. Math. Appl. (2000).
