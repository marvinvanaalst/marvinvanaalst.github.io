---
title: 'ODE sensitivities'
description: 'Forward sensitivity equations vs continuous and discrete adjoints, backsolve vs interpolating vs checkpointed adjoints, and which is cheapest for how many parameters.'
aliases:
  [
    sensitivity analysis,
    forward sensitivities,
    adjoint method,
    continuous adjoint,
    discrete adjoint,
    backsolve adjoint,
    interpolating adjoint,
    quadrature adjoint,
    checkpointing,
    CVODES,
    local sensitivity,
    gradient of ODE solution
  ]
tags: [ODE, differentiation, adjoint, parameter estimation]
order: 3
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

For $y' = f(y, p, t)$ with $n$ states, $p$ parameters and a loss $L$:

- **Few parameters** ($p \lesssim 50$–$100$): **forward sensitivities** (or forward-mode AD through the solver). Cost ~$(1 + p)\times$ the ODE solve, trivially gives full trajectories of $\partial y/\partial p$.
- **Many parameters, scalar loss** (neural ODEs, large models): **adjoints**. Cost ~a constant number of extra ODE solves, independent of $p$.
- Among adjoints: **interpolating/checkpointed** adjoints are robust; **backsolve** (re-integrate $y$ backwards) uses $O(1)$ memory but is unstable for stiff or dissipative systems; **discrete adjoints** (reverse-mode AD through the solver steps) give exact gradients of the computed solution.
- Stiff problems: the adjoint ODE is just as stiff and must be solved with an implicit method too.

## Decision table

| Method                            | Cost                       | Memory                     | Gradient of                          | Use when                                     |
| --------------------------------- | -------------------------- | -------------------------- | ------------------------------------ | -------------------------------------------- |
| Finite differences                | $p$ (or $2p$) extra solves | low                        | approximate                          | quick checks, tiny $p$                       |
| Forward sensitivities             | $n(1 + p)$-dim system      | $O(np)$ per time point     | continuous solution                  | $p$ small, need $S(t)$ (e.g. FIM)            |
| Forward-mode AD through solver    | as forward sensitivities   | $O(np)$                    | discrete solution                    | $p$ small, easy to set up                    |
| Continuous adjoint, backsolve     | ~2–3 solves                | $O(n)$                     | continuous (approx.)                 | non-stiff, reversible dynamics               |
| Continuous adjoint, interpolating | ~2–3 solves                | dense forward solution     | continuous (approx.)                 | general, enough memory                       |
| Continuous adjoint, checkpointed  | ~3–4 solves                | checkpoints only           | continuous (approx.)                 | long horizons, large $n$                     |
| Discrete adjoint / reverse AD     | ~3–5× forward              | all steps (or checkpoints) | exact gradient of numerical solution | optimisation that needs consistent gradients |

## Variants

### Forward sensitivity equations

- Sensitivities $S = \partial y/\partial p \in \mathbb R^{n\times p}$ satisfy

$$
S' = \frac{\partial f}{\partial y}S + \frac{\partial f}{\partial p}, \qquad S(0) = \frac{\partial y_0}{\partial p}
$$

- Solve together with $y$. Jacobian products $J_y S$ are computed with forward-mode AD (one JVP per parameter) instead of forming $J_y$.
- **Staggered / simultaneous corrector** (CVODES): reuse the Newton matrix of the state system for the sensitivity system in implicit solvers.
- Gives the **Fisher information** directly: $\mathrm{FIM} = \sum_t S(t)^\top \Sigma^{-1} S(t)$ (see [experimental design](/ref/freq/experimental-design)).

### Continuous adjoint

For a loss $L = \int_0^T g(y, p, t)\,dt$ (or a sum over data points):

1. Solve the forward ODE for $y(t)$ on $[0, T]$.
2. Solve the **adjoint ODE backwards** from $\lambda(T) = 0$ (plus jumps $\partial g_i/\partial y$ at data points):

$$
\lambda' = -\Big(\frac{\partial f}{\partial y}\Big)^\top\lambda - \Big(\frac{\partial g}{\partial y}\Big)^\top
$$

3. Accumulate the gradient by quadrature: $\dfrac{dL}{dp} = \lambda(0)^\top\dfrac{\partial y_0}{\partial p} + \displaystyle\int_0^T \Big(\lambda^\top\frac{\partial f}{\partial p} + \frac{\partial g}{\partial p}\Big)dt$.

- The products $\lambda^\top J_y$ and $\lambda^\top J_p$ are VJPs from reverse-mode AD of $f$.
- **How to get $y(t)$ during the backward pass:**
  - **Backsolve:** integrate $y$ backwards alongside $\lambda$. $O(n)$ memory, but backward integration of a dissipative/stiff forward system is unstable.
  - **Interpolating:** store the dense forward solution and interpolate.
  - **Checkpointing:** store $y$ at checkpoints, re-solve forward between them during the backward pass.

### Discrete adjoint

- Apply reverse-mode AD to the sequence of solver steps (including Newton iterations or using implicit differentiation for them).
- Exact gradient of the discretised loss, consistent with what an optimiser sees; continuous adjoints give gradients of the exact solution, which differ by $O(\text{tol})$.
- Memory: every step's state and stage values, unless checkpointed.

### Cost comparison

|                    | Forward sensitivities                   | Adjoint                                 |
| ------------------ | --------------------------------------- | --------------------------------------- |
| Scales with        | number of parameters $p$                | number of outputs (1 for a scalar loss) |
| Gives              | $\partial y(t)/\partial p$ at all times | $dL/dp$ only                            |
| Typical break-even | —                                       | $p \approx 50$–$100$                    |

## Other lenses

- **AD.** Forward sensitivities = forward-mode AD of the continuous problem; adjoints = reverse mode. The continuous vs discrete distinction is "differentiate then discretise" vs "discretise then differentiate" (see [automatic differentiation](/ref/diff/automatic-differentiation)).
- **Optimal control.** The adjoint equation is the costate equation of Pontryagin's principle; gradient-based optimal control and parameter estimation use the same machinery (see [optimal control & MPC](/ref/optimise/optimal-control-mpc)).
- **Statistics.** Sensitivities scaled by parameter values and outputs ($\partial\log y/\partial\log p$) are the local sensitivity coefficients of metabolic control analysis; their Gram matrix is the FIM, whose spectrum reveals sloppy and non-identifiable directions.

## Pitfalls & diagnostics

- **Tolerances:** sensitivities need error control too (include them in the error norm), or the gradient is much less accurate than the solution.
- **Backsolve instability:** symptoms are gradients that change wildly with tolerance. Switch to interpolating or checkpointed adjoints.
- **Discontinuities and events** need jump conditions for $S$ and $\lambda$ at event times; most solvers handle this only if events are declared.
- **Chaotic systems:** sensitivities grow exponentially with the horizon; long-time gradients are meaningless (use shorter windows, multiple shooting or shadowing methods).
- **Check gradients** against finite differences on a short horizon and a few parameters before trusting them in an optimiser.

## See also

- [Automatic differentiation](/ref/diff/automatic-differentiation)
- [Integrate ODEs](/ref/integrate/ode-initial-value)
- [Fitting ODE models](/ref/optimise/fitting-odes)
- [Neural ODEs & UDEs](/ref/optimise/neural-odes-udes)
- [Global sensitivity analysis](/ref/dynamics/global-sensitivity)

**References:**

- Cao, Li, Petzold & Serban, "Adjoint sensitivity analysis for differential-algebraic equations", SIAM J. Sci. Comput. (2003).
- Hindmarsh et al., "SUNDIALS: suite of nonlinear and differential/algebraic equation solvers", ACM TOMS (2005).
- Ma et al., "A comparison of automatic differentiation and continuous sensitivity analysis for derivatives of differential equation solutions" (2021).
