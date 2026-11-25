---
title: 'Differential-algebraic equations'
description: 'Index, mass-matrix form, consistent initialisation, index reduction and the solvers (BDF, Radau) that handle DAEs.'
aliases:
  [
    DAE,
    index,
    differentiation index,
    mass matrix,
    semi-explicit DAE,
    consistent initial conditions,
    index reduction,
    Pantelides algorithm,
    dummy derivatives,
    IDA,
    DASSL,
    Radau,
    algebraic constraints,
    conservation laws,
    acausal modelling,
    Modelica
  ]
tags: [DAE, ODE, constraints]
order: 4
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- A DAE mixes differential and algebraic equations: $`F(t, y, y') = 0`$, typically semi-explicit $`x' = f(x, z)`$, $`0 = g(x, z)`$.
- *Index 1* ($`\partial g/\partial z`$ invertible): solve directly with BDF (DASSL/IDA) or Radau IIA, as easily as a stiff ODE. Most chemical, metabolic and circuit models with fast equilibria are index 1.
- *Higher index* (constrained mechanics, ideal electrical networks): reduce the index symbolically (Pantelides + dummy derivatives, as in Modelica tools) before solving, or use Radau for index ≤ 3 with care.
- *Consistent initial conditions* are part of the problem: algebraic variables (and hidden constraints) must satisfy the constraints at $`t_0`$.
- Mass-matrix form $`My' = f(y)`$ with singular $`M`$ is the common solver interface.

## Decision table

| Problem | Index | Solver | Notes |
| --- | --- | --- | --- |
| ODE with quasi-steady-state equations | 1 | BDF (IDA), Radau IIA, Rosenbrock (Rodas) | as cheap as a stiff ODE |
| Conservation laws written as constraints | 1 | same; or eliminate them | elimination reduces size and removes drift |
| Circuits (MNA formulation) | 1–2 | BDF, Radau | index 2 for some topologies |
| Constrained mechanics (position level) | 3 | index reduction to 1, or Radau with scaled errors | GGL / Baumgarte stabilisation |
| Acausal / equation-based models | any | structural analysis (Pantelides) + index-1 solver | Modelica, ModelingToolkit |
| Explicit RK | — | not applicable | explicit methods cannot handle algebraic equations |

## Variants

### Index

- *Differentiation index:* number of times (some of) the equations must be differentiated to obtain an explicit ODE for all variables.
  - Index 1: $`0 = g(x, z)`$ with invertible $`g_z`$ (one differentiation gives $`z'`$).
  - Index 2: $`g`$ does not depend on $`z`$, but $`g_x f_z`$ is invertible (e.g. incompressibility, holonomic constraints at velocity level).
  - Index 3: mechanical systems with position constraints and Lagrange multipliers.
- *Higher index is harder:* errors in derivatives of the inputs appear in the solution, hidden constraints arise, and numerical methods lose order.

### Index-1 solvers

- *BDF* (DASSL, IDA): replace $`y'`$ by the BDF formula and solve $`F(t, y, \text{BDF}(y)) = 0`$ with Newton. Fully implicit form $`F(t, y, y') = 0`$.
- *Radau IIA:* works on $`My' = f(y)`$; order 5 for index-1 algebraic components; also usable for index 2 and 3 with order reduction.
- *Rosenbrock (Rodas):* linearly implicit, efficient for small/medium index-1 mass-matrix DAEs.
- Error control on algebraic variables of higher-index problems must be scaled ($`h^{\text{index}-1}`$) or excluded.

### Consistent initialisation

- Given the differential states $`x_0`$, solve $`g(x_0, z_0) = 0`$ for $`z_0`$ (Newton). With $`F(t, y, y') = 0`$, also compute consistent $`y'_0`$.
- For higher index, hidden constraints (derivatives of $`g`$) also have to hold.
- IDA's `IDACalcIC`, Brown–Hindmarsh–Petzold method; or initialise by a short pseudo-transient integration.

### Index reduction

- *Pantelides algorithm:* structural (graph matching) analysis that finds which equations to differentiate.
- *Dummy derivatives* (Mattsson–Söderlind): after differentiating, replace some derivatives by new algebraic variables so the system is index 1 again without overdetermination.
- *Drift:* solving the differentiated system alone only enforces the derivative of the constraint, so the original constraint drifts. Stabilise by projection, Baumgarte stabilisation, or Gear–Gupta–Leimkuhler (GGL) formulation.

### Eliminating constraints instead

- *Conservation laws* (e.g. total enzyme, moiety conservation) can be eliminated: keep the independent species, compute dependent ones algebraically. Smaller system, no constraint drift, and a non-singular Jacobian at steady state (see [stoichiometric analysis](/ref/dynamics/stoichiometric-analysis)).
- *Quasi-steady-state equations* can be solved analytically for fast species when possible, turning the DAE into a reduced ODE (see [timescale reduction](/ref/dynamics/timescale-reduction)).

## Other lenses

- *Singular perturbation.* An index-1 DAE is the $`\varepsilon \to 0`$ limit of the stiff ODE $`\varepsilon z' = g(x, z)`$; the stiff solver and the DAE solver are solving nearly the same problem. Tikhonov's theorem says when this limit is valid.
- *Optimisation.* Constrained mechanics is a saddle-point (KKT-like) system at every time step; Lagrange multipliers are the constraint forces.
- *Modelling languages.* Acausal (equation-based) modelling produces DAEs naturally; the compiler's structural analysis, tearing and index reduction are graph algorithms (matchings, SCCs, see [graph algorithms](/ref/graphs/graph-algorithms)).

## Pitfalls & diagnostics

- *Inconsistent initial conditions* cause failures at the first step or silent jumps in algebraic variables. Always initialise explicitly.
- *Singular Jacobian of algebraic equations* ($`g_z`$ singular somewhere along the trajectory) means the index changes locally: impasse points, the model breaks down.
- *Hidden higher index:* a model that looks index 1 may be index 2 (e.g. an algebraic equation that doesn't involve the algebraic variable it should determine). Check the structure.
- *Error control of algebraic variables* in index-2/3 problems causes tiny steps or failure; exclude them from the error norm or scale.
- *Conservation constraints plus full ODE* together (redundant equations) give a singular system; use either the ODE or the reduced DAE, not both.

## See also

- [Integrate ODEs](/ref/integrate/ode-initial-value)
- [Timescale separation & model reduction](/ref/dynamics/timescale-reduction)
- [Stoichiometric analysis](/ref/dynamics/stoichiometric-analysis)
- [Nonlinear equations](/ref/solve/nonlinear-roots)

*References:*

- Brenan, Campbell & Petzold, _Numerical Solution of Initial-Value Problems in Differential-Algebraic Equations_.
- Hairer & Wanner, _Solving ODEs II_, ch. VI–VII.
- Ascher & Petzold, _Computer Methods for ODEs and DAEs_, ch. 9–10.
