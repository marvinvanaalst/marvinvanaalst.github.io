---
title: 'Optimal control & MPC'
description: "Pontryagin's principle, direct vs indirect methods, direct transcription and multiple shooting, LQR, and model predictive control."
aliases:
  [
    optimal control,
    Pontryagin maximum principle,
    Hamiltonian,
    costate,
    adjoint,
    indirect methods,
    direct methods,
    direct transcription,
    direct collocation,
    direct multiple shooting,
    LQR,
    Riccati equation,
    model predictive control,
    MPC,
    receding horizon,
    dynamic programming,
    Hamilton-Jacobi-Bellman,
    bang-bang control,
    CasADi,
    acados
  ]
tags: [optimal control, optimisation, ODE, control]
order: 9
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- Problem: choose $u(t)$ to minimise $\int_0^T\ell(x, u)\,dt + \Phi(x(T))$ subject to $x' = f(x, u)$, constraints on $x$ and $u$.
- **Direct methods** (discretise, then optimise) are the practical default: **direct multiple shooting** or **direct collocation**, giving a sparse NLP for IPOPT/SQP. Path constraints are easy.
- **Indirect methods** (Pontryagin: optimise, then discretise) give a two-point boundary value problem; very accurate but need good guesses for the costates and the switching structure.
- **Linear dynamics + quadratic cost:** LQR, solved by a Riccati equation; the closed-form feedback is the baseline for everything else.
- **MPC:** solve a finite-horizon problem repeatedly online, apply the first control, shift the horizon. Handles constraints; needs fast solvers (warm starts, real-time iterations).

## Decision table

| Method                    | Formulation                           | Strengths                          | Weaknesses                                        | Use when                                         |
| ------------------------- | ------------------------------------- | ---------------------------------- | ------------------------------------------------- | ------------------------------------------------ |
| Indirect (Pontryagin BVP) | state + costate BVP                   | high accuracy, insight (switching) | needs costate guesses, structure known in advance | aerospace trajectories, analysis                 |
| Direct single shooting    | optimise controls; integrate dynamics | small NLP, easy                    | unstable dynamics, poor convergence               | short horizons, stable systems                   |
| Direct multiple shooting  | controls + segment start states       | robust, parallel, uses ODE solvers | larger NLP                                        | **default** for ODE models (CasADi, acados)      |
| Direct collocation        | states and controls on a mesh         | very robust, path constraints easy | large NLP, mesh refinement                        | complex constraints, unstable dynamics           |
| Dynamic programming / HJB | value function on a grid              | global optimum, feedback law       | curse of dimensionality ($d \lesssim 4$–6)        | low-dimensional state                            |
| LQR                       | Riccati equation                      | closed form, fast                  | linear, unconstrained                             | stabilisation, local control around a trajectory |
| MPC (linear, QP)          | QP per sampling instant               | constraints, fast (μs–ms)          | needs a linear model                              | process control, robotics                        |
| Nonlinear MPC             | NLP per sampling instant              | full nonlinear model               | computation time                                  | when linear MPC isn't enough                     |

## Variants

### Pontryagin's maximum principle

- Hamiltonian $H = \ell(x, u) + \lambda^\top f(x, u)$.
- Necessary conditions:
  - state: $x' = \partial H/\partial\lambda$
  - costate: $\lambda' = -\partial H/\partial x$, $\lambda(T) = \partial\Phi/\partial x(T)$
  - optimality: $u^*(t) = \arg\min_u H(x, u, \lambda)$ over admissible $u$
- Control-affine problems with bounded controls often give **bang-bang** controls (switching between bounds) or singular arcs.
- Solve the resulting BVP by (multiple) shooting on the costate initial values, or collocation.

### Direct multiple shooting

1. Discretise $u$ (piecewise constant on $N$ intervals) and introduce state variables $s_k$ at interval starts.
2. On each interval, integrate the ODE from $s_k$ with $u_k$ (ODE solver with sensitivities).
3. NLP: minimise the summed cost over $(s_k, u_k)$ s.t. continuity $x(t_{k+1}; s_k, u_k) = s_{k+1}$ and path constraints at the nodes.
4. Solve with SQP or interior point, exploiting the block-banded structure.

### Direct collocation

- Approximate states by polynomials on each mesh interval; enforce the dynamics at collocation points (trapezoidal, Hermite–Simpson, Radau/Legendre pseudospectral).
- All states and controls at all points are decision variables: big, very sparse NLP. Mesh refinement where the solution changes quickly.

### LQR

- For $x' = Ax + Bu$, cost $\int(x^\top Qx + u^\top Ru)\,dt$: optimal feedback $u = -R^{-1}B^\top Px$ with $P$ from the algebraic Riccati equation $A^\top P + PA - PBR^{-1}B^\top P + Q = 0$.
- Discrete-time version for sampled systems; iterative LQR / DDP for nonlinear trajectory optimisation by repeated linearisation.

### Model predictive control

1. At time $t$, measure or estimate the state (Kalman filter / MHE).
2. Solve the optimal control problem over horizon $[t, t + T_h]$ with constraints.
3. Apply the first control move; shift the horizon; repeat.

- **Real-time iteration:** one SQP iteration per sampling time, warm-started from the shifted previous solution.
- Stability via terminal costs/constraints; robustness via tube MPC or scenario trees.

## Other lenses

- **Adjoints.** The costate is the adjoint of the dynamics; gradient-based direct methods compute the same quantities via adjoint sensitivities (see [ODE sensitivities](/ref/diff/ode-sensitivities)).
- **Reinforcement learning.** RL solves the same problem with unknown dynamics by sampling; dynamic programming and the HJB equation are the common foundation. Model-based RL ≈ learned model + MPC.
- **Estimation.** The Kalman filter is the dual of LQR (Riccati equations with transposed matrices); moving-horizon estimation is the estimation counterpart of MPC (see [Kalman filters](/ref/signals/kalman-filters)).
- **Parameter estimation.** Fitting ODEs is optimal control with constant "controls" (the parameters); the same shooting/collocation tools apply (see [fitting ODE models](/ref/optimise/fitting-odes)).

## Pitfalls & diagnostics

- **Scaling** of states, controls and time is critical for NLP convergence.
- **Discretisation artefacts:** chattering controls from coarse meshes or too fine control grids; check convergence under mesh refinement.
- **Singular arcs** are hard for both direct and indirect methods; add small regularisation of control effort and monitor.
- **Initial guesses:** use a simulated trajectory with a simple feedback controller or solve an easier version first (homotopy in constraints/costs).
- **MPC timing:** worst-case solve time must fit the sampling period; plan for solver failures (fallback controls).

## See also

- [Constrained optimisation](/ref/optimise/constrained)
- [Fitting ODE models](/ref/optimise/fitting-odes)
- [ODE sensitivities](/ref/diff/ode-sensitivities)
- [Kalman filters](/ref/signals/kalman-filters)
- [Periodic orbits & chaos](/ref/dynamics/periodic-orbits-chaos) (BVPs)

**References:**

- Betts, _Practical Methods for Optimal Control and Estimation Using Nonlinear Programming_ (2nd ed.).
- Rawlings, Mayne & Diehl, _Model Predictive Control: Theory, Computation, and Design_ (2nd ed.).
- Liberzon, _Calculus of Variations and Optimal Control Theory_.
