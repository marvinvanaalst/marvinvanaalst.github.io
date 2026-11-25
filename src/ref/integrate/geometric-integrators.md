---
title: 'Geometric integrators'
description: 'Symplectic Euler, Störmer–Verlet and splitting methods for Hamiltonian systems, energy drift vs bounded energy error, and Lie-group integrators.'
aliases:
  [
    symplectic integrator,
    Verlet,
    Stormer-Verlet,
    leapfrog,
    velocity Verlet,
    symplectic Euler,
    splitting methods,
    Strang splitting,
    Yoshida,
    Hamiltonian systems,
    energy drift,
    backward error analysis,
    Lie group integrators,
    RATTLE,
    SHAKE,
    variational integrators,
    reversible
  ]
tags: [ODE, Hamiltonian, structure preservation]
order: 5
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- For Hamiltonian (conservative) dynamics over *long times*, use a *symplectic* method: energy error stays bounded (oscillates at $`O(h^p)`$) instead of drifting.
- *Störmer–Verlet / leapfrog* is the default: 2nd order, symplectic, time-reversible, one force evaluation per step. Molecular dynamics, N-body, plasma, HMC all use it.
- Higher order: compose Verlet steps (Yoshida, Suzuki) or use symplectic partitioned RK methods; Gauss collocation is symplectic but implicit.
- *Splitting methods* (Lie, Strang) solve $`f = f_1 + f_2`$ by alternating exact or cheap sub-flows; Verlet is a Strang splitting of kinetic and potential parts.
- Symplectic integrators need *fixed step sizes*; naive adaptivity destroys the long-time benefit.

## Decision table

| Method | Order | Symplectic | Reversible | Cost per step | Use when |
| --- | --- | --- | --- | --- | --- |
| Explicit Euler | 1 | no | no | 1 force | never for conservative dynamics |
| Symplectic Euler | 1 | yes | no | 1 force | simplest structure-preserving method |
| Störmer–Verlet / leapfrog | 2 | yes | yes | 1 force | *default* for separable $`H = T(p) + V(q)`$ |
| Yoshida / Suzuki compositions | 4, 6, 8 | yes | yes | 3, 7, 15 forces | higher accuracy, smooth potentials |
| Implicit midpoint / Gauss RK | 2, $`2s`$ | yes | yes | nonlinear solve | non-separable Hamiltonians |
| RATTLE / SHAKE | 2 | yes (on constraint manifold) | yes | 1 force + projections | holonomic constraints (rigid bonds) |
| Lie-group methods (RKMK, Crouch–Grossman) | various | depends | depends | exponential maps | rotations, rigid bodies, $`SO(3)`$, $`SE(3)`$ |
| Standard RK45 | 5 | no | no | 6 forces + adaptivity | short times, high accuracy, dissipative systems |

## Variants

### Hamiltonian structure

- $`q' = \partial H/\partial p`$, $`p' = -\partial H/\partial q`$. The exact flow conserves $`H`$ and is *symplectic* (preserves phase-space area/volume: $`\Phi'^\top J\Phi' = J`$).
- A symplectic integrator exactly solves a nearby *modified Hamiltonian* $`\tilde H = H + h^p H_p + \dots`$ (backward error analysis), so $`H`$ itself oscillates but does not drift for exponentially long times.

### Symplectic Euler and Störmer–Verlet

- *Symplectic Euler:* $`p_{n+1} = p_n - h\nabla V(q_n)`$, then $`q_{n+1} = q_n + h\,M^{-1}p_{n+1}`$.
- *Velocity Verlet* (for $`H = \tfrac12 p^\top M^{-1}p + V(q)`$):
  1. $`p_{n+1/2} = p_n - \tfrac h2\nabla V(q_n)`$
  2. $`q_{n+1} = q_n + h M^{-1}p_{n+1/2}`$
  3. $`p_{n+1} = p_{n+1/2} - \tfrac h2\nabla V(q_{n+1})`$
- The force at the end of a step is reused at the start of the next, so it costs one gradient per step.
- *Leapfrog* is the same method with positions and momenta staggered by half a step.

### Splitting and composition

- *Lie–Trotter:* $`\Phi_h \approx \varphi^{[1]}_h\circ\varphi^{[2]}_h`$, order 1.
- *Strang:* $`\varphi^{[1]}_{h/2}\circ\varphi^{[2]}_h\circ\varphi^{[1]}_{h/2}`$, order 2, symmetric.
- *Composition for higher order:* $`\Psi_h = \Phi_{\gamma_1 h}\circ\Phi_{\gamma_2 h}\circ\Phi_{\gamma_1 h}`$ with $`\gamma_1 = 1/(2 - 2^{1/3})`$, $`\gamma_2 = 1 - 2\gamma_1`$ (Yoshida) turns a symmetric order-2 method into order 4. Note the negative substep.
- Splitting also works outside Hamiltonian mechanics: reaction–diffusion, Schrödinger (split-step Fourier), stiff + non-stiff parts.

### Constraints and Lie groups

- *SHAKE / RATTLE:* Verlet plus Lagrange multipliers to keep holonomic constraints (bond lengths) satisfied; RATTLE also constrains velocities and stays symplectic.
- *Lie-group integrators:* keep the state on the manifold (e.g. rotation matrices stay orthogonal) by updating through the exponential map: $`R_{n+1} = R_n\exp(h\,\widehat{\Omega})`$. Avoids renormalisation drift of quaternions or matrices (see [rotations](/ref/space/rotations)).
- *Variational integrators:* discretise the action integral rather than the equations; automatically symplectic and momentum-preserving.

## Other lenses

- *Backward error analysis.* Every integrator is the exact flow of a modified ODE. Symplectic methods have Hamiltonian modified equations; non-symplectic ones get dissipative or anti-dissipative terms, which is the energy drift.
- *Sampling.* Hamiltonian Monte Carlo relies on Verlet being volume-preserving and reversible, which keeps the Metropolis acceptance step valid (see [HMC & NUTS](/ref/bayes/hmc-nuts)).
- *Optimisation.* Momentum methods (heavy ball, Nesterov) are discretisations of damped Hamiltonian dynamics; conformal symplectic integrators preserve the damping structure.

## Pitfalls & diagnostics

- *Adaptive step sizes* break symplecticity; long-time energy behaviour degrades. Use fixed steps, or time-transformations (Sundman) designed to preserve structure.
- *Step size limit:* Verlet is only stable for $`h\omega \lt 2`$ for the fastest frequency $`\omega`$; bond vibrations limit MD time steps to \~1–2 fs (hence constraints on bonds).
- *Dissipative or driven systems* don't benefit from symplecticity; use standard high-order methods.
- *Diagnostics:* plot $`H(t) - H(0)`$ over long runs. Bounded oscillation = good; linear or random-walk drift = non-symplectic method, too large a step, or a non-smooth potential (cutoffs).
- *Non-smooth potentials* (truncated interactions) produce energy drift even with Verlet; smooth the cutoff.

## See also

- [Integrate ODEs](/ref/integrate/ode-initial-value)
- [HMC & NUTS](/ref/bayes/hmc-nuts)
- [Rotations](/ref/space/rotations)
- [Manifold optimisation](/ref/optimise/manifold-optimisation)

*References:*

- Hairer, Lubich & Wanner, _Geometric Numerical Integration_ (2nd ed.).
- Leimkuhler & Reich, _Simulating Hamiltonian Dynamics_.
- McLachlan & Quispel, "Splitting methods", Acta Numerica (2002).
