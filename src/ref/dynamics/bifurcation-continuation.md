---
title: 'Bifurcation analysis & continuation'
description: 'Natural vs pseudo-arclength continuation, detecting folds, Hopf points and branch points, and two-parameter continuation of bifurcation curves.'
aliases:
  [
    bifurcation,
    continuation,
    numerical continuation,
    pseudo-arclength,
    natural parameter continuation,
    fold,
    saddle-node,
    limit point,
    Hopf bifurcation,
    branch point,
    transcritical,
    pitchfork,
    period doubling,
    codimension two,
    cusp,
    Bogdanov-Takens,
    AUTO,
    MatCont,
    BifurcationKit,
    bistability,
    hysteresis
  ]
tags: [dynamical systems, bifurcation, continuation]
order: 2
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- Trace a branch of steady states (or periodic orbits) $f(x, p) = 0$ as a parameter $p$ varies.
- **Natural parameter continuation** (step $p$, re-solve with the previous solution as guess) fails at folds where the branch turns back.
- **Pseudo-arclength continuation** parametrises by arclength along the curve: passes through folds and is the standard (AUTO, MatCont, BifurcationKit).
- **Detect bifurcations** with test functions along the branch: fold ($\det J$ changes sign or the $p$-component of the tangent vanishes), Hopf (a complex pair crosses the imaginary axis), branch points.
- **Two-parameter continuation** of the detected points draws bifurcation diagrams (e.g. bistability regions bounded by fold curves meeting at a cusp).

## Decision table

| Method / test                | What it does                                        | Use when                                          |
| ---------------------------- | --------------------------------------------------- | ------------------------------------------------- |
| Parameter sweep + simulation | run to steady state for each $p$                    | quick look; misses unstable branches              |
| Natural continuation         | step $p$, Newton from previous solution             | monotone branches, no folds                       |
| Pseudo-arclength             | predictor (tangent) + corrector on augmented system | **default**; folds, hysteresis loops              |
| Moore–Penrose continuation   | corrector orthogonal to the tangent                 | similar to pseudo-arclength, robust               |
| Fold test                    | sign change of $\det J$ or tangent's $p$-component  | saddle-node points, bistability boundaries        |
| Hopf test                    | bialternate product / eigenvalue pair crossing      | onset of oscillations                             |
| Branch point test            | sign change of the augmented Jacobian's determinant | symmetry-breaking, transcritical/pitchfork        |
| Periodic orbit continuation  | collocation or shooting for cycles                  | follow oscillations, period doubling, homoclinics |

| Bifurcation (codim 1)     | Generic signature                              | Consequence                                                              |
| ------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------ |
| Fold (saddle-node)        | real eigenvalue crosses 0, branch turns back   | creation/annihilation of steady states; hysteresis                       |
| Hopf                      | complex pair crosses the imaginary axis        | birth of a limit cycle (super- or subcritical)                           |
| Transcritical / pitchfork | real eigenvalue crosses 0 at a branch crossing | exchange of stability; symmetry breaking (non-generic without structure) |
| Period doubling           | Floquet multiplier crosses $-1$                | route to chaos                                                           |
| Homoclinic                | cycle collides with a saddle, period → ∞       | oscillations disappear with diverging period                             |

## Variants

### Pseudo-arclength continuation

Given a point $(x_0, p_0)$ and unit tangent $(\dot x_0, \dot p_0)$:

1. **Predictor:** $(x, p) = (x_0, p_0) + \Delta s\,(\dot x_0, \dot p_0)$.
2. **Corrector:** Newton on the augmented system $f(x, p) = 0$, $\dot x_0^\top(x - x_0) + \dot p_0(p - p_0) - \Delta s = 0$. Its Jacobian $\begin{pmatrix} f_x & f_p \\ \dot x_0^\top & \dot p_0\end{pmatrix}$ stays non-singular at simple folds.
3. **New tangent:** null vector of $[f_x\ f_p]$, oriented consistently with the previous one.
4. **Step control:** grow $\Delta s$ when Newton converges in few iterations, shrink on failure.

### Detecting and locating bifurcations

- Monitor test functions between consecutive points; on a sign change, locate the point by bisection or by solving an extended (defining) system.
- **Fold:** $\dot p = 0$ along the tangent, or $\det f_x = 0$.
- **Hopf:** eigenvalues $\pm i\omega$; test function via the bialternate product or tracking the rightmost eigenvalues. First Lyapunov coefficient decides super- (stable cycle born) vs subcritical (unstable cycle, hard onset).
- **Branch points:** switch branches using the second tangent direction (from the null space at the branch point).

### Two-parameter continuation

- Continue fold or Hopf points in $(p_1, p_2)$ using minimally extended defining systems (e.g. $f = 0$, $f_x v = 0$, $\lVert v\rVert = 1$ for folds).
- Codimension-2 points organise the diagram: **cusp** (two fold curves meet: bistability region appears), **Bogdanov–Takens** (fold and Hopf meet), **Bautin** (Hopf changes criticality).

### Periodic orbits

- Continue limit cycles from a Hopf point with orthogonal collocation (AUTO) or shooting; detect fold of cycles, period doubling (multiplier $-1$), Neimark–Sacker (torus).
- See [periodic orbits & chaos](/ref/dynamics/periodic-orbits-chaos).

## Other lenses

- **Root finding.** Continuation is homotopy for parameter-dependent equations; the same predictor-corrector solves hard nonlinear systems by deforming an easy one (see [nonlinear equations](/ref/solve/nonlinear-roots)).
- **Biology.** Folds give switches and hysteresis (bistable genetic switches, cell fate); Hopf bifurcations give oscillators (circadian clocks, calcium oscillations, glycolytic oscillations). Bifurcation diagrams are the qualitative fingerprint of a model.
- **Algebraic geometry.** For polynomial models, steady states and bifurcation loci are solutions of polynomial systems; discriminants of the steady-state equations describe fold curves exactly.

## Pitfalls & diagnostics

- **Simulating instead of continuing** misses unstable branches and gets hysteresis wrong; always continue.
- **Step size too large** jumps between branches near folds or crossings; check that consecutive points are close and that eigenvalues change continuously.
- **Scale parameters and states** (log-parameters often help); badly scaled augmented systems make Newton fail spuriously.
- **Conservation laws** must be removed first (singular $f_x$ everywhere otherwise).
- **Hopf detection with large systems:** tracking all eigenvalues is expensive; use test functions or shift-invert Arnoldi for the rightmost ones.

## See also

- [Steady states & stability](/ref/dynamics/steady-states-stability)
- [Periodic orbits & chaos](/ref/dynamics/periodic-orbits-chaos)
- [Nonlinear equations](/ref/solve/nonlinear-roots)
- [Reaction network theory](/ref/dynamics/reaction-network-theory)

**References:**

- Kuznetsov, _Elements of Applied Bifurcation Theory_ (3rd ed.), ch. 10.
- Allgower & Georg, _Introduction to Numerical Continuation Methods_.
- Dhooge, Govaerts & Kuznetsov, "MATCONT: a MATLAB package for numerical bifurcation analysis of ODEs", ACM TOMS (2003).
