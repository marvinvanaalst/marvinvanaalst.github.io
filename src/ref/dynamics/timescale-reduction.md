---
title: 'Timescale separation & model reduction'
description: 'QSSA and rapid equilibrium, Tikhonov theory and slow manifolds, POD, balanced truncation and dynamic mode decomposition.'
aliases:
  [
    model reduction,
    model order reduction,
    QSSA,
    quasi-steady-state approximation,
    rapid equilibrium,
    Michaelis-Menten derivation,
    Tikhonov theorem,
    singular perturbation,
    slow manifold,
    CSP,
    ILDM,
    POD,
    proper orthogonal decomposition,
    balanced truncation,
    Gramians,
    DMD,
    dynamic mode decomposition,
    Koopman,
    reduced basis,
    lumping
  ]
tags: [model reduction, timescales, dynamical systems]
order: 3
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Mechanistic reduction* (keeps interpretability): QSSA for fast intermediates, rapid-equilibrium for fast reversible reactions, lumping of species. Justified by Tikhonov's theorem when timescales are well separated.
- *Data/projection-based reduction* (keeps accuracy, loses mechanism): *POD/Galerkin* for nonlinear PDE-like systems, *balanced truncation* for linear input-output systems (with error bounds), *DMD* for extracting modes and frequencies from data.
- Always check the reduced model against the full model on the relevant inputs; reductions are valid only in the regime they were derived for.

## Decision table

| Method | Input needed | Preserves | Error control | Use when |
| --- | --- | --- | --- | --- |
| QSSA | model equations | mechanism (reduced) | asymptotic ($`\varepsilon \to 0`$) | fast intermediates (enzyme complexes) |
| Rapid equilibrium | model equations | mechanism | asymptotic | fast reversible binding |
| Tikhonov / slow manifold (CSP, ILDM) | equations + Jacobian | slow dynamics | asymptotic, can be refined | combustion, large kinetic networks |
| Lumping | equations / structure | aggregate species | exact for exact lumpability | many similar species (isomers, states) |
| POD + Galerkin | snapshots of solutions | energy-optimal subspace | empirical | large nonlinear systems, PDEs |
| Balanced truncation | linear system $`(A, B, C)`$ | input-output behaviour | $`\le 2\sum`$ discarded Hankel singular values | linear(ised) control systems |
| DMD | time series data | linear dynamics in a subspace | empirical | modes, frequencies, growth rates from data |
| Reduced basis (greedy) | parametrised PDE solves | parameter dependence | a-posteriori error estimators | many queries over parameters (UQ, optimisation) |

## Variants

### QSSA and rapid equilibrium

- *QSSA:* set $`\dot c \approx 0`$ for a fast-relaxing intermediate $`c`$ and solve algebraically. For $`E + S \rightleftharpoons C \to E + P`$ this gives Michaelis–Menten $`v = V_{\max}S/(K_M + S)`$ with $`K_M = (k_{-1} + k_2)/k_1`$.
- Validity (Segel–Slemrod): $`E_0 \ll S_0 + K_M`$; the standard condition "$`E_0 \ll S_0`$" is sufficient but not necessary. Total QSSA extends validity to high enzyme concentrations.
- *Rapid equilibrium:* fast reversible reactions are at equilibrium, $`C = E\,S/K_d`$; the slow variables are the totals (lumped pools) that the fast reactions conserve. Cleaner than QSSA when a reaction (not a species) is fast.

### Tikhonov theory and slow manifolds

- Singularly perturbed system: $`x' = f(x, z)`$, $`\varepsilon z' = g(x, z)`$. If the fast subsystem has a stable equilibrium $`z = h(x)`$ (all eigenvalues of $`g_z`$ negative), then as $`\varepsilon \to 0`$ solutions converge to the reduced system $`x' = f(x, h(x))`$ after an initial layer.
- *Fenichel theory:* for small $`\varepsilon \gt 0`$ there is an invariant slow manifold $`\varepsilon`$-close to $`z = h(x)`$.
- *Without explicit $`\varepsilon`$:* identify fast and slow directions from Jacobian eigenvalues (computational singular perturbation, intrinsic low-dimensional manifolds) or by nondimensionalisation.

### Proper orthogonal decomposition

1. Collect snapshots $`X = [x(t_1), \dots, x(t_m)]`$ (simulations over representative inputs/parameters).
2. SVD $`X = U\Sigma V^\top`$; keep the first $`r`$ left singular vectors $`U_r`$ (energy captured $`\sum_{i \le r}\sigma_i^2/\sum\sigma_i^2`$).
3. Galerkin projection: $`x \approx U_r a`$, $`a' = U_r^\top f(U_r a)`$.
4. For nonlinear $`f`$, evaluating $`f(U_r a)`$ still costs $`O(n)`$: use hyper-reduction (DEIM, empirical interpolation) to sample only a few components.

### Balanced truncation

- For $`x' = Ax + Bu`$, $`y = Cx`$: controllability Gramian $`P`$ ($`AP + PA^\top + BB^\top = 0`$) and observability Gramian $`Q`$ ($`A^\top Q + QA + C^\top C = 0`$).
- Transform to coordinates where $`P = Q = \Sigma`$ (Hankel singular values), truncate states with small $`\sigma_i`$.
- Guarantees stability and an $`H_\infty`$ error bound $`\le 2\sum_{i \gt r}\sigma_i`$. Large-scale: low-rank Lyapunov solvers (ADI, Krylov).

### Dynamic mode decomposition

1. Snapshot pairs $`X = [x_1, \dots, x_{m-1}]`$, $`X' = [x_2, \dots, x_m]`$; seek $`A`$ with $`X' \approx AX`$.
2. Truncated SVD $`X = U_r\Sigma_rV_r^\top`$, reduced operator $`\tilde A = U_r^\top X'V_r\Sigma_r^{-1}`$.
3. Eigenvalues of $`\tilde A`$ give frequencies and growth rates; modes $`\Phi = X'V_r\Sigma_r^{-1}W`$.

- Approximates the Koopman operator on linear observables; extended DMD uses nonlinear dictionaries.

## Other lenses

- *Linear algebra.* POD is PCA of trajectories; balanced truncation is an SVD of the Hankel operator; DMD is an eigendecomposition of a least-squares fitted linear map (see [low-rank](/ref/approx/low-rank)).
- *Numerics.* The stiff ODE and the reduced DAE are two treatments of the same timescale separation; the DAE solver effectively performs the QSSA numerically (see [DAEs](/ref/integrate/daes)).
- *Identifiability.* Fast parameters in a QSSA regime appear only in combinations (e.g. $`K_M`$, $`V_{\max}`$); reduction explains why individual rate constants are not identifiable from slow data (see [identifiability](/ref/freq/identifiability)).

## Pitfalls & diagnostics

- *QSSA outside its validity regime* can be qualitatively wrong (e.g. high enzyme concentrations, missing oscillations).
- *POD extrapolates poorly:* the basis only spans dynamics seen in the snapshots. Include the inputs and parameter ranges you will use.
- *Projection can destroy stability* for nonlinear or non-normal systems; balanced truncation preserves it, Galerkin POD does not in general.
- *Validate* reduced vs full models on held-out scenarios, including transients, not just steady states.

## See also

- [DAEs](/ref/integrate/daes)
- [Low-rank approximation](/ref/approx/low-rank)
- [Surrogate models](/ref/approx/surrogates)
- [Identifiability](/ref/freq/identifiability)

*References:*

- Segel & Slemrod, "The quasi-steady-state assumption: a case study in perturbation", SIAM Review (1989).
- Antoulas, _Approximation of Large-Scale Dynamical Systems_.
- Kutz, Brunton, Brunton & Proctor, _Dynamic Mode Decomposition_.
