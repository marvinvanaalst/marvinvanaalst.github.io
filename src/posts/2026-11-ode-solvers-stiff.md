---
title: 'Inside Stiff ODE Solvers'
description: 'BDF, Radau, Rosenbrock and SDIRK compared by the decisions that matter: the shape of the nonlinear system, how the Jacobian and its factorisation are reused, how error is estimated and interpolated, and what changes for large systems.'
categories:
  - musings
date: '2026-11-26'
author: 'Marvin van Aalst'
template: tutorials
published: true
---

# Inside Stiff ODE Solvers

The [previous post](/blog/2026-11-ode-solvers-nonstiff) compared non-stiff solvers by their design decisions: which solution is kept, how the error is estimated, how the step is controlled and how the solution is interpolated. Stiff solvers make all of those decisions too, but the expensive part is somewhere else. Every implicit method has to solve a nonlinear system in every step, and that turns into three new questions:

- What shape does the system have, and how many linear solves does it take?
- How often do you recompute the Jacobian, and how often do you refactorise the iteration matrix?
- What changes when the system is too large to factorise at all?

The answers matter more than the choice of tableau. On a stiff problem with an expensive Jacobian, a solver that refactorises too often or recomputes $`J`$ every step can be several times slower than one with the same method and a smarter reuse policy.

This post compares four families: BDF/NDF, Radau IIA, Rosenbrock/W and (E)SDIRK. The implementations are Hairer's RADAU5, RADAU, RODAS and SDIRK4; ODEPACK's LSODA (as vendored by SciPy); SUNDIALS' CVODE and ARKODE; SciPy's `BDF` and `Radau`; OrdinaryDiffEq.jl's `FBDF`, `QNDF`, `RadauIIA5`, `AdaptiveRadau`, `Rodas5P`, `TRBDF2`, `Kvaerno*` and `KenCarp*`; and diffrax's `Kvaerno3–5` and `KenCarp3–5`. All were read at pinned commits. For RODAS, RADAU and SDIRK4, whose originals couldn't be downloaded automatically, I used the copies in ODEInterface.jl and cpmech/hwode. The textbook background is on the [initial value problems reference page](/ref/integrate/ode-initial-value). The third post covers [what lies between stiff and non-stiff](/blog/2026-11-ode-solvers-hybrid).

## The shape of the nonlinear system

Every implicit method ends up solving systems with a matrix of the form $`I - h\gamma J`$ (or $`M - h\gamma J`$ with a mass matrix). The families differ in how many such systems there are per step, how large they are, and whether they need a Newton iteration at all.

### Radau IIA: one big system, split into a real and a complex part

Three-stage Radau IIA (order 5) is a fully implicit Runge–Kutta method: all three stages are coupled, so a Newton step on the stage values is a $`3n \times 3n`$ linear system with matrix $`I \otimes I - hA \otimes J`$. Factorising that directly would cost $`27\times`$ a single $`n\times n`$ factorisation.

The trick every Radau code uses, due to Butcher and to Hairer and Wanner, is to diagonalise $`A^{-1}`$. Its eigenvalues are one real value $`\hat\gamma`$ and a complex pair $`\hat\alpha \pm i\hat\beta`$. A change of variables with the eigenvector matrix $`T`$ decouples the system into

```math
\Big(\tfrac{\hat\gamma}{h} I - J\Big)\,\Delta W_1 = r_1,
\qquad
\Big(\tfrac{\hat\alpha + i\hat\beta}{h} I - J\Big)(\Delta W_2 + i\,\Delta W_3) = r_2 + i\,r_3 .
```

That's one real and one complex $`n\times n`$ factorisation, roughly $`1 + 4 = 5`$ real-factorisation equivalents instead of 27. RADAU5 hard-codes $`T`$, $`T^{-1}`$ and the eigenvalues ($`\hat\gamma \approx 3.638`$, `radau5.f` lines 715–743).

The codes agree on the transformation but differ in how they handle the complex system:

- *RADAU5* has its own complex LU (`DECC`/`SOLC` in `decsol.f`) working on separate real and imaginary arrays. Fortran 77's `COMPLEX` type wasn't used.
- *SciPy's* `Radau` builds a complex matrix and calls `lu_factor` (or `splu` for sparse Jacobians) on it ([`radau.py`](https://github.com/scipy/scipy/blob/808b5af948c418c105d0734284cfc88084d24db1/scipy/integrate/_ivp/radau.py#L463-L464)).
- *OrdinaryDiffEq's* `RadauIIA5` copies Hairer's $`T`$ to all digits and builds a complex $`W_2`$ that LinearSolve.jl factorises ([source](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqFIRK/src/firk_perform_step.jl#L884-L897)).

None of them uses the textbook alternative of rewriting the complex system as a real $`2n \times 2n`$ one, which would cost twice as much.

RADAU5 has one more trick: it can reduce $`J`$ to Hessenberg form once per Jacobian. Every subsequent refactorisation with a new $`h`$ then costs $`O(n^2)`$ instead of $`O(n^3)`$. OrdinaryDiffEq has the analogue as `LHLFactorization`, where the Hessenberg reduction stays real and only the shift is complex.

The variable-order versions scale the same idea up. Hairer's `RADAU` and OrdinaryDiffEq's `AdaptiveRadau` switch between 3, 5 and 7 stages (orders 5, 9 and 13), at the cost of one real and $`(s-1)/2`$ complex factorisations.

### SDIRK and ESDIRK: one factorisation shared by all stages

Singly diagonally implicit RK methods make $`A`$ lower triangular with the same $`\gamma`$ on the whole diagonal. The stages can then be solved one after another, each an $`n \times n`$ Newton problem with the same matrix $`I - h\gamma J`$. ESDIRK methods additionally make the first stage explicit, which buys a higher stage order (more on that below).

Sharing $`\gamma`$ is the whole point of the design: one factorisation per step serves every stage. All codes exploit this.

- *OrdinaryDiffEq* builds $`W`$ at the first implicit stage and marks it as reusable: "All implicit stages share γ on the diagonal; reuse the W from stage 1" ([source](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqSDIRK/src/generic_imex_perform_step.jl#L191-L192)).
- *diffrax* computes the Jacobian once per step, at the first stage for SDIRK and the second for ESDIRK. It even asserts that the diagonal is constant.
- *ARKODE* gets the same effect from its general rule: refactorise only when $`\gamma`$ changes by more than 20 %, which a constant diagonal never does.
- *Hairer's SDIRK4* forms one $`E = M/(h\gamma) - J`$ per step.

### Rosenbrock: no Newton iteration at all

Rosenbrock methods linearise once per step and never iterate. Each stage solves

```math
(I - h\gamma J)\,k_i = h f\Big(y_n + \sum_{j<i}\alpha_{ij}k_j\Big) + hJ\sum_{j<i}\gamma_{ij}k_j ,
```

with the same matrix for all stages. That makes one factorisation and $`s`$ back-substitutions per step, with no convergence test, no iteration count and no risk of Newton failure. Rodas5 and Rodas5P use 8 stages, so 8 solves. Hairer's RODAS uses 6 solves.

The price is that $`J`$ enters the method coefficients. A Rosenbrock method's order conditions assume the _exact_ Jacobian, so a stale or approximate $`J`$ doesn't just slow convergence, it lowers the order. W-methods (Steihaug–Wolfbrandt) add order conditions that make the method correct for any matrix $`W`$, at the cost of more stages or lower order. OrdinaryDiffEq ships both kinds and treats them differently, as described below.

### BDF: one system per step

BDF methods have the smallest system: one $`n \times n`$ nonlinear system per step, solved by a few simplified Newton iterations with $`I - h\gamma J`$, where $`\gamma = 1/\alpha_k`$ depends on the order. That system changes little from step to step, which is why BDF codes have the most aggressive reuse of factorisations (next section).

How the history is stored differs, as in the Adams case of the previous post:

- *CVODE and LSODA* use a Nordsieck array. CVODE recomputes the coefficients from the actual step history each step ([`cvode.c`](https://github.com/LLNL/sundials/blob/312fc0f3684f27209ca9dc9249d194436eb41a7a/src/cvode/cvode.c#L2975-L2988)), while LSODA uses fixed tables.
- *SciPy's* `BDF` stores backward differences and interpolates them onto an equally spaced grid whenever $`h`$ changes (the "quasi-constant step size" of Shampine and Reichelt).
- *OrdinaryDiffEq's* `FBDF` re-interpolates its history onto equally spaced past points with a Lagrange polynomial, so that fixed BDF coefficients apply.

| Family         | Systems per step               | Factorisations per step (when refreshed) | Newton iteration? |
| -------------- | ------------------------------ | ---------------------------------------- | ----------------- |
| Radau IIA (3)  | one $`3n`$, decoupled          | 1 real + 1 complex                       | yes               |
| SDIRK / ESDIRK | $`s`$ sequential $`n\times n`$ | 1, shared by all stages                  | yes, per stage    |
| Rosenbrock     | $`s`$ linear $`n\times n`$     | 1                                        | no                |
| BDF / NDF      | one $`n\times n`$              | 1, often reused for many steps           | yes               |

## Stability vs. accuracy in the coefficients

Explicit pairs spend their free coefficients on a small error constant. Implicit methods have a second budget to manage: how they treat modes with $`h\lambda \to -\infty`$.

### A-stable is not enough

A method is A-stable if it is stable for every $`h\lambda`$ in the left half-plane. That says nothing about how strongly it _damps_ very stiff modes. The trapezoidal rule (Crank–Nicolson) is A-stable, but its stability function tends to $`-1`$ as $`h\lambda \to -\infty`$: a stiff component is not damped at all, it just flips sign every step. L-stable methods have $`R(\infty) = 0`$ and kill such components.

![Magnitude of the stability function on the negative real axis for trapezoidal, implicit Euler, Radau IIA and Kvaerno5](/tutorials/ode-solvers-implicit-damping.png)

The figure shows the effect. At $`h\lambda = -10^4`$, Crank–Nicolson still has $`\lvert R\rvert \approx 1`$. Radau IIA and Kvaerno5 damp the mode by $`10^{-3}`$–$`10^{-4}`$ per step, and implicit Euler damps it fastest of all.

L-stability usually comes from being _stiffly accurate_: the last row of $`A`$ equals $`b`$, so $`y_{n+1}`$ is just the last stage value. Radau IIA has this property by construction: OrdinaryDiffEq's update is literally `u = uprev + z3`. The ESDIRK methods in all three libraries carry it in their docstrings ("A-L stable stiffly-accurate"), and Rodas5P is built with $`b`$ equal to the last row of its coefficient matrix ([tableau](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl#L65-L72)).

### Stage order and order reduction

The classical order of an RK method describes its behaviour as $`h\to 0`$ with $`h\lambda`$ small. On stiff problems, $`h\lambda`$ is large, and the effective order drops towards the _stage order_, the accuracy of the internal stages. SDIRK methods have stage order 1, so they can lose most of their order on stiff problems. ESDIRK methods reach stage order 2 thanks to the explicit first stage. Kennedy and Carpenter put that into their names: in ARKODE's default `ESDIRK436L2SA`, "L2SA" stands for L-stable, stage order 2, stiffly accurate. ARKODE's default implicit tables are this family at every order from 3 to 5 ([`arkode_arkstep.h`](https://github.com/LLNL/sundials/blob/312fc0f3684f27209ca9dc9249d194436eb41a7a/include/arkode/arkode_arkstep.h#L58-L63)). Radau IIA, with stage order 3, degrades least.

### BDF: order against angle

BDF methods trade stability for order more bluntly. BDF1 and BDF2 are A-stable. From order 3 on, the stability region leaves out a wedge around the imaginary axis. The method is only $`A(\alpha)`$-stable, stable in a sector of half-angle $`\alpha`$ around the negative real axis. BDF7 and higher aren't even zero-stable, which is why every code here stops at order 5.

![BDF stability boundaries for orders 1 to 6](/tutorials/ode-solvers-bdf-stability.png)

Computed from the boundary locus in the figure, the angles are 86.0° for BDF3, 73.4° for BDF4 and 51.8° for BDF5. A problem with eigenvalues close to the imaginary axis, such as a lightly damped oscillation, is therefore a bad fit for high-order BDF. CVODE has an optional _stability-limit detection_ for exactly this case (off by default), and OrdinaryDiffEq's `FBDF` turns its equivalent `stald` on by default. Both reduce the order when they detect that the step is being limited by the edge of the wedge.

### NDF: giving back a little angle for accuracy

Klopfenstein's and Shampine's numerical differentiation formulas (NDFs) add one term to BDF, $`\kappa\,\gamma_k\,\nabla^{k+1}y_{n+1}`$. That shrinks the error constant at the cost of a slightly smaller stability angle. Both SciPy's `BDF` and OrdinaryDiffEq's `QNDF` use Shampine and Reichelt's $`\kappa = (-0.185, -1/9, -0.0823, -0.0415, 0)`$ for orders 1–5 ([`bdf.py`](https://github.com/scipy/scipy/blob/808b5af948c418c105d0734284cfc88084d24db1/scipy/integrate/_ivp/bdf.py#L246-L249), [`QNDF`](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqBDF/src/algorithms.jl#L392-L399)). Computed from those values:

| Order | BDF angle | NDF angle | Error constant BDF → NDF   | Step gain at equal accuracy |
| ----- | --------- | --------- | -------------------------- | --------------------------- |
| 1     | 90°       | 90°       | 0.500 → 0.315              | 26 %                        |
| 2     | 90°       | 90°       | 0.333 → 0.167              | 26 %                        |
| 3     | 86.0°     | 80.4°     | 0.250 → 0.099              | 26 %                        |
| 4     | 73.4°     | 66.2°     | 0.200 → 0.114              | 12 %                        |
| 5     | 51.8°     | 51.8°     | unchanged ($`\kappa = 0`$) | —                           |

The coefficients were chosen so that each order gains about a quarter in step size. At order 5 the angle is already too small to give any away. OrdinaryDiffEq's `FBDF` is plain BDF, even though its docstring claims Shampine's κ values: the struct has no κ field.

### Re-fitting Rosenbrock coefficients: Rodas5P

Rosenbrock tableaux have a lot of free parameters, and the classical choices were made with ODEs in mind. Steinebach re-fitted Rodas5's coefficients in 2023 (Rodas5P, with $`\gamma = 0.2119`$ instead of $`0.19`$), describing the result as having "improved stability in the adaptive time stepping embedding". It is now the default stiff method of OrdinaryDiffEq for small, tight-tolerance or mass-matrix problems.

### Tolerance transformation

One more coefficient-level decision hides in the tolerance handling. RADAU5 internally replaces $`\text{rtol}`$ by $`0.1\,\text{rtol}^{2/3}`$ (and scales $`\text{atol}`$ accordingly). Its 3rd-order error estimate controls a 5th-order method, and the transformation makes the delivered accuracy scale roughly proportionally to the requested one. OrdinaryDiffEq's `RadauIIA5` does the same (`cache.rtol = reltol^(2 / 3) / 10`). SciPy's `Radau` doesn't, so `rtol=1e-6` means different things in SciPy and in the other two.

## Newton and the iteration matrix

This is where stiff codes differ most, and where most of the speed differences come from. Every code uses _simplified_ Newton: the iteration matrix is built from a Jacobian that may be several steps old, and it is reused across iterations and often across steps. The design questions are when to recompute $`J`$, when to refactorise $`I - h\gamma J`$, and how to tell that an old matrix has gone bad.

### The convergence test

All codes monitor the contraction rate $`\theta_k = \lVert\Delta_k\rVert / \lVert\Delta_{k-1}\rVert`$ of the Newton corrections and estimate the remaining error as $`\theta/(1-\theta)\,\lVert\Delta_k\rVert`$. The details differ:

| Code                | Max iterations | Stop when                                                                                              | Give up when                                                             |
| ------------------- | -------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| RADAU5              | 7              | $`\tfrac{\theta}{1-\theta}\lVert\Delta\rVert \le \text{fnewt}`$, $`\theta`$ geometric mean of last two | $`\theta \ge 0.99`$, or predicted not to converge in time                |
| SciPy Radau / BDF   | 6 / 4          | same, $`\text{tol} = \max(10\varepsilon/\text{rtol}, \min(0.03, \sqrt{\text{rtol}}))`$                 | rate $`\ge 1`$, or predicted not to converge                             |
| CVODE / ARKODE      | 3              | $`\lVert\delta\rVert\min(1, \text{crate}) \le 0.1 \cdot`$ error tolerance                              | $`\lVert\delta\rVert > 2\lVert\delta_{\text{prev}}\rVert`$ (ARKODE: 2.3) |
| LSODA               | 3              | $`\lVert\delta\rVert \min(1, 1.5\,\text{crate}) \le`$ tolerance                                        | $`\lVert\delta\rVert > 2\lVert\delta_{\text{prev}}\rVert`$               |
| OrdinaryDiffEq      | 10             | $`\tfrac{\theta}{1-\theta}\lVert\Delta\rVert < \kappa = 0.01`$                                         | $`\theta > 2`$ (Radau: $`\theta > 1`$)                                   |
| diffrax `VeryChord` | 10             | $`\tfrac{\theta}{1-\theta}\lVert\Delta\rVert < 0.01`$, at least 2 iterations                           | $`\theta > 2`$ or non-finite                                             |

Two features are worth copying:

- *Predicted non-convergence.* RADAU5 and SciPy check whether the iteration can still converge within the remaining iterations at the current rate. If not, they abort early instead of wasting iterations ([`radau.py`](https://github.com/scipy/scipy/blob/808b5af948c418c105d0734284cfc88084d24db1/scipy/integrate/_ivp/radau.py#L119-L133)).
- *Rate memory.* CVODE smooths the rate as $`\max(0.3\cdot\text{crate}_{\text{old}}, \text{ratio})`$, so a single lucky iteration can't fake convergence. OrdinaryDiffEq's Newton solver uses the same 0.3.

The Newton tolerance is tied to the error tolerance everywhere: fnewt in Hairer's codes, 0.1 times the local error tolerance in CVODE, the controller's `rtol`/`atol` in diffrax. Solving the nonlinear system much more accurately than the step's own error would be wasted work.

### When to refresh $`J`$ and the factorisation

This is the most varied part of the whole comparison:

- *RADAU5* keeps $`J`$ for the next step if the last Newton iteration contracted fast ($`\theta < 0.001`$). If, in addition, the new step size is within $`[1, 1.2]`$ of the old one, it keeps the _step size_ and the factorisation too. It freezes $`h`$ to avoid a refactorisation (`radau5.f` lines 504–513, 1067–1074).
- *SciPy Radau* has a similar rule: recompute $`J`$ after an accepted step only if Newton needed more than 2 iterations and the rate exceeded $`10^{-3}`$. Keep the factorisation if $`J`$ is kept and the step would grow by less than 20 %, again freezing $`h`$ ([source](https://github.com/scipy/scipy/blob/808b5af948c418c105d0734284cfc88084d24db1/scipy/integrate/_ivp/radau.py#L508-L517)).
- *SciPy BDF* never recomputes $`J`$ proactively, only after a Newton failure with a stale one. The factorisation is rebuilt on every step-size or order change. After an _error-test_ failure it deliberately keeps the old factorisation even though $`h`$ changed ("As we didn't have problems with convergence, we don't reset LU here").
- *CVODE* uses a two-level policy. It refactorises $`I - \gamma J`$ when $`\gamma`$ has changed by more than 30 %, after a failure, or every 20 steps. Inside that, it re-evaluates $`J`$ only every 51 steps or after a convergence failure that points to a bad $`J`$ ([`cvode.c`](https://github.com/LLNL/sundials/blob/312fc0f3684f27209ca9dc9249d194436eb41a7a/src/cvode/cvode.c#L3115-L3118), [`cvode_ls.c`](https://github.com/LLNL/sundials/blob/312fc0f3684f27209ca9dc9249d194436eb41a7a/src/cvode/cvode_ls.c#L1585-L1590)). When it solves with a factorisation built for an old $`\gamma`$, it rescales the solution by $`2/(1 + \gamma/\gamma_{\text{old}})`$. That is a first-order correction that makes the stale matrix good enough for longer. ARKODE uses the same scheme with a 20 % threshold.
- *LSODA* has a single level: $`J`$ and the iteration matrix are recomputed together when $`h\ell_0`$ changes by more than 30 % or every 20 steps.
- *OrdinaryDiffEq* (SDIRK and BDF via its Newton solver) builds a new $`W`$ when $`\gamma\,dt`$ changes by more than 20 %, after an error failure or on divergence. It builds a new $`J`$ only when the iteration failed with a stale one, with no age limit.
- *OrdinaryDiffEq's* `RadauIIA5` reuses $`J`$ only after fast convergence, with a hard limit of 50 steps. It forces a fresh contraction measurement every 16 one-iteration steps, constants the source says were "tuned on vdp/rober/orego/hires/bruss" ([source](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqFIRK/src/firk_perform_step.jl#L1-L8)).
- *Rosenbrock methods* can't reuse $`J`$, because their order depends on it. Hairer's RODAS and OrdinaryDiffEq's strict Rosenbrock methods recompute $`J`$ every accepted step. After a rejection only the matrix is rebuilt with the new $`h`$.
- *OrdinaryDiffEq's W-methods* (`Rosenbrock23`, `ROS34PW*`, `Rodas23W`, …) may keep $`J`$ for up to 20 steps or until $`\gamma\,dt`$ changes by 3 %, a policy the code calls "CVODE-inspired". The reuse is switched off for mass-matrix problems, because "stale Jacobians cause order reduction for DAEs", and for Krylov solvers ([source](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqDifferentiation/src/derivative_utils.jl#L88-L107)).
- *diffrax* computes a fresh Jacobian (by JAX autodiff) and a fresh factorisation every step, and reuses nothing across steps. If the Newton iteration fails, it sets the error estimate to infinity, so the step controller simply rejects the step and shrinks $`h`$.

These policies reflect what each code assumes is expensive. CVODE was built for large PDE discretisations, where a Jacobian costs many function evaluations and a factorisation is the dominant cost, so it reuses both as long as it can. diffrax runs on accelerators, where an autodiff Jacobian of a moderately sized system is cheap and batched linear algebra is fast. It also lives in a functional framework where carrying a cached factorisation across steps is awkward. Hairer's codes sit in between. They freeze the step size to avoid refactorising, which is a decision explicit codes never have to make.

Several codes also adjust the step-size safety factor to the Newton iteration count. RADAU5, SciPy's `Radau` and OrdinaryDiffEq's predictive controller all lower the safety factor when Newton needed many iterations, using RADAU5's formula $`\text{fac} = 0.9\,(2\,\text{nit} + 1)/(2\,\text{nit} + \text{newt})`$. A step that was hard to solve is taken as a sign that the next one shouldn't be larger.

## Error estimation

### Radau: filtering the estimate

Radau IIA has no natural embedded method of order 4. RADAU5 combines the stage values into a 3rd-order estimate. Applied naively on a stiff problem, that estimate is huge for the stiff components, because it contains terms like $`hJ\,e`$, and the step would be rejected over and over. Hairer and Wanner's fix is to filter it through the iteration matrix that has already been factorised:

```math
\text{err} = \Big(\tfrac{\hat\gamma_0}{h} I - J\Big)^{-1}\Big[f(t_n, y_n) + \tfrac{1}{h}\sum_i e_i Z_i\Big].
```

For non-stiff components the filter is close to the identity times a constant. For stiff ones it divides by $`h\lambda`$ and cancels the spurious growth. The extra cost is one back-substitution.

If the filtered error still exceeds 1, RADAU5 filters a second time, after one extra function evaluation, but only on the first step or after a rejection. The three Radau codes have each picked a different subset of those two cases:

- *RADAU5* refilters on the first step _or_ after a rejection.
- *SciPy* refilters only after a rejection ([`radau.py`](https://github.com/scipy/scipy/blob/808b5af948c418c105d0734284cfc88084d24db1/scipy/integrate/_ivp/radau.py#L485-L495)).
- *OrdinaryDiffEq* refilters only on the first step or after a discontinuity ([source](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqFIRK/src/firk_perform_step.jl#L1103-L1105)).

The same filter shows up elsewhere:

- *Hairer's SDIRK4* filters its embedded difference through its iteration matrix.
- *OrdinaryDiffEq* does the same for `TRBDF2`, `Kvaerno5` and `KenCarp4` (the `smooth_est` option, on by default).
- *ARKODE and diffrax* use the plain embedded difference for their ESDIRK methods. On very stiff problems that difference can be pessimistic.

### Rosenbrock: the last stage is the estimate

Because Rodas-type methods are stiffly accurate, their embedded solution can be the second-to-last stage. The error estimate is then simply the last stage increment, $`\tilde b = e_s`$, both in Hairer's RODAS and in OrdinaryDiffEq's `Rodas4`/`Rodas5P` tableaux. That's free and naturally behaves well on stiff components, since the last stage has passed through the same $`(I - h\gamma J)^{-1}`$ as everything else.

### BDF: predictor against corrector

BDF codes estimate the local error from the difference between the predicted and the corrected value, scaled by the error constant of the current order:

- *CVODE* uses $`\lVert y_n - y_n^{(0)}\rVert\cdot t_q`$.
- *SciPy* multiplies the accumulated Newton correction by `error_const[order]`, which already includes the NDF modification $`\kappa\gamma_k + 1/(k+1)`$.
- *OrdinaryDiffEq's* `FBDF` additionally estimates the errors at orders $`k-1`$, $`k-2`$ and $`k+1`$ from finite differences over its history, for order selection.

Again the estimate is free: the Newton iteration produces it.

## Dense output

Each family has a natural interpolant, and only some codes use it.

- *Radau IIA* is a collocation method. The polynomial through the stage values satisfies the ODE at the collocation points and is a free continuous extension. RADAU5's `CONTR5`, SciPy's `Radau` and OrdinaryDiffEq's `RadauIIA5` all use it. The same polynomial, extrapolated forward, is also the starting guess for the next step's Newton iteration. That's a nice reuse, and a reason Radau's Newton iteration often needs only one or two corrections.
- *Rosenbrock methods* need a dedicated interpolant, because their stages are not solution values. RODAS has one built from stage combinations. OrdinaryDiffEq ships "stiff-aware" interpolants: 3rd order for Rodas4, 4th order for Rodas5/5P, and Shampine–Reichelt's 2nd-order formula for `Rosenbrock23`. These are constructed so that the interpolant doesn't reintroduce the stiff components the step has just damped. A plain Hermite cubic through $`f_n`$ and $`f_{n+1}`$ can do exactly that.
- *BDF* methods have their history polynomial. SciPy's `BDF` evaluates the Newton-form polynomial through its backward differences. CVODE's `CVodeGetDky` evaluates the Nordsieck polynomial, but only within the last step.
- *(E)SDIRK* methods fall back to Hermite: cubic in diffrax's `Kvaerno3–5` and OrdinaryDiffEq's SDIRK family, a custom 2nd/3rd-order interpolant for diffrax's `KenCarp`, and Hermite by default in ARKODE. diffrax's `ImplicitEuler` deliberately uses _linear_ interpolation, because a Hermite cubic "produces very oscillatory interpolations" on stiff problems ([source](https://github.com/patrick-kidger/diffrax/blob/e950aa2cf1974c02816c1542f20beabf7cca496b/diffrax/_solver/implicit_euler.py#L36-L42)). The comment states the same point as the Rosenbrock interpolants above.

## Step and order control

### Gustafsson's predictive controller

Implicit one-step methods mostly use a controller due to Gustafsson. It is designed for a situation that's common in stiff problems: the step size limited by accuracy on a slowly varying solution, with an error estimate that fluctuates. It compares the classical proposal with one extrapolated from the last two accepted steps,

```math
\frac{h_{\text{new}}}{h} = \text{fac}\cdot\text{err}_n^{-1/4}\cdot\min\Big(1,\ \frac{h_n}{h_{n-1}}\Big(\frac{\text{err}_{n-1}}{\text{err}_n}\Big)^{1/4}\Big),
```

and takes the smaller of the two. RADAU5 uses it by default. Its documentation says that it "seems to produce safer results", while the classical controller "produces often slightly faster runs" on simple problems. RODAS uses it, SciPy's `Radau` writes it in exactly this "take the minimum" form ([`radau.py`](https://github.com/scipy/scipy/blob/808b5af948c418c105d0734284cfc88084d24db1/scipy/integrate/_ivp/radau.py#L169-L175)), and OrdinaryDiffEq's `PredictiveController` is the default for its Radau methods.

The other codes don't use it by default:

- *OrdinaryDiffEq's SDIRK methods* get the generic PI controller, even though the `PredictiveController` docstring calls it "the default for algorithms like the (E)SDIRK methods".
- *ARKODE* defaults to an I-controller for implicit methods too. It offers Gustafsson's implicit controller as an option.
- *diffrax* has no predictive controller at all.

### Keep-step bands

Implicit codes avoid small step-size changes, because each change costs a refactorisation (or, for BDF, a rescaling of the history):

| Code                 | Keep $`h`$ when $`h_{\text{new}}/h`$ is in                 | Why                                         |
| -------------------- | ---------------------------------------------------------- | ------------------------------------------- |
| RADAU5               | $`[1, 1.2]`$, if $`J`$ is also kept                        | reuse the factorisation                     |
| SciPy Radau          | below 1.2, including decreases                             | reuse the factorisation                     |
| OrdinaryDiffEq Radau | $`[0.83, 1]`$: the band is applied to $`h/h_{\text{new}}`$ | avoid small shrinks                         |
| OrdinaryDiffEq FBDF  | $`[0.5, 1.11]`$                                            | avoid history re-interpolation              |
| CVODE                | below 1.5, including decreases                             | avoid Nordsieck rescaling and refactorising |
| LSODA                | below 1.1                                                  | same                                        |
| ARKODE               | none                                                       | —                                           |

SciPy's `Radau` and CVODE therefore never shrink the step after an accepted step. Only a rejection does that. OrdinaryDiffEq's Radau controller uses the same numbers as RADAU5, `qsteady` $`\in [1, 1.2]`$, but applies them to $`q = h/h_{\text{new}}`$. It ends up suppressing small _decreases_ instead of small increases, the mirror image of Hairer's band ([source](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqCore/src/integrators/controllers.jl#L1299-L1322)).

### Choosing the BDF order

BDF codes also choose the order, by estimating the error the next step would have at orders $`k-1`$, $`k`$ and $`k+1`$ and picking the order that allows the largest step:

- *SciPy* waits until $`k + 1`$ steps of equal size have been taken, then takes the best of the three ([`bdf.py`](https://github.com/scipy/scipy/blob/808b5af948c418c105d0734284cfc88084d24db1/scipy/integrate/_ivp/bdf.py#L428-L452)).
- *CVODE* waits $`q + 1`$ steps after each order change. It biases the estimates by 6 for orders $`q-1`$ and $`q`$ and by 10 for $`q+1`$, and on ties prefers keeping the order, then lowering, then raising it. After 3 error failures in a row it forces the order down.
- *LSODA* biases by 1.3, 1.2 and 1.4 and changes the step only for a gain of at least 10 %.
- *OrdinaryDiffEq's* `FBDF` uses CVODE's bias of 6 ("BIAS2 = 6 matches CVODE"). It only raises the order when the error estimates across orders $`k-2 \dots k+1`$ decrease monotonically, which guards against noise in the higher differences.

All of them bias towards _lower_ orders. That's for stability: a too-high BDF order on a problem with eigenvalues near the imaginary axis is the classic way to make a BDF code crawl.

The variable-order Radau codes use the Newton iteration to choose the stage count instead of an error estimate. Hairer's `RADAU` adds stages when the contraction rate is below 0.002 and the step is stable, and drops stages when it exceeds 0.8. OrdinaryDiffEq's `AdaptiveRadau` uses a smoothed Newton iteration count instead: more stages below 2.6 iterations on average, fewer above 8. The logic is the same in both: fast convergence means the step is easy and a higher order pays off.

## Large systems

Everything so far assumed dense LU, $`O(n^3)`$ per factorisation. For a few hundred unknowns that's fine. For a discretised PDE with $`10^5`$ unknowns it's impossible. The codes offer three ways out.

### Banded Jacobians

If $`J`$ is banded, as for 1D discretisations or reaction networks ordered sensibly, a banded LU costs $`O(n\,b^2)`$. RADAU5 (lower/upper bandwidths `MLJAC`/`MUJAC`), CVODE (band linear solver), and LSODA (`lband`/`uband`, also exposed by SciPy) all support it.

Banded structure also makes finite-difference Jacobians cheap. Columns that are more than one bandwidth apart don't interact, so a single perturbation can probe all of them at once, and $`b + 1`$ function evaluations give the whole Jacobian. RADAU5 and CVODE both do this column grouping.

### Sparse Jacobians and colouring

For general sparsity, the same idea becomes graph colouring. Two columns that share no nonzero row can be perturbed together, so the number of function evaluations for a Jacobian drops from $`n`$ to the number of colours (Curtis, Powell and Reid).

- *SciPy* takes a `jac_sparsity` pattern, groups columns greedily, and factorises with SuperLU (`splu`), for both the real and the complex Radau systems ([`radau.py`](https://github.com/scipy/scipy/blob/808b5af948c418c105d0734284cfc88084d24db1/scipy/integrate/_ivp/radau.py#L322-L331)).
- *OrdinaryDiffEq* turns a `jac_prototype` or sparsity pattern into a sparse automatic-differentiation backend with greedy colouring ([source](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqDifferentiation/src/alg_utils.jl#L152-L163)). The colouring then determines the number of forward-mode AD passes instead of finite differences. The factorisation goes through LinearSolve.jl, which picks a sparse LU.
- *diffrax* materialises the Jacobian through lineax. Sparsity has to come from the user's choice of linear solver.

### Matrix-free Newton–Krylov

When even a sparse factorisation is too expensive, the alternative is to never form $`J`$: solve each Newton system with a Krylov method (GMRES) that only needs products $`Jv`$.

- *CVODE* is the reference implementation. It approximates $`Jv \approx [f(y + \sigma v) - f(y)]/\sigma`$ with $`\sigma = 1/\lVert v\rVert_{\text{WRMS}}`$, solves only to $`0.05`$ times the Newton tolerance, and skips the linear solve entirely when the right-hand side is already below that tolerance ([`cvode_ls.c`](https://github.com/LLNL/sundials/blob/312fc0f3684f27209ca9dc9249d194436eb41a7a/src/cvode/cvode_ls.c#L1694-L1712)). Preconditioning is left to the user, with banded and band-block-diagonal preconditioners shipped as modules.
- *OrdinaryDiffEq* computes $`Jv`$ by forward-mode AD instead of a difference quotient, and solves with LinearSolve.jl's Krylov methods to the integrator's `reltol`. Its default algorithm switches to `FBDF` with GMRES above 500 unknowns ([`default_alg.jl`](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqDefault/src/default_alg.jl#L155-L168)).
- *SciPy's* stiff solvers have no Krylov option at all. They offer dense or sparse LU only.
- *diffrax* can take any lineax solver, including iterative ones.

The families don't all survive this equally:

- *BDF and ESDIRK* are fine. The Newton iteration only needs $`J`$ to converge, not to be exact, so an inexact Krylov solve just costs an extra iteration. That's why the large-system default is BDF with GMRES, in CVODE as in OrdinaryDiffEq.
- *Radau* would need a Krylov solve of a complex system with a complex shift, which is harder to precondition.
- *Rosenbrock* methods, as above, need the exact Jacobian in their coefficients, so an inexact linear solve reduces their order. OrdinaryDiffEq turns off Jacobian reuse for W-methods whenever a Krylov solver is used, because "Krylov convergence depends on W quality". The general rule: the more a method relies on $`J`$ being exact, the worse it fits a matrix-free setting.

## Mass matrices and DAEs, briefly

Many stiff problems come as $`My' = f(t, y)`$ with a singular $`M`$, a differential-algebraic system. Support varies:

- *RADAU5* accepts a full or banded $`M`$ and handles DAEs up to index 3 by scaling the error of index-2 and index-3 components by $`h`$ and $`h^2`$. RODAS and SDIRK4 also accept $`M`$.
- *OrdinaryDiffEq* threads `mass_matrix` through Radau, Rosenbrock and SDIRK. Its default algorithm switches to `Rodas5P` when $`M \neq I`$, and W-method Jacobian reuse is disabled for such problems.
- *ARKODE* supports fixed and time-dependent $`M`$ with its own mass-matrix linear solver.
- *CVODE* has no mass matrix. DAEs go to SUNDIALS' IDA, which works on the residual form $`F(t, y, y') = 0`$.
- *SciPy and diffrax* have no mass-matrix support at all.

Index, consistent initialisation and the rest are in the [DAE reference page](/ref/integrate/daes).

## Summary

| Decision             | Hairer                           | SciPy                                 | SUNDIALS                             | OrdinaryDiffEq.jl                                  | diffrax                   |
| -------------------- | -------------------------------- | ------------------------------------- | ------------------------------------ | -------------------------------------------------- | ------------------------- |
| Families             | Radau (5; 5/9/13), Rodas, SDIRK4 | Radau, BDF/NDF, LSODA                 | BDF (CVODE), ESDIRK (ARKODE)         | all four, plus W-methods                           | ESDIRK (Kvaerno, KenCarp) |
| Complex Radau system | hand-written complex LU          | complex LU                            | —                                    | complex LU via LinearSolve                         | —                         |
| $`J`$ reuse          | if $`\theta < 0.001`$            | Radau: similar; BDF: only on failure  | every ≤ 51 steps                     | Newton: on failure; Radau: fast convergence, ≤ 50  | never                     |
| Factorisation reuse  | freeze $`h`$ in $`[1, 1.2]`$     | freeze $`h`$ below 1.2 (Radau)        | ≤ 20 steps, $`\gamma`$ change ≤ 30 % | $`\gamma\,dt`$ change ≤ 20 %                       | never across steps        |
| Error estimate       | filtered (Radau, SDIRK4)         | filtered (Radau), predictor–corrector | predictor–corrector, embedded        | filtered (Radau, ESDIRK), last stage (Rodas)       | embedded, unfiltered      |
| Controller           | Gustafsson                       | Gustafsson (Radau), I (BDF)           | CVODE: per order; ARKODE: I          | Gustafsson (Radau), PI (SDIRK), CVODE-style (FBDF) | I                         |
| Large systems        | banded                           | banded (LSODA), sparse + colouring    | banded, Krylov + preconditioners     | sparse AD + colouring, Krylov                      | via lineax                |
| Mass matrix          | yes                              | no                                    | ARKODE yes, CVODE no (IDA)           | yes                                                | no                        |

What transfers:

- *The tableau matters less than the linear algebra.* Radau's decoupling, SDIRK's shared $`\gamma`$ and Rosenbrock's single factorisation are all ways of minimising factorisations per step. Every reuse rule above is a way of minimising them across steps.
- *Freeze the step to save a factorisation.* Hairer's codes, SciPy and CVODE all prefer a slightly suboptimal step size to a new factorisation. Explicit codes have no reason to do this. Implicit codes have every reason.
- *Filter error estimates through the iteration matrix.* It costs one back-substitution and makes estimates for Radau and SDIRK behave on stiff components.
- *Know which parts must be exact.* Newton-based methods tolerate stale or approximate Jacobians. Rosenbrock methods don't, unless they're W-methods. That one property decides which family survives a matrix-free, Krylov-based setting.
