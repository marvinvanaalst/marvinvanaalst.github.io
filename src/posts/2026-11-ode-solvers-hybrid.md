---
title: 'Between Stiff and Non-Stiff ODE solvers'
description: 'Stiffness detection and auto-switching, IMEX splitting, stabilised explicit and exponential integrators: four ways to avoid committing to explicit or implicit up front.'
categories:
  - musings
date: '2026-11-27'
author: 'Marvin van Aalst'
template: tutorials
published: true
---

# Between Stiff and Non-Stiff ODE solvers

The [first post](/blog/2026-11-ode-solvers-nonstiff) in this series was about explicit methods, which are cheap per step but whose step size collapses once the problem becomes stiff. The [second](/blog/2026-11-ode-solvers-stiff) was about implicit methods, which take large steps but pay for a Jacobian, a factorisation and a Newton iteration in every step. Real problems rarely fall cleanly on one side. A metabolic model is non-stiff during a transient and stiff once the fast reactions have equilibrated. A reaction–diffusion system is stiff in the diffusion and harmless in the reactions. A chemical network may be stiff in only one or two eigenvalues.

This post covers four ways to avoid committing up front:

1. *Detect and switch.* Run an explicit method, watch for stiffness, and change method when it shows up.
2. *Split the right-hand side.* Treat the stiff part implicitly and the rest explicitly, in the same step (IMEX).
3. *Stretch the explicit stability region.* Use many cheap stages to cover a long stretch of the negative real axis (RKC, ROCK, RKL).
4. *Integrate the stiff linear part exactly.* Use the matrix exponential and its relatives (exponential integrators).

As before, the comparison covers Hairer's Fortran codes, ODEPACK (as vendored by SciPy), SUNDIALS (CVODE and ARKODE), SciPy's `solve_ivp`, OrdinaryDiffEq.jl and diffrax, all read at pinned commits. The background on stability regions is in the [reference page on initial value problems](/ref/integrate/ode-initial-value).

## Detecting stiffness and switching

A switching solver needs three ingredients:

- a cheap estimate of the dominant eigenvalue (or at least of whether it limits the step),
- a decision rule that turns that estimate into "switch now",
- hysteresis, so that it doesn't flip-flop on every step.

The interesting part is that the codes agree on almost nothing beyond this list.

### Estimating $`h\lambda`$ for free

An explicit RK step already contains everything needed for a crude eigenvalue estimate. If two stages are evaluated at the same time point, then

```math
\frac{\lVert f(t, y_a) - f(t, y_b) \rVert}{\lVert y_a - y_b \rVert} \approx \lvert\lambda_{\max}\rvert
```

whenever the difference $`y_a - y_b`$ is dominated by the stiffest mode. That is the case exactly when stiffness is the problem: an explicit method on the edge of its stability region amplifies the stiff component above everything else. In effect it's one step of power iteration, and it costs nothing extra.

*Hairer's DOPRI5* uses stages 6 and 7, which both sit at $`t + h`$ (stage 7 is the FSAL evaluation at $`y_{n+1}`$), takes Euclidean norms, and compares $`h\lvert\lambda\rvert`$ to 3.25. That is roughly where DOPRI5's stability boundary crosses the negative real axis. *DOP853* does the same with its stage 12 and the post-step evaluation, against a threshold of 6.1 ([Hairer's codes](http://www.unige.ch/~hairer/software.html), `dopri5.f` lines 433–482, `dop853.f` lines 689–700).

*OrdinaryDiffEq.jl's* `Tsit5` copies the idea, with a comment that reads "Hairer II, page 22 modified to use the Inf norm" ([source](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqTsit5/src/tsit_perform_step.jl#L157-L164)). It divides componentwise, $`\lVert (k_7 - k_6) ./ (g_7 - g_6) \rVert_\infty`$, instead of taking a ratio of norms. That's more sensitive to a single stiff component, but it can produce $`0/0`$. The default algorithm deliberately counts a NaN estimate as "stiff", so it falls back to an implicit method rather than ignoring the test ([source](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqDefault/src/default_alg.jl#L124-L130)). `Vern7` does the same with its last two stages. The estimate is compared against a hard-coded stability size per method: 3.5068 for Tsit5, 4.64 for Vern7, 3.3066 for DP5. These are the real-axis extents of the [stability regions in the first post](/blog/2026-11-ode-solvers-nonstiff).

*LSODA* (Petzold 1983, the ODEPACK code that SciPy's `LSODA` wraps) runs Adams methods with a _functional_ (fixed-point) corrector iteration. That iteration contracts with rate $`\approx h\,\ell_0 \lVert J\rVert`$, so its observed convergence rate is itself a Lipschitz estimate: `pdest = max(pdest, rate / |h·el0|)`. The code forces at least two corrector iterations so that a rate can be measured at all ([`lsoda.c`](https://github.com/scipy/scipy/blob/808b5af948c418c105d0734284cfc88084d24db1/scipy/integrate/src/lsoda.c#L757-L782)). Again, the estimate costs no extra evaluations. It's a by-product of a corrector that runs anyway.

On the way _back_ from stiff to non-stiff, both switching codes use a different estimate, for the same reason: the stiff method already has a Jacobian. LSODA takes a weighted max-row-sum norm of $`J`$, recovered from the Newton matrix $`I - h\ell_0 J`$. OrdinaryDiffEq sets `eigen_est = opnorm(J, Inf)` every time the stiff solver forms a fresh Jacobian ([source](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqDifferentiation/src/derivative_utils.jl#L355-L357)). A norm is an upper bound on the spectral radius, so this biases the decision towards staying stiff. That's the safe direction.

### Deciding

The codes then differ sharply in what they compare the estimate against.

- *Hairer* compares $`h\lvert\lambda\rvert`$ with the method's own stability boundary. If it exceeds the boundary, the step was stability-limited.
- *OrdinaryDiffEq* does the same, $`\lvert \lambda h\rvert / \beta_{\text{nonstiff}} > 0.9`$. It does so on both branches, so even while running the stiff method it asks "would the explicit method be stable at this step size?" ([`composite_algs.jl`](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqCore/src/composite_algs.jl#L17-L22)).
- *LSODA* asks a more economic question: which method would allow the larger step at the same accuracy? It converts the current error estimate into the step each family could take, using precomputed error constants for Adams orders 1–12 and BDF orders 1–5. It caps the Adams step by its stability interval, $`h\cdot\text{pdlast} \le \text{sm1}[q]`$, where `sm1` runs from 0.5 at order 1 down to 0.025 at order 12 ([source](https://github.com/scipy/scipy/blob/808b5af948c418c105d0734284cfc88084d24db1/scipy/integrate/src/lsoda.c#L1200-L1221)). There is no "stiffness ratio" anywhere in the code, just two predicted step sizes.

LSODA's version is the most principled: it switches when switching pays, not when some threshold is crossed. It also captures something the other two miss. High-order Adams methods have tiny stability intervals, so a problem can be stability-limited for Adams at order 10 and not at order 2. While the Adams order is above 5, LSODA skips the switching test entirely, on the assumption that a problem running at high Adams order isn't stiff.

### Hysteresis

Every code needs to avoid flip-flopping, and every code chose different numbers:

| Code            | Go stiff when                                                                 | Go back when                                 | On switch                                         |
| --------------- | ----------------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------- |
| DOPRI5 / DOP853 | 15 positive tests (6 negatives in a row reset the count)                      | never: there is no stiff method to switch to | abort with `IDID = -4`                            |
| LSODA           | BDF step $`\geq 5\times`$ Adams step, at least 20 steps after the last switch | Adams step $`\geq`$ BDF step (factor 1)      | restart, order capped                             |
| OrdinaryDiffEq  | 11 consecutive stiff tests                                                    | 4 consecutive non-stiff tests                | $`h \times 2`$ going stiff, $`\div 2`$ going back |

Two patterns stand out.

First, the asymmetry. LSODA demands a fivefold advantage to go stiff but none to come back, and OrdinaryDiffEq waits 11 steps one way and 4 the other. Going stiff is expensive: Jacobian evaluations, factorisations, a restart of the history. Going back is cheap. So both codes are reluctant to enter the stiff regime and eager to leave it.

Second, Hairer's codes don't switch at all. DOPRI5's stiffness test runs on every 1000th accepted step by default, and on every step once it has fired. After 15 positives it returns with "problem is probably stiff". This is a different philosophy, and a reasonable one: a non-stiff code that silently becomes slow is worse than one that tells you to use RADAU5. The original Fortran also has two quirks:

- The abort only happens with printing turned off. With `IPRINT > 0`, DOPRI5 prints "the problem seems to become stiff" once and carries on. SciPy's C translation (behind the legacy `scipy.integrate.ode('dopri5')`) always aborts ([`dop.c`](https://github.com/scipy/scipy/blob/808b5af948c418c105d0734284cfc88084d24db1/scipy/integrate/src/dop.c#L467-L487)).
- DOPRI5 uses the signed step $`h`$ instead of $`\lvert h\rvert`$ in its estimate, so integrating backwards in time silently disables the detector. DOP853 uses $`\lvert h\rvert`$.

OrdinaryDiffEq's default algorithm (`solve(prob)` without a method) is the most elaborate version of the switching idea. It is a six-method composite:

- *Non-stiff side:* `Tsit5` or `Vern7`, chosen by `reltol`.
- *Stiff side:* `Rosenbrock23`, `Rodas5P`, `FBDF` or a Krylov `FBDF`, chosen by tolerance and system size (more than 50 or 500 states) ([source](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqDefault/src/default_alg.jl#L146-L168)).
- *Mass matrices:* problems with one start stiff and stay stiff.

One trap for readers of its docstring: `AutoSwitch`'s documented defaults (`nonstifftol = 3//4`, `switch_max = 10`) don't match the constructor (`9//10` and `5`) ([source](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqCore/src/composite_algs.jl#L4-L15)). Despite its name, `switch_max` doesn't cap the number of switches. It only decides when the integrator's error checks are skipped.

### Why the others don't switch

*CVODE* fixes Adams or BDF once, in `CVodeCreate(lmm)`, and never changes it ([`cvode.c`](https://github.com/LLNL/sundials/blob/312fc0f3684f27209ca9dc9249d194436eb41a7a/src/cvode/cvode.c#L240-L274)). That's notable because CVODE descends from the same Hindmarsh lineage as LSODA and has all the machinery: Nordsieck arrays for both families, error constants, a corrector iteration whose convergence rate it monitors. The only stability heuristic it ships is BDF _stability-limit detection_ (`CVodeSetStabLimDet`), which lowers the BDF order when an oscillatory mode sits near the edge of the $`A(\alpha)`$ region. That isn't a method switch.

*diffrax* has no switching either. Its documentation tells you to choose: `Tsit5` for non-stiff, `Kvaerno3/4/5` for stiff, `KenCarp4` for split stiffness. *SciPy's* `solve_ivp` likewise asks the user to "first try to run 'RK45'" and to move to `Radau` or `BDF` if it struggles. Its one switching method, `LSODA`, is the ODEPACK code ([`ivp.py`](https://github.com/scipy/scipy/blob/808b5af948c418c105d0734284cfc88084d24db1/scipy/integrate/_ivp/ivp.py#L229-L241)).

None of these codes state their reasons in the source, so what follows is my reading. Switching requires the user to supply everything the stiff method needs (a Jacobian, or at least the structure to build one), even if the switch never happens. It also requires both methods' state to be translatable mid-integration. In LSODA that's easy, because both families share one Nordsieck history. In a Runge–Kutta composite it means a restart: OrdinaryDiffEq re-initialises the target method's cache on every switch. Most users of such a library know whether their problem is stiff, and for them a clean separation is easier to reason about. And in a JAX library like diffrax, a data-dependent method switch would mean compiling both branches.

## IMEX: split the right-hand side

If you _know_ which part of $`f`$ is stiff, you don't need to detect anything. Write

```math
y' = f_E(t, y) + f_I(t, y),
```

with $`f_E`$ non-stiff (advection, slow reactions, forcing) and $`f_I`$ stiff (diffusion, fast relaxation), and use an _additive_ Runge–Kutta (ARK) method. That pairs an explicit tableau $`(A^E, b^E, c^E)`$ with an implicit one $`(A^I, b^I, c^I)`$:

```math
Y_i = y_n + h\sum_{j<i} a^E_{ij} f_E(Y_j) + h\sum_{j\le i} a^I_{ij} f_I(Y_j),
\qquad
y_{n+1} = y_n + h\sum_i \big(b^E_i f_E(Y_i) + b^I_i f_I(Y_i)\big).
```

The Newton iteration only ever sees $`\partial f_I/\partial y`$. If $`f_I`$ is linear (diffusion), the "Newton" iteration is a single linear solve. If it's local (reactions in each grid cell), the Jacobian is block-diagonal.

### Coupling, and why the tables share weights

Each tableau has to satisfy its own order conditions, plus _coupling_ conditions for the trees in which $`f_E`$ and $`f_I`$ appear together. Those are far easier to satisfy if the two tableaux share $`b`$ and $`c`$, and that's what the Kennedy–Carpenter pairs every library uses do. In diffrax's `KenCarp3`, the explicit and implicit tableaux are literally constructed from the same `b_sol`, `b_error` and `c` objects ([source](https://github.com/patrick-kidger/diffrax/blob/e950aa2cf1974c02816c1542f20beabf7cca496b/diffrax/_solver/kencarp3.py#L31-L78)). In ARKODE's `ARK324L2SA`, the two halves carry identical $`b`$, embedding $`d`$ and $`c`$.

In fact, diffrax `KenCarp3`, OrdinaryDiffEq `KenCarp3` and ARKODE `ARK324L2SA` are the same method. All three have $`\gamma = 1767732205903/4055673282236 \approx 0.4359`$ and the same first weight to sixteen digits. The name encodes the method's properties: additive RK, order 3(2), 4 stages, L-stable, stage order 2, stiffly accurate. The implicit half is an ESDIRK (explicit first stage, then a constant diagonal $`\gamma`$), and stiffly accurate means the last row of $`A^I`$ equals $`b`$.

The constant diagonal is what makes these methods cheap. Every implicit stage solves a system with the same matrix $`I - h\gamma J`$, so one Jacobian and one factorisation per step serve all stages. diffrax computes the Jacobian once, at the second stage, and reuses it ([source](https://github.com/patrick-kidger/diffrax/blob/e950aa2cf1974c02816c1542f20beabf7cca496b/diffrax/_solver/kencarp3.py#L165-L171)).

diffrax's `Sil3` (Whitaker–Kar) is a useful counterexample:

- Its two halves share $`c`$ but _not_ $`b`$: $`b^E = (1/2, -1/2, 1, 0)`$ and $`b^I = (3/8, 0, 3/8, 1/4)`$.
- Its implicit diagonal varies between stages, so it needs a new Jacobian every stage.
- Its error estimate is, in the source's own words, "just Heun".

It works, but it shows what the shared-weight, constant-diagonal design buys.

None of the libraries checks that user-supplied tables actually share $`b`$ and $`c`$:

- *ARKODE and diffrax* only check that both tables have the same number of stages. ARKODE quietly assumes shared abscissae on top of that: it uses $`c^I`$ for the stage time whenever an implicit part exists.
- *OrdinaryDiffEq* stores a separate explicit abscissa vector `ce` that defaults to `c`. Only its IMEX-SSP and BHR families override it.

### What differs between the libraries

The transferable differences are small but real:

- *Which part is which.* diffrax takes `MultiTerm(explicit_term, implicit_term)`, explicit first. OrdinaryDiffEq's `SplitODEProblem` puts the _implicit_ part first (`f1`) and the explicit part second (`f2`). ARKODE's `ARKStepCreate(fe, fi, ...)` puts the explicit part first. Get this wrong and the solver treats your diffusion explicitly.
- *Which generation of tables.* diffrax's `KenCarp4` and `KenCarp5` are the 2003 Kennedy–Carpenter tables (6 and 8 stages). ARKODE's defaults at orders 4 and 5 are the newer 2019 tables ARK4(3)7L[2]SA and ARK5(4)8L[2]SAb ([`arkode_arkstep.h`](https://github.com/LLNL/sundials/blob/312fc0f3684f27209ca9dc9249d194436eb41a7a/include/arkode/arkode_arkstep.h#L66-L73)), and ARKODE's default order is 4. OrdinaryDiffEq ships both generations (`KenCarp4`, `KenCarp47`, `KenCarp58`), plus the ARS and IMEX-SSP families.
- *Non-split use.* OrdinaryDiffEq's KenCarp methods also work as plain ESDIRK solvers on an unsplit problem: $`f_1 + f_2`$ (or just $`f`$) goes entirely into the implicit solve. diffrax enforces the `MultiTerm` structure, so KenCarp can't be called without a split.
- *Newton starting guess.* diffrax adds its own `a_predictor` tableau that extrapolates from earlier stages to start each stage's Newton iteration. OrdinaryDiffEq's KenCarp tables carry a similar Hermite-type stage predictor.

The natural next step is _multirate_ integration: the fast part gets its own smaller steps inside each slow step instead of being treated implicitly. ARKODE's `MRIStep` is the only general implementation among these libraries. It accepts any inner stepper for the fast part, with IMEX coupling for the slow part. That's a topic of its own.

## Stabilised explicit methods

The third option is to stay explicit and buy stability with stages. An $`s`$-stage explicit RK method has a polynomial stability function of degree $`s`$. If you give up on high order, you can spend all those degrees of freedom on keeping $`\lvert R(z)\rvert \le 1`$ along as much of the negative real axis as possible. The optimal polynomial is a shifted Chebyshev polynomial, and its stable interval grows like $`s^2`$ instead of the $`\sim s`$ of a classical method. The cost per step grows linearly with $`s`$, the stable step quadratically.

![Second-order RKC stability functions on the negative real axis compared with DP5](/tutorials/ode-solvers-chebyshev.png)

The figure shows the second-order RKC polynomial (Sommeijer, Shampine and Verwer) with the damping $`\varepsilon = 2/13`$ that both OrdinaryDiffEq and ARKODE use. With 10 evaluations it is stable to $`h\lambda \approx -65`$, against DP5's $`-3.3`$ with 6 evaluations. The damping keeps $`\lvert R\rvert`$ below about 0.95 between the Chebyshev extrema. Without it, $`\lvert R\rvert`$ touches 1 at every extremum, and errors at those frequencies are never damped.

This works for exactly one class of problem: stiffness from eigenvalues that are real (or nearly so) and spread along the negative axis. That's typical of diffusion, where a fine spatial grid means $`\lambda_{\min} \sim -1/\Delta x^2`$. It doesn't help with oscillatory stiffness. Nor does it help with a few isolated, huge eigenvalues, where $`s`$ would have to grow unreasonably.

### Picking the number of stages

Each step, the code needs the spectral radius $`\rho`$ to choose the smallest $`s`$ whose stable interval $`\beta(s)`$ covers $`h\rho`$. The stage-count formulas invert $`\beta(s)`$:

| Method | $`\beta(s)`$                | Stage count in OrdinaryDiffEq                                 |
| ------ | --------------------------- | ------------------------------------------------------------- |
| RKC    | $`\approx 0.65\,(s^2 - 1)`$ | $`s = \lfloor\sqrt{1 + 1.54\,h\rho}\rfloor + 1`$, no cap      |
| ROCK2  | $`\approx 0.81\,s^2`$       | $`s = \lfloor\sqrt{(1.5 + h\rho)/0.811}\rfloor + 1`$, max 200 |
| ROCK4  | $`\approx 0.35\,s^2`$       | $`s = \lfloor\sqrt{(3 + h\rho)/0.353}\rfloor + 1`$, max 152   |
| RKL2   | $`(s^2 + s - 2)/4`$         | $`s = \lceil(\sqrt{9 + 8h\rho} - 1)/2\rceil`$, odd, max 200   |

([`rkc_perform_step.jl`](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqStabilizedRK/src/rkc_perform_step.jl#L22-L25)). The RKC constant checks out against the figure: $`0.653\cdot(10^2 - 1) = 64.6`$. ARKODE uses the same RKC relation, written as $`s = \lceil\sqrt{1 + \tfrac{3/2}{1 - 2\varepsilon/15}\,h\lvert\lambda_R\rvert}\,\rceil`$, and the same RKL2 formula.

The two other families differ in the polynomial:

- *ROCK* (Abdulle) uses orthogonal polynomials composed with a small fixed polynomial, which reaches orders 2 and 4 with the intervals above.
- *RKL* (Meyer, Balsara and Aslam) uses Legendre polynomials instead of Chebyshev ones. That gives a smaller interval per stage, but stable internal stages and a simpler recursion. It's popular for super-time-stepping in astrophysics.

The caps matter. ROCK2, ROCK4, SERK and RKL all limit the _next step_ so that their maximum stage count stays stable. OrdinaryDiffEq's `RKC` has no stage cap and no such step limit, so a stiff enough problem simply gets very many stages.

### Estimating the spectral radius

This is where the two libraries that ship these methods differ most.

*OrdinaryDiffEq* estimates $`\rho`$ itself, with a nonlinear power iteration on $`f`$ ([`rkc_utils.jl`](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqStabilizedRK/src/rkc_utils.jl#L24-L78)):

- The estimate is $`\rho \approx \lVert f(u + \delta z) - f(u)\rVert/\delta`$, with $`\delta = \lVert u\rVert\sqrt{\varepsilon_{\text{mach}}}`$.
- The iteration is warm-started from the previous step's eigenvector and stops at 1 % relative change. The result is multiplied by 1.2 for safety.
- It re-estimates every 25 accepted steps, after every rejection, and at discontinuities. The 25 matches the classical RKC/ROCK Fortran codes, and the comment notes that "a stale estimate … self-corrects through a rejection".
- Users can supply their own bound instead.

*ARKODE's* `LSRKStep` refuses to run an RKC or RKL method without an eigenvalue source: either a user function or a pluggable estimator object ([source](https://github.com/LLNL/sundials/blob/312fc0f3684f27209ca9dc9249d194436eb41a7a/src/arkode/arkode_lsrkstep.c#L371-L378)). The estimator is either a power iteration or a 3-dimensional Arnoldi iteration on a difference-quotient Jacobian.

- ARKODE treats the dominant eigenvalue as _complex_ and checks whether $`(h\lambda_R, h\lambda_I)`$ lies inside the stability region. The check uses an ellipse fit whose constants the source says "were obtained empirically".
- If the point lies outside, ARKODE adds stages and, failing that, retries the step with a smaller $`h`$.
- Its safety factor is 1.01, against OrdinaryDiffEq's 1.2. ARKODE trusts the estimate more and checks the stability region explicitly instead.

## Exponential integrators

The last option goes furthest: integrate the stiff _linear_ part exactly. Write $`y' = Ay + N(y)`$. Variation of constants gives

```math
y(t_n + h) = e^{hA} y_n + \int_0^h e^{(h-\tau)A} N(y(t_n + \tau))\, d\tau,
```

and exponential Runge–Kutta methods approximate only the integral, using the functions $`\varphi_k(z) = \sum_{j \ge 0} z^j/(j+k)!`$. Exponential Euler, for instance, is

```math
y_{n+1} = y_n + h\,\varphi_1(hA)\big(Ay_n + N(y_n)\big).
```

The stiffness is absorbed into $`\varphi_k(hA)`$: there is no stability limit from $`A`$ and no Newton iteration. The cost moves into evaluating $`\varphi_k(hA)v`$.

Among the libraries here, only OrdinaryDiffEq.jl ships these methods. diffrax and SUNDIALS have none. ARKODE's "MERK" multirate tables are exponential in name only: they sub-integrate the fast part with an inner stepper instead of evaluating $`\varphi`$-functions. The design questions are still worth spelling out, because they decide whether such a method is competitive at all:

- *Where does $`A`$ come from?*
  - Classical exponential RK (Lawson, ETDRK2–4, HochOst4) needs a fixed linear operator. In OrdinaryDiffEq that's the linear part of a `SplitFunction` or, without a split, the Jacobian, with $`N(u) = f(u) - Au`$ ([source](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqExponentialRK/src/exponential_rk_perform_step.jl#L2-L5)).
  - Exponential Rosenbrock methods (`Exprb32`, `Exprb43`) and EPIRK relinearise every step with a fresh Jacobian. They are the exponential counterpart of a Rosenbrock method.
- *Dense or Krylov?*
  - For a constant $`A`$ of moderate size, OrdinaryDiffEq precomputes $`\varphi_k(hA)`$ densely once and caches it. Caching requires a `SplitFunction` and makes these methods fixed-step "due to caching" ([source](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqExponentialRK/src/exponential_rk_caches.jl#L164-L172)).
  - For large $`A`$ it uses Arnoldi (or Lanczos, if $`A`$ is Hermitian) with a fixed Krylov dimension of 30. Only the EPIRK family defaults to _adaptive_ Krylov (Niesen–Wright), which sub-steps the $`\varphi`$-evaluation to a tolerance.
- *Error control.* Most classical exponential RK methods are fixed-step. The adaptive ones (`ERK32ZB`, `ERK43ZB`, `Exprb32`, `Exprb43`, EPIRK) carry embedded estimates.

Exponential integrators win when $`A`$ is large, stiff and cheap to multiply by, and $`N`$ is mild. Semilinear PDEs are the canonical case. They lose when the stiffness lives in the nonlinearity, because then every step needs a new Jacobian _and_ a Krylov approximation. More on the $`\varphi`$-functions is in the reference page on [parallel-in-time and exponential methods](/ref/integrate/parallel-in-time-exponential).

## Summary

| Approach            | What it needs from you               | Where it wins                   | Who ships it                                                                                       |
| ------------------- | ------------------------------------ | ------------------------------- | -------------------------------------------------------------------------------------------------- |
| Detect and abort    | nothing                              | telling you to switch           | Hairer DOPRI5, DOP853                                                                              |
| Detect and switch   | a Jacobian, just in case             | stiffness that comes and goes   | ODEPACK LSODA (SciPy), OrdinaryDiffEq `AutoTsit5` and default algorithm; not CVODE, not diffrax    |
| IMEX                | the split $`f_E + f_I`$              | known stiff/non-stiff structure | ARKODE `ARKStep` (and `MRIStep`), OrdinaryDiffEq KenCarp/ARS/IMEX-SSP, diffrax KenCarp3–5 and Sil3 |
| Stabilised explicit | a spectral-radius bound or estimator | large, real, diffusive spectra  | ARKODE `LSRKStep` (RKC, RKL), OrdinaryDiffEq RKC, ROCK2/4, SERK, RKL                               |
| Exponential         | a linear operator $`A`$              | large semilinear problems       | OrdinaryDiffEq ETDRK, Exprb, EPIRK                                                                 |

What transfers from all this:

- *A good eigenvalue estimate is almost always free.* It can come from two stages at the same time point, a corrector convergence rate, or a Jacobian you already have. Nobody pays extra function evaluations just to detect stiffness. The exception is the stabilised methods, which need $`\rho`$ to pick a stage count.
- *Asymmetric hysteresis.* Entering the stiff regime is expensive and leaving it is cheap, so every switching code is slow to go stiff and quick to come back.
- *Shared weights in IMEX pairs.* Sharing $`b`$ and $`c`$, plus a constant implicit diagonal, is what makes IMEX pairs practical. The same Kennedy–Carpenter tables underpin three otherwise unrelated libraries.
- *Know where your stiffness lives.* Stabilised and exponential methods only pay off if you know which part of your spectrum is stiff. If you don't, a switching solver or a plain implicit method is the safer default.
