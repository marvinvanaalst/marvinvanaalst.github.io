---
title: 'Inside Non-Stiff ODE Solvers'
description: 'What actually differs between DOPRI5, Tsit5, BS5, DOP853 and Adams codes once you ignore the language they are written in: which solution is kept, how the tableau weights are spent, how error is estimated, controlled and interpolated.'
categories:
  - musings
date: '2026-11-25'
author: 'Marvin van Aalst'
template: tutorials
published: true
---

# Inside Non-Stiff ODE Solvers

Ask five libraries to integrate the same non-stiff ODE and you get five answers. They agree to within tolerance, but they take different steps and spend a different number of right-hand-side evaluations. Most of that is not about Fortran vs. Python vs. Julia vs. JAX. It comes from a handful of design decisions that every solver has to make, and that each code made differently, for reasons:

- Which of the two embedded solutions do you keep?
- What do you do with the free coefficients once the order conditions are satisfied?
- How do you turn two solutions into an error estimate, and that estimate into a new step size?
- How do you get a solution between steps?
- Or do you skip Runge–Kutta altogether and reuse the past, as Adams methods do?

This post goes through those decisions one at a time and compares the answers of Hairer's Fortran codes (DOPRI5, DOP853), SciPy's `solve_ivp`, Julia's OrdinaryDiffEq.jl, JAX's diffrax and SUNDIALS (ARKODE for Runge–Kutta, CVODE for Adams), plus the older ODEPACK/VODE codes that CVODE grew out of. It is the first of three. [The second](/blog/2026-11-ode-solvers-stiff) does the same for stiff solvers, and [the third](/blog/2026-11-ode-solvers-hybrid) covers everything in between: stiffness detection, IMEX splitting, stabilised explicit and exponential methods.

I assume you know what an embedded Runge–Kutta pair and a Butcher tableau are. If not, the [reference page on initial value problems](/ref/integrate/ode-initial-value) has the short version, and [step control & events](/ref/integrate/step-control-events) covers the controller basics.

## Which solution is propagated

An embedded pair computes two solutions from the same stages,

```math
y_{n+1} = y_n + h \sum_{i=1}^{s} b_i k_i, \qquad
\hat y_{n+1} = y_n + h \sum_{i=1}^{s} \hat b_i k_i,
```

of orders $`p`$ and $`\hat p = p - 1`$. Their difference estimates the local error of the _lower_-order one. So the first design decision is almost philosophical: which of the two do you continue with?

*Fehlberg (1969) kept the lower order.* RKF45 advances with the 4th-order solution and uses the 5th-order one only to estimate its error. This is the honest choice: the estimate measures the error of exactly the number you keep, up to higher-order terms. Fehlberg accordingly tuned his free coefficients to make the _4th-order_ formula accurate. The 5th-order formula is there to be subtracted, nothing more.

*Dormand and Prince (1980) kept the higher order.* This is called _local extrapolation_. You advance with the 5th-order solution but still control the step with the 4th-order error estimate. The estimate is now pessimistic: it bounds the error of a solution you threw away, and the one you kept is (asymptotically) more accurate. You get a free order of accuracy, and the tolerance becomes a loose upper bound instead of a target. Shampine argued for this throughout the 1970s and '80s, and essentially every modern code agrees:

- *Hairer's DOPRI5* advances with the 5th-order weights $`b = (35/384, 0, 500/1113, \dots)`$ ([`dopri5.f`](http://www.unige.ch/~hairer/software.html), the `Y1 = Y + H*(A71*K1 + …)` update).
- *SciPy's `RK45`* says so in its docstring: "steps are taken using the fifth-order accurate formula (local extrapolation is done)" ([`rk.py`](https://github.com/scipy/scipy/blob/808b5af948c418c105d0734284cfc88084d24db1/scipy/integrate/_ivp/rk.py#L377-L383)). `RK23` (Bogacki–Shampine 3(2)) and `DOP853` do the same.
- *OrdinaryDiffEq.jl* (`DP5`, `Tsit5`, `BS5`, `Vern*`), *diffrax* (`Dopri5`, `Tsit5`, `Bosh3`, `Dopri8`) and *SUNDIALS ARKODE* (whose tables carry both orders, `q` and `p`, and advance with `q`) all propagate the higher order ([diffrax `Tsit5`](https://github.com/patrick-kidger/diffrax/blob/e950aa2cf1974c02816c1542f20beabf7cca496b/diffrax/_solver/tsit5.py#L159-L163)).

Once you propagate the higher order, the design target flips. It is now the _higher_-order formula whose error coefficients should be small. The lower-order one only needs to produce a trustworthy estimate. That is the real difference between RKF45 and DP5, and it explains most of the performance gap between them. The next section quantifies it.

## How the free coefficients are spent

A 6- or 7-stage explicit pair of order 5(4) has more coefficients than order conditions, so a family of methods satisfies the conditions. Everything that distinguishes DP5 from Tsit5 from BS5 is how the designers spent the remaining freedom. The main levers:

### 1. The principal error coefficient

The local error of an order-$`p`$ method expands over rooted trees $`t`$ of order $`p+1`$:

```math
\text{LTE} = h^{p+1} \sum_{|t| = p+1} \tau(t)\, F(t)(y_n) + \mathcal{O}(h^{p+2}),
\qquad
\tau(t) = \frac{1}{\sigma(t)}\Big(\Phi(t) - \frac{1}{\gamma(t)}\Big).
```

$`\Phi(t)`$ is the elementary weight the tableau produces for that tree, and $`1/\gamma(t)`$ is what the exact solution needs. The elementary differentials $`F(t)`$ belong to the problem, not the method, so the method's only handle is the vector of $`\tau(t)`$. Its 2-norm $`A^{(p+1)} = \lVert \tau^{(p+1)} \rVert_2`$ is the standard figure of merit. For a fixed accuracy, the number of steps scales like $`(A^{(p+1)})^{1/p}`$, so cost per unit accuracy is roughly

```math
E = s_{\text{eff}} \cdot \big(A^{(p+1)}\big)^{1/p},
```

with $`s_{\text{eff}}`$ the function evaluations per step after first-same-as-last (FSAL) reuse.

### 2. Simplifying assumptions, kept or dropped

Solving the order conditions by hand is only feasible if you first impose _simplifying assumptions_. These are extra linear constraints that make most order conditions automatically true. Two of them matter here:

```math
C(2):\ \sum_j a_{ij} c_j = \tfrac12 c_i^2 \quad (i \geq 3),
\qquad
D(1):\ \sum_i b_i a_{ij} = b_j (1 - c_j).
```

Dormand–Prince imposed both. Tsitouras (2011) asked what happens if you keep only the column condition $`D(1)`$ and drop the row condition $`C(2)`$. The paper's title says exactly that: _"Runge–Kutta pairs of order 5(4) satisfying only the first column simplifying assumption"_. Solving the order conditions becomes a nonlinear numerical search instead of a hand derivation, but the extra freedom buys a smaller principal error. You can see the dropped assumption directly in the coefficients. Checking $`C(2)`$ row by row gives residuals of zero for DP5 in every row it applies to, but 0.0005–0.004 for Tsit5.

### 3. FSAL

If the last stage is evaluated at $`(t_{n+1}, y_{n+1})`$, it is the first stage of the next step. DP5, Tsit5, BS3 and BS5 all do this. A 7-stage method then costs 6 evaluations per accepted step, and an 8-stage BS5 costs 7. Fehlberg's RKF45 doesn't, so it pays 6 evaluations for 6 stages. That sounds equal to DP5, but it uses its sixth stage for the error estimate and gets no free stage for the next step.

### 4. The rest of the budget

There is more you can buy: a larger stability region, so that mildly stiff transients don't force tiny steps; a good error estimator; a cheap dense output (the [dense output section](#dense-output) below). Each competes with a small $`A^{(p+1)}`$.

### The numbers

Here are the actual values, computed from the tableaux as they ship in the libraries. Tsit5 comes from diffrax, BS5 from OrdinaryDiffEq.jl, DOP853 from SciPy, and DP5 and RKF45 from the original papers. As a check, DP5 reproduces Dormand and Prince's published $`\lVert\tau^{(6)}\rVert_2 = 3.99\times10^{-4}`$.

| Pair                   | Kept order | $`s_{\text{eff}}`$ | $`A^{(p+1)}`$        | $`E`$ (rel. to DP5) | Real stability interval |
| ---------------------- | ---------- | ------------------ | -------------------- | ------------------- | ----------------------- |
| RKF45 (Fehlberg)       | 4          | 6                  | $`1.8\times10^{-3}`$ | (order 4)           | $`[-3.02, 0]`$          |
| DP5 (Dormand–Prince)   | 5          | 6                  | $`4.0\times10^{-4}`$ | 1.00                | $`[-3.31, 0]`$          |
| Tsit5 (Tsitouras)      | 5          | 6                  | $`1.4\times10^{-4}`$ | 0.81                | $`[-3.51, 0]`$          |
| BS5 (Bogacki–Shampine) | 5          | 7                  | $`2.2\times10^{-5}`$ | 0.65                | $`[-3.99, 0]`$          |
| DOP853 (Hairer)        | 8          | 12                 | $`6.3\times10^{-6}`$ | (order 8)           | $`[-6.39, 0]`$          |

Read the $`E`$ column as "function evaluations needed for the same local error, asymptotically, relative to DP5". Tsit5 gets about 20 % cheaper purely by shrinking $`A^{(6)}`$ by a factor of three at the same cost per step. BS5 goes further: it spends a whole extra stage to cut the principal error by another factor of six, and comes out about a third cheaper than DP5.

![Stability regions of BS3, DP5, Tsit5, BS5 and DOP853](/tutorials/ode-solvers-explicit-stability.png)

Three caveats keep this from being the whole story:

- *It's asymptotic.* $`E`$ describes the limit $`h \to 0`$. At loose tolerances the $`h^{p+2}`$ terms still matter. Tsit5's $`\lVert\tau^{(7)}\rVert`$ is 15 times its $`\lVert\tau^{(6)}\rVert`$, DP5's only 10 times. That's the price of the smaller leading term, and measured gains on real problems are usually more modest than 20 %.
- *The stability interval moves too.* The degree-5 Taylor polynomial, which every 5-stage order-5 method would share, is stable on $`[-3.22, 0]`$. The extra stages give DP5, Tsit5 and BS5 free coefficients in $`R(z)`$, and all three use them to stretch that interval. BS5 stretches it the most. For mildly stiff problems, where step size is limited by stability rather than accuracy, that matters more than $`A^{(6)}`$.
- *The error estimate and interpolant cost extra.* BS5's small error coefficient comes with an 8-stage tableau, two error estimators, and a dense output that needs three extra stages. That bookkeeping explains why it lives in OrdinaryDiffEq.jl but not in SciPy or diffrax. diffrax's `Dopri5` even carries a comment that the BS5 pair "is slightly more efficient" but was never implemented ([source](https://github.com/patrick-kidger/diffrax/blob/e950aa2cf1974c02816c1542f20beabf7cca496b/diffrax/_solver/dopri5.py#L50-L53)).

Where the libraries landed follows directly from this table:

- *diffrax* recommends `Tsit5` as the default and notes that `Dopri5` was the long-standing default (MATLAB's `ode45`, torchdiffeq) but that "`Tsit5` is now reckoned on being slightly more efficient overall" ([docs](https://github.com/patrick-kidger/diffrax/blob/e950aa2cf1974c02816c1542f20beabf7cca496b/docs/usage/how-to-choose-a-solver.md?plain=1#L13-L17)).
- *OrdinaryDiffEq.jl's* default algorithm starts non-stiff problems on `Tsit5` and switches to `Vern7` once `reltol` drops below $`10^{-6}`$ ([`default_alg.jl`](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqDefault/src/default_alg.jl#L146-L153)).
- *SUNDIALS ARKODE* uses the very same Tsitouras pair as its default 5th-order explicit table, both in `ERKStep` and as the explicit part of `ARKStep` ([`arkode_erkstep.h`](https://github.com/LLNL/sundials/blob/312fc0f3684f27209ca9dc9249d194436eb41a7a/include/arkode/arkode_erkstep.h#L45), [`arkode_butcher_erk.def`](https://github.com/LLNL/sundials/blob/312fc0f3684f27209ca9dc9249d194436eb41a7a/src/arkode/arkode_butcher_erk.def#L664-L668)). Its defaults at the other orders follow the same logic: Bogacki–Shampine at 3, Verner pairs at 6–9. (ARKODE's overall default order is 4, a Sofroniou–Spaletta pair, so you only get Tsit5 by asking for order 5.)
- *SciPy* stays with DP5 as `RK45`, the default of `solve_ivp`. That's the conservative choice: DP5 is the best-studied pair in existence, and 20 % isn't worth breaking anyone's results.

The 8th-order comparison has the same flavour. diffrax's `Dopri8` is Prince and Dormand's original RK8(7)13M: a 7th-order embedded estimator, 13 evaluations per step (14 stages with FSAL), $`A^{(9)} = 4.5\times10^{-6}`$. Hairer's DOP853 (also SciPy `DOP853` and OrdinaryDiffEq `DP8`) uses a different Dormand–Prince 8th-order formula from Hairer–Nørsett–Wanner's book that needs only 12 evaluations, accepts a slightly larger $`A^{(9)} = 6.3\times10^{-6}`$, and replaces the 7th-order estimator with a combined 5th- and 3rd-order one. Why that is a good trade is the subject of the next section.

## How the error is estimated

The embedded difference

```math
\text{err} = y_{n+1} - \hat y_{n+1} = h \sum_i (b_i - \hat b_i)\, k_i
```

is the default everywhere. diffrax stores $`b - \hat b`$ directly and asserts that it sums to zero. SciPy calls it `E`, ARKODE computes $`h\sum_j (b_j - d_j)F_j`$, and OrdinaryDiffEq calls it `btilde`. The interesting decisions are about what goes into $`\hat b`$ and when the stages it needs are computed.

### The FSAL stage in the estimate

DP5 and Tsit5 both put a nonzero weight on the seventh, FSAL stage in $`\hat b`$ ($`1/40`$ for DP5, $`1/66`$ for Tsit5 in OrdinaryDiffEq's `btilde`). That stage is free on an accepted step, because it is the next step's first stage. But it has to be computed _before_ the error test, so a rejected step pays for it and then throws it away. Hairer's DOPRI5, SciPy's `RK45` and OrdinaryDiffEq's `Tsit5` all evaluate $`f(t+h, y_{n+1})`$ before deciding.

DOP853 was designed the other way round. Its error estimate doesn't use the FSAL stage, and Hairer's code only evaluates it after acceptance, so a rejected step costs 11 evaluations instead of 12 (`dop853.f` lines 647, 685–690). The ports differ here:

- *SciPy* uses a generic `rk_step` that always computes the FSAL stage.
- *OrdinaryDiffEq's `DP8`* computes the FSAL stage _and_ the three dense-output stages before the accept/reject decision whenever dense output is on. A rejected step then costs four evaluations more than in Hairer's code ([source](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqHighOrderRK/src/high_order_rk_perform_step.jl#L486-L517)).

None of this changes the method. It changes what a rejection costs, which matters for problems with frequent rejections (events, discontinuities, tight tolerances near the stability limit).

### DOP853: two estimates in one

Hairer's DOP853 doesn't use a 7th-order embedded solution at all. It computes two cheaper error vectors, $`e_5`$ (the difference to a 5th-order solution) and $`e_3`$ (the difference to a 3rd-order solution), and combines them:

```math
\text{err} = \lvert h\rvert\,\frac{\lVert e_5\rVert^2}{\sqrt{\lVert e_5\rVert^2 + 0.01\,\lVert e_3\rVert^2}}.
```

The two limits explain the design:

- *Small steps.* For small $`h`$, $`e_3 \sim h^4`$ dominates the denominator and $`\text{err} \approx 10\,\lvert h\rvert\, \lVert e_5\rVert^2/\lVert e_3\rVert \sim h^{9}`$. The estimate scales like the local error of the _8th_-order solution, which is why the controller can use the exponent $`1/8`$ even though no 7th-order formula exists.
- *Large steps.* When the asymptotics haven't kicked in, $`e_3`$ is no longer much larger than $`e_5`$ and the estimate falls back to roughly $`\lvert h\rvert\,\lVert e_5\rVert`$, an honest 5th-order estimate. It can't be fooled by an accidentally small 8th-order term.

That is how DOP853 gets away with 12 evaluations: there's no need for a 7th-order embedded formula, and the estimate is still conservative where it matters. SciPy's `DOP853` reproduces the formula exactly ([`rk.py`](https://github.com/scipy/scipy/blob/808b5af948c418c105d0734284cfc88084d24db1/scipy/integrate/_ivp/rk.py#L815-L823)). OrdinaryDiffEq's `DP8` writes it with RMS norms, which is algebraically the same thing. diffrax's `Dopri8` is the 13M pair, so it uses a plain 8–7 difference instead.

### BS5: two estimators, used together

Bogacki and Shampine gave BS5 two estimates. One uses only the first six stages, so it's available before the last two stages are computed. The other uses all eight. In principle the cheap one lets a code reject a hopeless step early. OrdinaryDiffEq computes both after the last stage and takes the maximum, so it doesn't use the early-rejection opportunity ([source](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl#L449-L469)). Taking the maximum of two independent estimates is a robustness choice: each can vanish by accident on some problem, but both together rarely do.

### Multistep methods

Adams codes estimate the error from the difference between predictor and corrector, scaled by an error constant. CVODE uses $`\text{dsm} = \lVert \text{acor}\rVert \cdot t_q`$ in its weighted RMS norm. LSODA uses the max-norm of the correction divided by a tabulated constant. OrdinaryDiffEq's `VCABM` uses the last modified divided difference. In every case the estimate comes out of the corrector for free, with no extra stage.

## Error norms and tolerance scaling

Before an error vector can be compared with 1, it has to be scaled and reduced to a number. Every code here divides componentwise by $`\text{sc}_i = \text{atol}_i + \text{rtol}_i \cdot \lvert y_i\rvert`$ and then takes a norm. They differ in two details, and both are visible in practice.

*Which $`\lvert y\rvert`$?*

- *Hairer, SciPy, OrdinaryDiffEq and diffrax* use $`\max(\lvert y_{n,i}\rvert, \lvert y_{n+1,i}\rvert)`$.
- *CVODE, ARKODE and LSODA* compute the weights once per step from $`y_n`$ alone ([`cvode.c`](https://github.com/LLNL/sundials/blob/312fc0f3684f27209ca9dc9249d194436eb41a7a/src/cvode/cvode.c#L4879-L4904)).

The difference shows up for a component that starts at zero and grows quickly, such as a product species in a reaction network. SUNDIALS controls its first steps with the absolute tolerance alone, while the others already grant it the relative tolerance of its new value.

*Which norm?*

- *RMS norm:* Hairer, SciPy, OrdinaryDiffEq (`internalnorm`, overridable), diffrax (the controller's `norm`, overridable) and SUNDIALS (weighted RMS) all use one.
- *Max norm:* LSODA uses a weighted max norm everywhere, even though its comments still talk about an "r.m.s. norm" ([`lsoda.c`](https://github.com/scipy/scipy/blob/808b5af948c418c105d0734284cfc88084d24db1/scipy/integrate/src/lsoda.c#L95-L115)).

The choice matters for large systems. An RMS norm lets a few components exceed their tolerance by up to $`\sqrt{N}`$ as long as the rest are fine. A max norm doesn't. So for a 10 000-component discretised PDE, the same `rtol` means something quite different in `LSODA` than in `BDF`.

SciPy also silently raises `rtol` to at least $`100\,\varepsilon_{\text{mach}}`$ ([`common.py`](https://github.com/scipy/scipy/blob/808b5af948c418c105d0734284cfc88084d24db1/scipy/integrate/_ivp/common.py#L48-L52)). It's a sensible floor, but it means `rtol=1e-16` doesn't do what it says.

## Step-size control

With a scalar error estimate $`\text{err}`$ (accept if $`\text{err} \le 1`$), the textbook step-size update is the _I-controller_

```math
h_{\text{new}} = h \cdot \text{safety} \cdot \text{err}^{-1/(\hat p + 1)},
```

clamped to $`[\text{fac}_{\min}, \text{fac}_{\max}]\cdot h`$. A _PI-controller_ also looks at the previous error,

```math
h_{\text{new}} = h \cdot \text{safety}\cdot \text{err}_n^{-\beta_1}\, \text{err}_{n-1}^{\beta_2},
```

which damps the oscillating step sequences that a pure I-controller produces when the step is limited by stability rather than accuracy. The background is on the [step control reference page](/ref/integrate/step-control-events). What's interesting here is who actually uses which.

| Code                    | Controller (default)                         | Exponents                                   | Safety | Bounds on $`h_{\text{new}}/h`$     |
| ----------------------- | -------------------------------------------- | ------------------------------------------- | ------ | ---------------------------------- |
| Hairer DOPRI5           | PI (Lund stabilisation, $`\beta = 0.04`$)    | $`0.17`$, $`0.04`$                          | 0.9    | $`[0.2, 10]`$                      |
| Hairer DOP853           | I ($`\beta = 0`$)                            | $`1/8`$                                     | 0.9    | $`[0.333, 6]`$                     |
| SciPy RK23/RK45/DOP853  | I                                            | $`1/(\hat p+1)`$: $`1/3`$, $`1/5`$, $`1/8`$ | 0.9    | $`[0.2, 10]`$ for all three        |
| OrdinaryDiffEq          | PI, $`\beta_1 = 0.7/p`$, $`\beta_2 = 0.4/p`$ | DP5 and DP8 reproduce Hairer exactly        | 0.9    | $`[0.2, 10]`$; DP8 $`[1/3, 6]`$    |
| diffrax `PIDController` | I (`pcoeff=0`, `icoeff=1`)                   | $`1/p`$                                     | 0.9    | $`[0.2, 10]`$                      |
| ARKODE ERKStep          | I                                            | $`1/(\hat p + 1)`$                          | 0.99   | $`[0.1, 25]`$                      |
| CVODE (Adams)           | I-type, per order                            | $`1/(q+1)`$ with bias 6                     | —      | $`\le 10`$; dead band $`(0, 1.5)`$ |

Sources: [`dopri5.f`/`dop853.f`](http://www.unige.ch/~hairer/software.html) (lines 263–292 and 275–297), [SciPy `rk.py`](https://github.com/scipy/scipy/blob/808b5af948c418c105d0734284cfc88084d24db1/scipy/integrate/_ivp/rk.py#L9-L12), [OrdinaryDiffEq `alg_utils.jl`](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqCore/src/alg_utils.jl#L797-L821), [diffrax `pid.py`](https://github.com/patrick-kidger/diffrax/blob/e950aa2cf1974c02816c1542f20beabf7cca496b/diffrax/_step_size_controller/pid.py#L301-L311), [ARKODE `arkode_erkstep_io.c`](https://github.com/LLNL/sundials/blob/312fc0f3684f27209ca9dc9249d194436eb41a7a/src/arkode/arkode_erkstep_io.c#L321-L326), [CVODE `cvode_impl.h`](https://github.com/LLNL/sundials/blob/312fc0f3684f27209ca9dc9249d194436eb41a7a/src/cvode/cvode_impl.h#L86-L110).

A few things stand out:

- *PI control is the exception.* Only Hairer's DOPRI5 and OrdinaryDiffEq use it by default. diffrax documents good PI settings (`pcoeff=0.4, icoeff=0.3`, among others) but defaults to I. ARKODE ships PI, PID, Gustafsson and Söderlind controllers but defaults to I for explicit methods. Hairer himself turns it off for DOP853. The higher the order, the smoother the step sequence anyway.
- *SciPy copies Hairer's methods, not Hairer's controller.* Its `RK45` has DP5's error estimate but no Lund term. Its `DOP853` has DOP853's combined estimate but SciPy's generic bounds $`[0.2, 10]`$ instead of Hairer's $`[0.333, 6]`$. OrdinaryDiffEq went the other way and matched Hairer's constants exactly for `DP5` and `DP8`.
- *ARKODE is less conservative.* Its explicit stepper uses a safety factor of 0.99 and allows 25-fold growth, against 0.9 and 10 everywhere else. The ARKODE-wide default is 0.9, and the source comments note that "CVODE uses 1.0".
- *Exponent conventions differ.* SciPy and ARKODE use the embedded order $`\hat p`$ in $`1/(\hat p + 1)`$. diffrax divides its gains by `solver.order()` ($`p = \hat p + 1`$), which gives the same number. OrdinaryDiffEq's PI gains scale with $`1/p`$, as in the table.

### After a rejection, and after a success

Two rules sit on top of the formula, and the codes disagree on both.

*No growth right after a rejection.*

- *Hairer, SciPy and ARKODE:* the step following a rejection may not be larger than the one that was rejected. Hairer sets a `REJECT` flag, SciPy uses `factor = min(1, factor)`, and ARKODE temporarily sets its maximum growth factor to 1.
- *OrdinaryDiffEq:* no equivalent clamp in its PI or I controllers.

The rule prevents a step that was just rejected from immediately being retried at an even larger size because of a lucky error estimate on the smaller step.

*No shrinking after a success.* Three codes refuse to _reduce_ $`h`$ after an accepted step:

- *diffrax* clamps the lower bound to 1 whenever the step is kept ([`pid.py`](https://github.com/patrick-kidger/diffrax/blob/e950aa2cf1974c02816c1542f20beabf7cca496b/diffrax/_step_size_controller/pid.py#L518-L520)).
- *CVODE* replaces any proposed ratio in $`(0, 1.5)`$ with 1. After a success the step either stays the same or grows by at least 50 % ([`cvode.c`](https://github.com/LLNL/sundials/blob/312fc0f3684f27209ca9dc9249d194436eb41a7a/src/cvode/cvode.c#L3667-L3677)).
- *LSODA* keeps $`h`$ unchanged unless the proposed ratio is at least 1.1.

For a multistep code this dead band is essential, because every step-size change costs a rescaling of the history and perturbs the error estimate. For a one-step method like diffrax's it's a cheap way to avoid shrinking steps based on an error estimate that just passed.

### The first step

Hairer–Nørsett–Wanner's initial-step algorithm (Section II.4 of their book) is the most-copied piece of code in this field:

1. Guess $`h_0 = 0.01\,\lVert y_0\rVert/\lVert f_0\rVert`$.
2. Take one explicit Euler step to estimate the second derivative.
3. Choose $`h_1`$ so that the leading error term would be $`0.01`$.
4. Return $`\min(100h_0, h_1)`$.

SciPy, OrdinaryDiffEq and diffrax all implement it with the same constants. They differ from Hairer's original in one detail: they use RMS norms, while Hairer's `HINIT` uses plain 2-norms that are not divided by $`N`$. OrdinaryDiffEq's source points this out ("note the lack of norm correction", [`initdt.jl`](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqCore/src/initdt.jl#L289-L305)). For large systems the two give different first steps.

The other codes use their own algorithms:

- *SUNDIALS* starts from the geometric mean of a lower and an upper bound and iterates on a finite-difference estimate of $`\lVert\ddot y\rVert`$, aiming for $`h = \sqrt{2/\lVert\ddot y\rVert}`$. That's an order-1 criterion whatever the method's order, and the result is then halved.
- *LSODA* uses a closed-form rule with no probe step at all, $`h_0^{-2} = 1/(\text{tol}\cdot w_0^2) + \text{tol}\,\lVert f_0\rVert^2`$.
- *CVODE* (and OrdinaryDiffEq, which mirrors it) then allows the step to grow by up to $`10^4`$ after the first step, against 10 afterwards. This gives a bad initial guess a quick way out.

## Dense output

Most users never ask for the solution only at step ends. They want a time grid, an event location, or a smooth plot. All the codes here answer such requests with a _continuous extension_ inside each step, $`y(t_n + \theta h) = y_n + h\sum_i b_i(\theta)\, k_i`$ with polynomial weights $`b_i(\theta)`$. There are three ways to get one.

*1. Cubic Hermite from the endpoints.* $`y_n`$, $`y_{n+1}`$, $`f_n`$ and $`f_{n+1}`$ determine a cubic, and with FSAL all four are free. That gives a 3rd-order interpolant, which is enough for a 3rd-order method:

- SciPy's `RK23`, diffrax's `Bosh3` and OrdinaryDiffEq's `BS3` all use it.
- ARKODE uses Hermite interpolation as its default for every method, capping the degree at one below the method order. Its default order-4 explicit method therefore gets a cubic. Asking for degree 4 or 5 makes ARKODE evaluate the right-hand side once or twice more _inside every interpolation call_, with no caching between calls ([`arkode_interp.c`](https://github.com/LLNL/sundials/blob/312fc0f3684f27209ca9dc9249d194436eb41a7a/src/arkode/arkode_interp.c#L578-L589)).

*2. A continuous extension from the existing stages.* With 7 stages available, you can choose $`b_i(\theta)`$ to satisfy the order conditions at every $`\theta`$ up to some order:

- DP5's extension (Shampine 1986, "the optimum value of $`c_6`$" in SciPy's comment) and Tsitouras' own interpolant for Tsit5 are both degree-4 polynomials. Checking the continuous order conditions numerically confirms that both are _4th_-order accurate inside the step, one order below the step itself.
- diffrax implements DP5's version as a quartic through $`y_0`$, $`y_1`$, $`f_0`$, $`f_1`$ and a midpoint value. Its docstrings for `Dopri5` and `Tsit5` call this "5th order interpolation", but the polynomials are degree 4, and the conditions say order 4.

*3. Extra stages, only when needed.* For high orders, the existing stages aren't enough:

- *DOP853* adds three more stages to reach a 7th-order interpolant (again confirmed numerically). Hairer computes them only after acceptance, and in event mode only for steps that actually contain an output point.
- *SciPy* computes them lazily but for every step once `dense_output=True`.
- *OrdinaryDiffEq* uses the same lazy pattern for `BS5` (3 extra stages), `Vern6` (3), `Vern7` (6), `Vern8` (8) and `Vern9` (10). The extra stages are only computed when someone asks for a value inside that step.
- *diffrax's `Dopri8`* takes a different route: a Bogacki–Shampine interpolant over the 14 existing stages, with no extra evaluations. Its weights are degree-6 polynomials, and the continuous order conditions hold only up to order 5, three orders below the step. The docstring claims "8th order interpolation".

![Dense output error between steps for SciPy RK23, RK45 (quartic) and DOP853, and for a cubic Hermite on the same DP5 steps](/tutorials/ode-solvers-dense-output.png)

The figure shows why this matters. It runs four fixed steps of $`h = 0.5`$ on $`y'' = -y`$.

- *RK45's quartic* keeps the error between steps at the level of the error at the step ends.
- *A cubic Hermite through the same DP5 steps* is ten times worse in the middle of each step. A 5th-order method with a 3rd-order interpolant wastes its accuracy on output.
- *DOP853's 7th-order interpolant* shows the same humps one order further down.

An interpolant one order below the method is fine, because the interpolation error then scales like the local error of the step. Two or three orders below is not.

The interpolant also feeds back into the tableau. Tsitouras derived the coefficients of Tsit5 together with its interpolant, and the extra stages of DOP853 and the Verner methods are part of the published tableaux. A pair designed without dense output in mind may simply have no good continuous extension from its own stages.

Multistep codes get dense output for free. Their Nordsieck array _is_ a polynomial, and CVODE's `CVodeGetDky` and LSODA's `intdy` just evaluate it. OrdinaryDiffEq's `VCABM` is the odd one out: it keeps only $`f_n`$ and $`f_{n+1}`$ for dense output and uses the generic cubic Hermite, not its own Adams polynomial ([source](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqAdamsBashforthMoulton/src/adams_bashforth_moulton_perform_step.jl#L1490-L1497)).

## Adams methods: reuse the past instead

Everything so far spends 6–16 function evaluations per step and forgets them afterwards. Adams methods keep them. An Adams–Bashforth predictor extrapolates the polynomial through the last $`k`$ values of $`f`$, and an Adams–Moulton corrector closes the step implicitly. In PECE mode (predict, evaluate, correct, evaluate) that costs _two_ evaluations per step at any order up to 12. When $`f`$ is expensive and the solution smooth, nothing else comes close. When the step size or order changes often, or $`f`$ has discontinuities, the history becomes a liability.

The transferable design question is how to keep the history consistent when the step size changes. There are three answers in these codes.

1. *Fixed coefficients, rescaled history.* Store the history as a Nordsieck array, the scaled derivatives $`[y, hy', h^2y''/2, \dots]`$ of the interpolating polynomial. A step change by a factor $`\eta`$ then just multiplies column $`j`$ by $`\eta^j`$. The method coefficients come from fixed tables, as if the steps had been equal. LSODA does this, with coefficient tables from `cfode` and rescaling `yh[j] *= rh^j` ([`lsoda.c`](https://github.com/scipy/scipy/blob/808b5af948c418c105d0734284cfc88084d24db1/scipy/integrate/src/lsoda.c#L606-L641)). It's cheap but slightly inaccurate after a step change, which is one reason to change steps rarely.
2. *Fixed leading coefficient.* CVODE (like VODE before it) also rescales the Nordsieck array, but _recomputes_ the corrector coefficients every step from the actual history of step sizes. In its own words: $`\Lambda'(x) = c\prod(1 + x/\xi_i)`$ with $`\xi_i = (t_n - t_{n-i})/h`$ ([`cvode.c`](https://github.com/LLNL/sundials/blob/312fc0f3684f27209ca9dc9249d194436eb41a7a/src/cvode/cvode.c#L2840-L2875)). This is more robust under frequent step changes, at the price of a little bookkeeping per step. OrdinaryDiffEq's experimental `AN5` and `JVODE` follow this scheme.
3. *Variable coefficients.* Shampine and Gordon's DE/STEP (and its successor DDEABM) skips the Nordsieck array entirely. It stores modified divided differences over the actual, unequal steps, and the coefficients follow from those. OrdinaryDiffEq's `VCABM` implements this directly from Hairer–Nørsett–Wanner III.5 and calls itself "derived from Shampine's DDEABM" ([source](https://github.com/SciML/OrdinaryDiffEq.jl/blob/d8ca96c5c93914ab733810555e6c0ed71e7bf7b7/lib/OrdinaryDiffEqAdamsBashforthMoulton/src/adams_utils.jl#L1-L29)). It's the most accurate way to vary the step, and the most expensive per step.

The corrector and the order are handled differently too:

- *How the corrector is solved.*
  - LSODA starts in Adams mode with plain fixed-point iteration and forces at least two iterations, so that the convergence rate doubles as the Lipschitz estimate for its [stiffness switch](/blog/2026-11-ode-solvers-hybrid).
  - CVODE creates a _Newton_ solver by default even for Adams ([`cvode.c`](https://github.com/LLNL/sundials/blob/312fc0f3684f27209ca9dc9249d194436eb41a7a/src/cvode/cvode.c#L458-L459)). For truly non-stiff problems you have to attach the fixed-point solver yourself, or you pay for Jacobians you don't need.
  - `VCABM` does a single PECE pass and tests the error _before_ the second evaluation, so a rejected step costs just one evaluation.
- *How the order is chosen.*
  - CVODE waits $`q + 1`$ steps at each order, then estimates the achievable step at orders $`q - 1`$, $`q`$ and $`q + 1`$ and takes the best. On ties it prefers to keep the order, then to lower it, and only then to raise it.
  - LSODA does the same every $`q + 1`$ steps, but raises the order only for a gain of at least 10 %.
  - `VCABM` ramps up to order 3 over the first steps, then compares error estimates at neighbouring orders.

All of these lean towards the current order and step, for the same reason as the dead bands above: changes are expensive for a method that lives off its history.

## Summary

| Decision       | Hairer (DOPRI5/DOP853)                                | SciPy `solve_ivp`           | OrdinaryDiffEq.jl                              | diffrax                              | SUNDIALS                                               |
| -------------- | ----------------------------------------------------- | --------------------------- | ---------------------------------------------- | ------------------------------------ | ------------------------------------------------------ |
| Default pair   | DP5 / DOP853                                          | DP5 (`RK45`)                | Tsit5, Vern7 at tight tol                      | Tsit5                                | ARKODE: order 4 (Sofroniou–Spaletta); Tsit5 at order 5 |
| Kept solution  | higher order                                          | higher order                | higher order                                   | higher order                         | higher order                                           |
| Error estimate | embedded; DOP853 combined 5/3                         | same as Hairer              | same; BS5 max of two                           | embedded; `Dopri8` plain 8–7         | embedded (ARKODE); corrector (CVODE)                   |
| Norm, scaling  | RMS, $`\max(\lvert y_n\rvert, \lvert y_{n+1}\rvert)`$ | RMS, same                   | RMS (overridable), same                        | RMS (overridable), same              | WRMS, $`\lvert y_n\rvert`$ only; LSODA max norm        |
| Controller     | PI (DOPRI5), I (DOP853)                               | I, no Hairer bounds         | PI; Hairer's constants for DP5/DP8             | I by default, PID available          | I by default, many available                           |
| Dense output   | order 4 / order 7 (3 extra stages)                    | Hermite / order 4 / order 7 | own interpolants, lazy extra stages            | Hermite / order 4 / `Dopri8` order 5 | Hermite (ARKODE), Nordsieck (CVODE)                    |
| Multistep      | —                                                     | `LSODA` (Adams + BDF)       | `VCABM` (variable coefficient), `AN5`, `JVODE` | —                                    | CVODE Adams (fixed leading coefficient)                |

If one thing transfers, it's that the _method_ is the smallest part of a solver. DP5 inside SciPy, DOPRI5 and OrdinaryDiffEq is the same tableau. Yet the three codes take different first steps, control the step with different formulas and bounds, pay differently for rejections, scale tolerances differently and interpolate differently. When two libraries disagree on cost for the same "method", that's usually where the difference comes from.

The [next post](/blog/2026-11-ode-solvers-stiff) does the same for stiff solvers, where the decisions get more expensive: Jacobians, factorisations and Newton iterations.
