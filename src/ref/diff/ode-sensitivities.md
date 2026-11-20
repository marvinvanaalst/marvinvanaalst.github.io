---
title: 'ODE sensitivities'
description: 'Forward sensitivities, continuous and discrete adjoints, reversible solvers, internal numerical differentiation, steady-state, periodic-orbit and stochastic sensitivities, and which is cheapest for how many parameters.'
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
    gradient of ODE solution,
    internal numerical differentiation,
    staggered corrector,
    decoupled direct method,
    Green's function method,
    fundamental matrix,
    reversible solver,
    reversible Heun,
    jet transport,
    saltation matrix,
    second-order adjoint,
    shadowing,
    least squares shadowing,
    coupled finite differences,
    Girsanov
  ]
tags: [ODE, differentiation, adjoint, parameter estimation]
order: 3
updated: '2026-11-20'
published: true
layout: ref
---

## TL;DR

For $y' = f(y, p, t)$ with $n$ states, $p$ parameters and a loss $L$:

- **Few parameters** ($p \lesssim 50$–$100$): **forward sensitivities** (or forward-mode AD through the solver). Cost ~$(1 + p)\times$ the ODE solve, trivially gives full trajectories of $\partial y/\partial p$.
- **Many parameters, scalar loss** (neural ODEs, large models): **adjoints**. Cost ~a constant number of extra ODE solves, independent of $p$.
- Among adjoints: **interpolating/checkpointed** adjoints are robust; **backsolve** (re-integrate $y$ backwards) uses $O(1)$ memory but is unstable for stiff or dissipative systems; **discrete adjoints** (reverse-mode AD through the solver steps) give exact gradients of the computed solution. **Algebraically reversible solvers** give $O(1)$ memory without the backsolve instability, but only for non-stiff problems.
- **No time integration needed** for steady states and periodic orbits: use the implicit function theorem on the defining equations.
- **Finite differences on an adaptive solver** are limited by the tolerance; if you must, freeze the step sequence (internal numerical differentiation).
- Stiff problems: the adjoint ODE is just as stiff and must be solved with an implicit method too.

## Decision table

| Method                                  | Cost                         | Memory                     | Gradient of                          | Use when                                          |
| --------------------------------------- | ---------------------------- | -------------------------- | ------------------------------------ | ------------------------------------------------- |
| Finite differences                      | $p$ (or $2p$) extra solves   | low                        | approximate (noise ~ tol)            | quick checks, tiny $p$                            |
| Internal numerical differentiation      | $p$ extra solves, no control | low                        | discrete solution (to $O(h)$)        | black-box RHS, solver internals accessible        |
| Forward sensitivities                   | $n(1 + p)$-dim system        | $O(np)$ per time point     | continuous solution                  | $p$ small, need $S(t)$ (e.g. FIM)                 |
| Forward-mode AD through solver          | as forward sensitivities     | $O(np)$                    | discrete solution                    | $p$ small, easy to set up                         |
| Green's function method                 | $n$ adjoint-like solves      | $O(n^2)$ per time point    | continuous solution                  | many parameters, all states, all times            |
| Continuous adjoint, backsolve           | ~2–3 solves                  | $O(n)$                     | continuous (approx.)                 | non-stiff, reversible dynamics                    |
| Continuous adjoint, interpolating       | ~2–3 solves                  | dense forward solution     | continuous (approx.)                 | general, enough memory                            |
| Continuous adjoint, quadrature          | ~2–3 solves                  | dense forward + adjoint    | continuous (approx.)                 | many parameters, keeps $p$ out of the ODE system  |
| Continuous adjoint, checkpointed        | ~3–4 solves                  | checkpoints only           | continuous (approx.)                 | long horizons, large $n$                          |
| Discrete adjoint / reverse AD           | ~3–5× forward                | all steps (or checkpoints) | exact gradient of numerical solution | optimisation that needs consistent gradients      |
| Reversible solver + discrete adjoint    | ~2–3× forward                | $O(n)$                     | exact gradient of numerical solution | non-stiff, long horizons, memory-bound            |
| Taylor / jet-transport integrator       | grows with $p$ and order     | polynomial coefficients    | continuous, any order                | few parameters, higher derivatives, bounds        |
| Matrix exponential (linear ODEs)        | one $2n\times 2n$ `expm`     | $O(n^2)$                   | exact                                | $y' = A(p)y$ with constant $A$                    |
| Implicit function theorem (steady state)| one linear solve per RHS     | one factorisation          | exact at the root                    | steady states, periodic orbits, collocation       |
| Shadowing (LSS, NILSS)                  | large least-squares problem  | whole trajectory           | long-time averages                   | chaotic systems                                   |

## Variants

### Finite differences and internal numerical differentiation

- Plain finite differences re-run the adaptive solver at $p + h e_i$. Different parameters give different step sequences, so $y(p)$ is only piecewise smooth with jumps of size ~tol. The error is then $O(\text{tol}/h)$: use $h \approx \sqrt{\text{tol}}$ and expect about half the solution's digits (see [finite differences](/ref/diff/finite-differences)).
- **Internal numerical differentiation (IND, Bock):** take the step sizes, orders and Newton iteration matrices from the nominal run and replay them for the perturbed runs. The perturbed solutions are then smooth functions of $p$, and the difference quotient is a consistent derivative of the discrete scheme. It is the finite-difference counterpart of forward-mode AD through the solver.
- **Complex step through the solver** works only if the error norm and step-size controller are made complex-safe (they use `abs` and `max`); otherwise the imaginary part is discarded.

### Forward sensitivity equations

- Sensitivities $S = \partial y/\partial p \in \mathbb R^{n\times p}$ satisfy

$$
S' = \frac{\partial f}{\partial y}S + \frac{\partial f}{\partial p}, \qquad S(0) = \frac{\partial y_0}{\partial p}
$$

- Solve together with $y$. Jacobian products $J_y S$ are computed with forward-mode AD (one JVP per parameter) instead of forming $J_y$.
- **Implicit solvers** (CVODES): the sensitivity system has the same Jacobian $J_y$ as the state, so the Newton matrix is reused.
  - **Simultaneous corrector:** one Newton iteration on the joint $n(1+p)$ system. Simple, but every sensitivity has to converge before the step is accepted.
  - **Staggered direct:** converge the state first, then solve the (linear) sensitivity corrector with the same matrix.
  - **Staggered corrector:** like staggered direct, but iterates the sensitivity corrector with the possibly outdated Newton matrix instead of re-forming it. Usually the fastest.
- **Decoupled direct method** (chemical kinetics, Dunker 1984): after each accepted state step, solve the linear sensitivity step with the state's factorisation. Same idea as staggered direct, stable for stiff systems.
- Gives the **Fisher information** directly: $\mathrm{FIM} = \sum_t S(t)^\top \Sigma^{-1} S(t)$ (see [experimental design](/ref/freq/experimental-design)).

### Green's function method

- The fundamental matrix $\Phi(t, s) = \partial y(t)/\partial y(s)$ solves $\partial_t\Phi = J_y\Phi$, $\Phi(s, s) = I$. It is the sensitivity to initial conditions.
- Any parameter sensitivity follows by superposition: $S(t) = \Phi(t, 0)S(0) + \int_0^t \Phi(t, s)\,\dfrac{\partial f}{\partial p}(s)\,ds$.
- Pays off when $p \gg n$ and you need all states at all times: compute $\Phi$ once ($n$ columns), then each parameter is only a quadrature. Also gives responses to arbitrary time-varying perturbations.

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
- **Where the gradient integral goes:**
  - **Augmented:** integrate the $p$-dimensional integral as extra ODE states alongside $\lambda$. Simple, but the error control and the system size now include $p$.
  - **Quadrature adjoint:** solve $\lambda$ alone (dense output), then evaluate the integral afterwards with a high-order quadrature. Keeps the stiff solve at size $n$ when $p$ is large.
- **Stability:** with $s = T - t$, the adjoint equation becomes $d\lambda/ds = J_y^\top\lambda$, which has the same eigenvalues as the forward problem. **The adjoint ODE is as stable backwards as the state is forwards.** Only re-integrating $y$ backwards (backsolve) turns decaying modes into growing ones.

### Discrete adjoint

- Apply reverse-mode AD to the sequence of solver steps (including Newton iterations or using implicit differentiation for them).
- Exact gradient of the discretised loss, consistent with what an optimiser sees; continuous adjoints give gradients of the exact solution, which differ by $O(\text{tol})$.
- Memory: every step's state and stage values, unless checkpointed. Binomial (Revolve) checkpointing gives $O(\log N)$ memory for $N$ steps at $O(N\log N)$ recomputation (see [automatic differentiation](/ref/diff/automatic-differentiation)).

### Algebraically reversible solvers

- Schemes whose step $y_{k} \to y_{k+1}$ can be inverted in closed form, so the backward pass reconstructs $y_k$ from $y_{k+1}$ **exactly** (to rounding) instead of re-solving the ODE backwards.
- Examples: **reversible Heun** (Kidger et al. 2021, also for SDEs), **asynchronous leapfrog** (Zhuang et al. 2021), and the general construction of McCallum & Foster (2024) that turns any explicit RK method into a reversible one.
- Combined with a discrete adjoint: $O(n)$ memory like backsolve, gradients exact like a discrete adjoint, and no backsolve instability because nothing is integrated backwards.
- Limits: explicit methods only (unsuitable for stiff systems), reduced stability regions, and rounding can still grow for strongly contracting dynamics.

### Taylor-series and jet-transport integrators

- Taylor-series integrators compute the solution's Taylor coefficients in time by AD of $f$. **Jet transport** carries a truncated polynomial in $\delta p$ (and $\delta y_0$) through every arithmetic operation, so the result is $y(t; p + \delta p)$ as a polynomial.
- Gives sensitivities of any order in one run and, with interval arithmetic, rigorous error bounds (see [reliable computing](/ref/reliable/floating-point)).
- Cost grows combinatorially with the number of parameters and the order: practical for a handful of parameters (celestial mechanics, uncertainty propagation).

### Linear ODEs: matrix exponential

- For $y' = A(p)y$ with constant $A$, $y(t) = e^{tA}y_0$. The derivative in the direction $E = \partial A/\partial p_i$ is the Fréchet derivative of `expm`, read off a block matrix (Van Loan 1978):

$$
\exp\!\left(t\begin{bmatrix}A & E\\ 0 & A\end{bmatrix}\right) = \begin{bmatrix}e^{tA} & \dfrac{\partial e^{tA}}{\partial p_i}\\ 0 & e^{tA}\end{bmatrix}
$$

- Exact, no time stepping; one $2n\times 2n$ `expm` per parameter direction (see [matrix functions](/ref/solve/matrix-functions)). Extends to piecewise-constant inputs and to linearised systems.

### Implicit function theorem: steady states, periodic orbits, boundary-value problems

- **Steady state** $f(x^*, p) = 0$: $\partial x^*/\partial p = -J_y^{-1}\,\partial f/\partial p$, one factorisation for all parameters. For a scalar loss, solve $J_y^\top\lambda = \nabla_x L$ once (the steady-state adjoint). This is how metabolic control analysis computes control coefficients (see [steady states & stability](/ref/dynamics/steady-states-stability)).
- **Periodic orbits:** apply the same theorem to the shooting equations $\varphi_T(y_0, p) - y_0 = 0$ plus a phase condition. It gives sensitivities of the orbit and of the period $T$ (see [periodic orbits & chaos](/ref/dynamics/periodic-orbits-chaos)).
- **Collocation and multiple shooting:** the discretised problem is one large nonlinear system $G(z, p) = 0$; sensitivities are one sparse linear solve with the Jacobian the Newton solver already has. This is the standard route in direct-transcription optimal control (see [optimal control & MPC](/ref/optimise/optimal-control-mpc)).
- All variants break down where $J$ becomes singular: at bifurcations the sensitivity diverges.

### Events and hybrid systems

- When an event $h(y, p, t) = 0$ triggers at $t^*$ (and possibly a jump $y^+ = \Delta(y^-)$), sensitivities jump too. The **saltation matrix**

$$
S^+ = \frac{\partial\Delta}{\partial y}S^- + \Big(f^+ - \frac{\partial\Delta}{\partial y}f^-\Big)\frac{dt^*}{dp}, \qquad \frac{dt^*}{dp} = -\frac{h_y S^- + h_p}{h_y f^- + h_t}
$$

  corrects $S$ (and, transposed, $\lambda$) at $t^*$.
- Requires a transversal crossing ($h_y f + h_t \neq 0$). Grazing events make sensitivities discontinuous in $p$.

### Second-order sensitivities

- **Forward-over-forward:** differentiate the forward sensitivity system again; $O(p^2)$ extra states.
- **Forward-over-adjoint (second-order adjoint):** a forward sensitivity in direction $v$ plus an adjoint for it gives the Hessian-vector product $\nabla^2 L\,v$ at the cost of a few extra solves. Used by Newton–CG, trust-region methods and Laplace approximations (see [nonlinear least squares](/ref/optimise/nonlinear-least-squares)).
- For least-squares losses, the Gauss–Newton approximation $S^\top S$ from first-order sensitivities is usually enough.

### Chaotic systems: shadowing

- In chaotic systems $S(t)$ grows like $e^{\lambda_{\max}t}$, so the derivative of a long-time average $\langle J\rangle_T$ is not $\lim \partial\langle J\rangle_T/\partial p$ computed by any of the methods above.
- **Least-squares shadowing (LSS)** finds the nearby trajectory that stays close to the original one for all time (a least-squares problem over the whole trajectory) and differentiates that. **NILSS** reduces the cost by restricting to the unstable subspace.
- Expensive; only use when long-time statistics are the quantity of interest.

### Stochastic kinetics

For trajectories of a jump process (Gillespie SSA, see [stochastic simulation](/ref/integrate/stochastic-simulation)), $\partial\,\mathbb E[F(X_T)]/\partial p$ has no pathwise derivative. Options:

- **Finite differences with coupled paths:** simulate $X(p)$ and $X(p + h)$ with **common random numbers** (CRN), or better **common reaction paths** (CRP, one Poisson stream per reaction in the random time-change representation), or the split coupling of Anderson (2012). Biased ($O(h)$), but the variance drops from $O(1/h^2)$ to $O(1/h)$ or $O(1)$ (see [variance reduction](/ref/random/variance-reduction)).
- **Likelihood ratio (score function, Girsanov):** $\partial_p\mathbb E[F] = \mathbb E[F\,\partial_p\log\mathcal L(\text{path})]$. Unbiased, one simulation per estimate, but the variance grows with the number of reactions and with $T$ (see [stochastic gradient estimators](/ref/diff/stochastic-gradients)).
- **Pathwise via an approximation:** differentiate τ-leaping or the chemical Langevin equation. Low variance, but differentiates the approximation.

## Cost comparison

|                    | Forward sensitivities                   | Adjoint                                 |
| ------------------ | --------------------------------------- | --------------------------------------- |
| Scales with        | number of parameters $p$                | number of outputs (1 for a scalar loss) |
| Gives              | $\partial y(t)/\partial p$ at all times | $dL/dp$ only                            |
| Typical break-even | —                                       | $p \approx 50$–$100$                    |

## Other lenses

- **AD.** Forward sensitivities = forward-mode AD of the continuous problem; adjoints = reverse mode. The continuous vs discrete distinction is "differentiate then discretise" vs "discretise then differentiate" (see [automatic differentiation](/ref/diff/automatic-differentiation)). IND is the finite-difference version of "discretise then differentiate".
- **Optimal control.** The adjoint equation is the costate equation of Pontryagin's principle; gradient-based optimal control and parameter estimation use the same machinery (see [optimal control & MPC](/ref/optimise/optimal-control-mpc)).
- **Statistics.** Sensitivities scaled by parameter values and outputs ($\partial\log y/\partial\log p$) are the local sensitivity coefficients of metabolic control analysis; their Gram matrix is the FIM, whose spectrum reveals sloppy and non-identifiable directions.
- **Dynamical systems.** The forward sensitivity equation is the variational equation; its fundamental matrix gives Floquet multipliers for periodic orbits and Lyapunov exponents for chaotic ones.

## Pitfalls & diagnostics

- **Tolerances:** sensitivities need error control too (include them in the error norm), or the gradient is much less accurate than the solution.
- **Backsolve instability:** symptoms are gradients that change wildly with tolerance. Switch to interpolating or checkpointed adjoints, or a reversible solver if the problem is non-stiff. The adjoint equation itself is not the culprit; the backward re-integration of $y$ is.
- **Finite differences on adaptive solvers:** derivatives look noisy or depend on $h$ in steps. Tighten tolerances well below $h^2$, or freeze the step sequence (IND).
- **Discontinuities and events** need jump conditions for $S$ and $\lambda$ at event times; most solvers handle this only if events are declared.
- **Near bifurcations** steady-state and periodic-orbit sensitivities diverge as $J$ becomes singular; check the condition number of $J$.
- **Chaotic systems:** sensitivities grow exponentially with the horizon; long-time gradients are meaningless (use shorter windows, multiple shooting or shadowing methods).
- **Check gradients** against finite differences on a short horizon and a few parameters before trusting them in an optimiser.

## See also

- [Automatic differentiation](/ref/diff/automatic-differentiation)
- [Finite differences](/ref/diff/finite-differences)
- [Stochastic gradient estimators](/ref/diff/stochastic-gradients)
- [Integrate ODEs](/ref/integrate/ode-initial-value)
- [Fitting ODE models](/ref/optimise/fitting-odes)
- [Neural ODEs & UDEs](/ref/optimise/neural-odes-udes)
- [Steady states & stability](/ref/dynamics/steady-states-stability)
- [Global sensitivity analysis](/ref/dynamics/global-sensitivity)

**References:**

- Cao, Li, Petzold & Serban, "Adjoint sensitivity analysis for differential-algebraic equations", SIAM J. Sci. Comput. (2003).
- Hindmarsh et al., "SUNDIALS: suite of nonlinear and differential/algebraic equation solvers", ACM TOMS (2005).
- Feehery, Tolsma & Barton, "Efficient sensitivity analysis of large-scale differential-algebraic systems", Appl. Numer. Math. (1997).
- Bock, "Numerical treatment of inverse problems in chemical reaction kinetics", Springer (1981).
- Dunker, "The decoupled direct method for calculating sensitivity coefficients in chemical kinetics", J. Chem. Phys. (1984).
- Hwang, Dougherty, Rabitz & Rabitz, "The Green's function method of sensitivity analysis in chemical kinetics", J. Chem. Phys. (1978).
- Ma et al., "A comparison of automatic differentiation and continuous sensitivity analysis for derivatives of differential equation solutions" (2021).
- Kidger, Foster, Li & Lyons, "Efficient and accurate gradients for neural SDEs", NeurIPS (2021).
- McCallum & Foster, "Efficient, accurate and stable gradients for neural ODEs" (2024).
- Van Loan, "Computing integrals involving the matrix exponential", IEEE Trans. Autom. Control (1978).
- Wang, Hu & Blonigan, "Least squares shadowing sensitivity analysis of chaotic limit cycle oscillations", J. Comput. Phys. (2014).
- Anderson, "An efficient finite difference method for parameter sensitivities of continuous time Markov chains", SIAM J. Numer. Anal. (2012).
