---
title: 'Neural ODEs & universal differential equations'
description: 'Training neural ODEs and UDEs: adjoint choice, stiffness during training, multiple shooting and minibatching, and recovering equations with SINDy or symbolic regression.'
aliases:
  [
    neural ODE,
    NODE,
    universal differential equations,
    UDE,
    scientific machine learning,
    SciML,
    hybrid models,
    grey-box models,
    augmented neural ODE,
    latent ODE,
    ODE-RNN,
    symbolic regression,
    SINDy,
    continuous normalizing flows,
    FFJORD,
    training stiffness,
    collocation training
  ]
tags: [scientific ML, ODE, neural networks, hybrid modelling]
order: 8
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Neural ODE:** $x' = f_\theta(x, t)$ with $f_\theta$ a neural network; trained by differentiating through an ODE solver.
- **UDE:** keep known mechanisms and learn only the unknown part, $x' = f_{\text{known}}(x, p) + \text{NN}_\theta(x)$. Far less data needed, interpretable, extrapolates better.
- **Gradients:** many parameters and a scalar loss ⇒ adjoints (interpolating/checkpointed) or reverse-mode AD through the solver. Small networks (≲ 100 params) ⇒ forward mode is often faster.
- **Training difficulty** is mostly a shooting problem: long horizons and oscillations give rugged losses. Use **multiple shooting**, growing time windows, or collocation pre-training.
- **Recover equations:** after training a UDE, apply **SINDy** or symbolic regression to the learned term to get closed-form expressions.

## Decision table

| Choice            | Options                                                                                      | Recommendation                                                                                               |
| ----------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Model form        | pure NODE, UDE (hybrid), latent ODE (encoder + ODE in latent space)                          | UDE whenever some mechanism is known                                                                         |
| Network size      | tiny MLPs (2–3 layers, 16–64 units) to large                                                 | small; smooth activations (tanh, softplus, GELU)                                                             |
| Solver            | explicit RK (Tsit5), stiff (Rodas, KenCarp), fixed step                                      | adaptive explicit unless stiff; stiff needs implicit + AD-compatible Jacobians                               |
| Gradient method   | forward mode, reverse AD through solver, interpolating adjoint, backsolve adjoint            | forward for few params; interpolating adjoint or discrete adjoint for many; avoid backsolve on stiff systems |
| Loss              | trajectory MSE, multiple shooting with continuity penalty, collocation (derivative matching) | collocation to pre-train, shooting to refine                                                                 |
| Optimiser         | Adam then L-BFGS                                                                             | Adam for robustness, finish with (L-)BFGS                                                                    |
| Equation recovery | SINDy, symbolic regression (PySR, SymbolicRegression.jl)                                     | SINDy on NN outputs over the data range                                                                      |

## Variants

### Neural ODE training

1. Forward: solve $x' = f_\theta(x, t)$ from $x_0$, evaluate the loss at observation times.
2. Backward: adjoint ODE or reverse AD for $\partial L/\partial\theta$ (see [ODE sensitivities](/ref/diff/ode-sensitivities)).
3. Update $\theta$ with Adam; optionally finish with BFGS.

- **Augmented NODEs:** add extra dimensions so trajectories can cross in the visible space (plain NODEs are homeomorphisms).
- **Latent ODEs:** encode irregular time series into an initial latent state, evolve with a NODE, decode.

### UDEs

- Embed a network where the model is uncertain: an unknown rate law, a missing interaction, a forcing term.
- Constrain with structure: positivity (softplus outputs), conservation laws (design the NN to output fluxes multiplied by the stoichiometry), known symmetries.
- Regularise the NN term (L2, smoothness) so the known part explains what it can.

### Making training work

- **Collocation pre-training:** smooth the data (splines, GPs), estimate derivatives, fit $f_\theta$ to match derivatives: no ODE solves, quick initialisation.
- **Multiple shooting:** split the trajectory into windows with free initial states and a continuity penalty; avoids exploding gradients and local minima from long integrations.
- **Curriculum:** train on short horizons first, extend gradually.
- **Minibatching:** sample trajectories or time windows per step.

### Stiffness during training

- Random initial networks can produce stiff or exploding dynamics; adaptive explicit solvers then take tiny steps and training stalls.
- Remedies: stiff solvers, regularise the Jacobian or solver step counts (penalise NFE), bound NN outputs, initialise the NN output layer near zero (so the UDE starts as the known model).

### Equation recovery

1. Train the UDE to fit data well.
2. Evaluate the learned term $\text{NN}_\theta(x)$ on states within the data range.
3. Fit a sparse combination of candidate terms with SINDy, or run symbolic regression.
4. Replace the NN by the recovered expression and refit parameters; validate by simulation.

## Other lenses

- **Dynamics.** ResNets are explicit Euler discretisations of NODEs; continuous-depth models allow adaptive "depth" via the solver (see [integrate ODEs](/ref/integrate/ode-initial-value)).
- **Generative modelling.** Continuous normalising flows use NODEs with the instantaneous change of variables $\frac{d\log p}{dt} = -\operatorname{tr}(\partial f/\partial x)$; flow matching trains them without simulation (see [generative models](/ref/learn/generative-models)).
- **Statistics.** A UDE is a semi-parametric model: mechanistic parameters plus a nonparametric component; identifiability of mechanistic parameters can be lost if the NN can mimic them.

## Pitfalls & diagnostics

- **NN absorbs known physics:** the hybrid fits but mechanistic parameters become meaningless; constrain the NN's inputs/outputs or regularise.
- **Extrapolation:** learned terms are only valid within the state space covered by data; check predictions outside it.
- **Adjoint instability** with backsolve on stiff/dissipative systems: noisy gradients. Use interpolating or checkpointed adjoints.
- **Slow training** usually means stiffness or too many solver steps; monitor the number of function evaluations per forward pass.
- **Report baselines:** the mechanistic model alone and a pure black-box model; the hybrid should beat both on held-out data.

## See also

- [Fitting ODE models](/ref/optimise/fitting-odes)
- [ODE sensitivities](/ref/diff/ode-sensitivities)
- [Sparse recovery](/ref/approx/sparse-recovery) (SINDy)
- [Physics-informed ML](/ref/learn/physics-informed-ml)
- [Stochastic gradient methods](/ref/optimise/stochastic-gradient)

**References:**

- Chen, Rubanova, Bettencourt & Duvenaud, "Neural ordinary differential equations", NeurIPS (2018).
- Rackauckas et al., "Universal differential equations for scientific machine learning" (2020).
- Kidger, "On neural differential equations", PhD thesis (2022).
