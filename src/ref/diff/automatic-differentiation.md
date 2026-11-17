---
title: 'Automatic differentiation'
description: 'Forward mode with dual numbers, reverse mode with tapes, their costs, checkpointing, sparse Jacobians by colouring and Hessian-vector products.'
aliases:
  [
    AD,
    autodiff,
    forward mode,
    reverse mode,
    backpropagation,
    dual numbers,
    tape,
    adjoint,
    VJP,
    JVP,
    pullback,
    pushforward,
    checkpointing,
    Jacobian colouring,
    Hessian-vector product,
    forward-over-reverse,
    source transformation,
    operator overloading,
    Enzyme
  ]
tags: [differentiation, AD, backpropagation]
order: 2
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- AD applies the chain rule to the program's elementary operations: **exact derivatives** (to rounding), not approximations.
- **Forward mode** computes $Jv$ (JVP) at ~2–3× the cost of $f$. Cost of a full Jacobian: $n$ passes. Wins for $n \lesssim m$ (few inputs).
- **Reverse mode** computes $v^\top J$ (VJP) at ~3–5× the cost of $f$, independent of $n$. A full gradient of a scalar loss costs one pass. Wins for $m \ll n$ (many parameters, scalar loss: neural networks, adjoints).
- Reverse mode needs the forward intermediates: **memory** is the bottleneck. Trade memory for recomputation with checkpointing.
- Sparse Jacobians: colouring + forward mode. Hessians: forward-over-reverse Hessian-vector products, never dense Hessians if you can avoid it.

## Decision table

| Task                                        | Best mode                         | Cost (multiples of one $f$ evaluation) |
| ------------------------------------------- | --------------------------------- | -------------------------------------- |
| Gradient of $f : \mathbb R^n \to \mathbb R$ | reverse                           | ~3–5, independent of $n$               |
| Jacobian, $n \ll m$                         | forward (or chunked forward)      | ~$n$ (chunks of 8–12 duals at once)    |
| Jacobian, $m \ll n$                         | reverse                           | ~$m$                                   |
| Sparse Jacobian                             | forward + colouring               | ~number of colours                     |
| JVP $Jv$                                    | forward                           | ~2–3                                   |
| VJP $v^\top J$                              | reverse                           | ~3–5 (+ memory)                        |
| Hessian-vector product $Hv$                 | forward-over-reverse              | ~5–10                                  |
| Dense Hessian                               | $n$ HVPs, or colouring for sparse | $O(n)$ HVPs                            |
| Derivative through an iterative solver      | implicit function theorem         | one extra linear solve with $J^\top$   |

## Variants

### Forward mode

- Propagate pairs $(v, \dot v)$ through each operation: $(a, \dot a) \cdot (b, \dot b) = (ab,\ \dot a b + a\dot b)$, $\sin(a, \dot a) = (\sin a,\ \cos a\,\dot a)$.
- **Dual numbers** $a + \dot a\epsilon$ with $\epsilon^2 = 0$ implement this with operator overloading.
- Seed $\dot x = v$ to get $\dot y = Jv$. Several seeds at once (chunk mode) amortise the primal computation.
- Memory: constant overhead, no tape. Easy to nest.

### Reverse mode

1. **Forward sweep:** run the program, recording operations and intermediates (the tape / Wengert list).
2. **Reverse sweep:** set $\bar y = v$ (for a scalar loss, $\bar y = 1$) and propagate adjoints backwards: for $z = g(a, b)$, $\bar a \mathrel{+}= \bar z\,\partial g/\partial a$, $\bar b \mathrel{+}= \bar z\,\partial g/\partial b$.
3. Read off $\bar x = J^\top v$.

- Backpropagation is reverse mode on a neural network.
- **Implementation styles:** operator-overloading tapes (dynamic, easy, overhead per operation), source transformation (static, fast, harder for dynamic control flow), compiler-level AD on IR (Enzyme, after optimisation).

### Memory and checkpointing

- Storing all intermediates costs memory proportional to the number of operations (time steps × state size for ODEs, depth × activations for networks).
- **Checkpointing:** store only selected states, recompute the rest during the reverse sweep. Binomial (Revolve) schedules give $O(\log T)$ memory with $O(T\log T)$ recomputation, or $\sqrt{T}$ memory with ~2× compute.
- In deep learning: gradient (activation) checkpointing per layer block.

### Jacobians, Hessians and structure

- **Colouring:** columns of $J$ with disjoint row patterns share one forward pass (or rows sharing one reverse pass). Requires the sparsity pattern (detected by tracing).
- **Hessian-vector product:** forward mode applied to the reverse-mode gradient, $Hv = \frac{d}{d\epsilon}\nabla f(x + \epsilon v)\big|_0$. Used in Newton–CG, trust region, Lanczos on the Hessian.
- **Implicit differentiation:** for $x^*(\theta)$ defined by $F(x^*, \theta) = 0$, $\frac{dx^*}{d\theta} = -F_x^{-1}F_\theta$. Differentiate fixed points, optimisers and nonlinear solves without unrolling iterations.

### Symbolic differentiation (a lens)

- Symbolic derivatives of expressions can explode in size (expression swell) unless common subexpressions are shared; with CSE, symbolic differentiation and reverse-mode AD coincide on straight-line code.
- Symbolic approaches shine for small, closed-form models (generated Jacobians in ODE code, simplification); AD for arbitrary programs with control flow.

## Other lenses

- **Category theory.** Forward mode is a functor from functions to (function, derivative) pairs; the chain rule is functoriality. Reverse mode uses the transposed (contravariant) derivative, i.e. pullbacks composed in reverse order (see [category theory](/ref/programs/category-theory-core)).
- **Linear algebra.** The Jacobian of a composition is a product $J = J_L\cdots J_2J_1$. Forward mode multiplies right to left starting from a vector; reverse mode left to right starting from a covector. Choosing the cheapest bracketing in general is the (NP-hard) optimal Jacobian accumulation problem.
- **Adjoints.** Reverse-mode AD of an ODE solver is the discrete adjoint method; continuous adjoints differentiate the ODE first, then discretise (see [ODE sensitivities](/ref/diff/ode-sensitivities)).

## Pitfalls & diagnostics

- **Non-differentiable code paths:** branches, `abs`, `max`, rounding, integer casts give derivatives of whichever branch was taken. Fine almost everywhere, wrong at switches.
- **Mutation and aliasing** break many reverse-mode systems or force copying; prefer pure functions for the parts you differentiate.
- **Differentiating through iterative algorithms** by unrolling is expensive and differentiates the approximation; use implicit differentiation at the converged solution.
- **Adaptive solvers:** AD through an adaptive ODE solver also differentiates the step-size controller. Usually harmless, occasionally gives noisy gradients.
- **Always gradient-check** a new model against central differences on a few random directions.
- **Reverse mode on small problems** can be slower than forward mode due to tape overhead; for $n \lesssim 10$–$100$ inputs forward is often faster.

## See also

- [Finite differences](/ref/diff/finite-differences)
- [ODE sensitivities](/ref/diff/ode-sensitivities)
- [Matrix calculus](/ref/diff/matrix-calculus)
- [Neural ODEs & UDEs](/ref/optimise/neural-odes-udes)
- [Staging & partial evaluation](/ref/programs/staging-partial-evaluation)

**References:**

- Griewank & Walther, _Evaluating Derivatives_ (2nd ed.).
- Baydin et al., "Automatic differentiation in machine learning: a survey", JMLR (2018).
- Blondel & Roulet, "The elements of differentiable programming" (2024).
