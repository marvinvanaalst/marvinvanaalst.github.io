---
title: 'Probabilistic programming'
description: 'Languages where models are programs: sample/observe semantics, traces, inference as effect handlers, automatic differentiation for HMC/VI, and how PPLs are designed and implemented.'
aliases:
  [
    probabilistic programming,
    PPL,
    generative model,
    sample,
    observe,
    factor,
    trace,
    execution trace,
    effect handlers for inference,
    Stan,
    PyMC,
    Turing.jl,
    Pyro,
    NumPyro,
    Gen,
    Church,
    measure-theoretic semantics,
    distribution monad,
    inference compilation,
    amortized inference,
    programmable inference,
    automatic differentiation,
    transforms,
    Jacobian adjustment,
    discrete variables,
    marginalization
  ]
tags: [programming languages, Bayesian, inference]
order: 13
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- A probabilistic program is a generative model written as code: *`sample`* draws latent variables, *`observe`* (conditioning) scores data. Inference computes the posterior over executions.
- *Two families:*
  - *Static-structure, differentiable* (Stan, PyMC, NumPyro): fixed set of continuous parameters, compiled to a log-density with gradients → NUTS/ADVI. Fast and robust.
  - *Universal* (Gen, Pyro, Turing, Church-style): arbitrary control flow, stochastic recursion, changing dimensionality → traces + more general (often slower) inference (SMC, MH on traces, programmable inference).
- *Implementation core:* interpret `sample`/`observe` with *effect handlers* (or a tracing/`log_prob` transformation); different handlers implement prior sampling, log-density evaluation, conditioning, replay, and inference algorithms.
- Constrained parameters need *bijective transforms* to $`\mathbb R^d`$ with Jacobian corrections; discrete latents are *marginalised* for gradient-based samplers.

## Decision table

| System style | Model class | Inference | Examples | Use when |
| --- | --- | --- | --- | --- |
| Compiled log-density DSL | fixed continuous parameters | NUTS, ADVI, Laplace, Pathfinder | Stan | statistical models, robust HMC |
| Embedded, graph/trace-based | fixed structure, host-language code | NUTS (JAX/PyTensor), VI, SMC | PyMC, NumPyro | Python ecosystems, GPU |
| Embedded universal | arbitrary programs | SVI, importance sampling, SMC, MCMC | Pyro, Turing.jl | deep generative models, flexible models |
| Programmable inference | universal + custom proposals/kernels | user-composed MCMC/SMC/VI | Gen | research, structured proposals |
| Simulation-based | black-box simulators | ABC / neural SBI | sbi, BayesFlow | intractable likelihoods (see [likelihood-free](/ref/bayes/likelihood-free)) |
| Probabilistic logic / discrete | finite discrete models | exact (knowledge compilation, BDDs) | ProbLog, Dice | exact discrete inference |

## Variants

### Semantics

- *Operational:* run the program, recording each `sample` (address, value, density) in a *trace*; `observe` multiplies the trace weight by the likelihood. Posterior ∝ prior trace density × weight.
- *Denotational:* programs denote (unnormalised) measures; the distribution monad gives compositional semantics; measure-theoretic and quasi-Borel space semantics handle higher-order programs.
- *Addresses:* each random choice needs a stable name (explicit in Gen/Pyro, derived from program position elsewhere) so traces can be compared and modified.

### Inference via handlers

- *Prior/forward sampling:* handle `sample` by drawing; ignore `observe` (or simulate data).
- *Log-density evaluation:* handle `sample` by looking up given values and adding $`\log p`$; `observe` adds the likelihood. Composed with AD → gradients for HMC/VI.
- *Conditioning / substitution:* fix values of named sites.
- *Importance sampling / SMC:* run forward, weight by observes; resample at observe points (for sequential models).
- *Trace MH:* propose changes to one choice, re-execute the rest, accept with the trace density ratio (handles changing structure).
- Composable handler stacks (Pyro's poutine, NumPyro's effect handlers) implement these as small interpreters (see [monads & effects](/ref/programs/monads-effects)).

### Making models gradient-friendly

- *Constraints:* sample in unconstrained space, transform (`exp`, `logit`, stick-breaking, Cholesky-of-correlation), add $`\log\lvert\det J\rvert`$.
- *Discrete latents:* enumerate and marginalise (mixtures, HMMs via forward algorithm) to keep the density continuous; or use Gibbs/SMC for them.
- *Reparametrisation:* non-centred forms for hierarchical models (see [Bayesian foundations](/ref/bayes/foundations-priors)).
- *Expensive components* (ODE solves): gradients via sensitivities/adjoints; PPLs integrate ODE solvers as differentiable primitives (see [ODE sensitivities](/ref/diff/ode-sensitivities)).

### Amortised and programmable inference

- *Guides* (Pyro) / variational families written as programs with the same addresses.
- *Inference compilation:* train a neural proposal from simulated traces, use it in importance sampling/SMC.
- *Programmable inference* (Gen): custom proposals, involutive MCMC, and SMC kernels composed with correctness checks.

## Other lenses

- *Monads and effects.* The probability monad (sample/bind) and algebraic effects with handlers are the clean foundations; inference algorithms are handlers that may resume continuations many times (see [monads & effects](/ref/programs/monads-effects)).
- *AD and compilers.* Modern PPLs are compilers: build a log-density graph, simplify, differentiate, JIT (see [automatic differentiation](/ref/diff/automatic-differentiation), [staging](/ref/programs/staging-partial-evaluation)).
- *Bayesian workflow.* PPLs make prior/posterior predictive checks, SBC and model comparison routine (see [model comparison](/ref/bayes/model-comparison)).
- *Causality.* Interventions are program transformations (replace a `sample` by a constant); counterfactuals = condition, intervene, resimulate with the same noise (see [counterfactuals](/ref/explain/counterfactuals)).

## Pitfalls & diagnostics

- *Universal ≠ efficient:* generic trace-based inference on complex models can be extremely slow; prefer static-structure models with gradients when the model allows.
- *Missing Jacobians* in custom transforms bias posteriors.
- *Stochastic control flow + HMC:* discontinuities in the log density break NUTS; restructure or marginalise.
- *Address collisions* (same name for different random choices in loops/recursion) silently corrupt inference.
- *Always validate* with simulated data (parameter recovery, SBC) before trusting results on real data.

## See also

- [Bayesian foundations & priors](/ref/bayes/foundations-priors)
- [HMC & NUTS](/ref/bayes/hmc-nuts)
- [Monads & effects](/ref/programs/monads-effects)
- [Variational inference](/ref/bayes/variational-inference)
- [Sequential Monte Carlo](/ref/bayes/sequential-monte-carlo)

*References:*

- van de Meent, Paige, Yang & Wood, "An introduction to probabilistic programming" (2018).
- Carpenter et al., "Stan: a probabilistic programming language", J. Stat. Softw. (2017).
- Cusumano-Towner et al., "Gen: a general-purpose probabilistic programming system with programmable inference", PLDI (2019).
