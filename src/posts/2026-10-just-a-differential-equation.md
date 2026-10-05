---
title: "It's Just Like a Differential Equation"
description: 'Neural networks, transformers and causal graphs all turn out to be differential equations: discretised with forward Euler, the one integrator nobody would pick on purpose'
categories:
  - musings
date: '2026-10-05'
author: 'Marvin van Aalst'
layout: tutorials
published: true
---

# It's Just Like a Differential Equation

If you work on ODEs long enough, people from other fields start coming up to you and saying "you know, my thing is basically a differential equation". They are usually right. Then you look at how they discretise it, and it is almost always forward Euler. The one integrator every numerics course introduces mainly so it can show you how it fails.

This post collects a few of these and then asks the obvious question: if your model is secretly a beautiful continuous-time dynamical system, why simulate it with the worst integrator available?

## A quick refresher on forward Euler

Given `dx/dt = f(x)`, forward Euler takes the slope at the current point and walks along it for a step `h`:

```
x_{n+1} = x_n + h · f(x_n)
```

That's it: one function evaluation, no error control, first-order accurate (halve the step, halve the error). It is also only conditionally stable. For stiff problems the step size has to shrink to absurdity, and for oscillators it doesn't even conserve what should be conserved.

Keep that update rule in mind, because it is about to show up everywhere.

## Exhibit A: residual networks

A residual block computes

```
x_{l+1} = x_l + F(x_l, θ_l)
```

Add an `h` in front of `F`, call the layer index "time", and you have forward Euler for `dx/dt = F(x, θ(t))`. This was spelled out more or less simultaneously by [Weinan E (2017)](https://doi.org/10.1007/s40304-017-0103-z), [Haber & Ruthotto (2017)](https://arxiv.org/abs/1705.03341), who used it to reason about the stability of forward propagation, and [Lu et al. (2018)](https://arxiv.org/abs/1710.10121), who noticed that other popular architectures (PolyNet, FractalNet, RevNet) correspond to _other_ discretisation schemes. The idea then went all the way with [Neural ODEs (Chen et al., 2018)](https://arxiv.org/abs/1806.07366): drop the layers and call an actual ODE solver.

## Exhibit B: recurrent neural networks

Same trick, this time over real time instead of depth. [AntisymmetricRNN (Chang et al., 2019)](https://arxiv.org/abs/1902.09689) starts from

```
dh/dt = tanh((W − Wᵀ) h + V x(t) + b)
```

whose antisymmetric weight matrix gives eigenvalues on the imaginary axis, so the ODE neither explodes nor vanishes. The ODE is nice, so they discretise it. With explicit Euler:

```
h_t = h_{t-1} + ε · tanh((W − Wᵀ) h_{t-1} + V x_t + b)
```

Eigenvalues on the imaginary axis are exactly where forward Euler is unstable. The paper knows this and adds a small diffusion term to push them back into Euler's stability region. That is a fun detail: the architecture's flagship property needed patching because of the integrator.

Leaky-integrator [echo state networks (Jaeger et al., 2007)](https://doi.org/10.1016/j.neunet.2007.04.016) and spiking networks of leaky integrate-and-fire neurons, which [Neftci, Mostafa & Zenke (2019)](https://arxiv.org/abs/1901.09948) explicitly write as RNNs, work the same way. Forward Euler on a membrane-potential ODE gives you the "leak" in leaky.

## Exhibit C: causal graphs

This one surprised me more. A popular way to model causality in a dynamical system is a _dynamic Bayesian network_ or time-unrolled structural causal model: `X_{t+1}` gets `X_t` and its causal parents as arrows into it. [Rubenstein et al. (2018)](https://arxiv.org/abs/1608.08028) note that to get such a model from an ODE, you approximate the continuous system with the Euler method, and the only real design choice is how fine `Δ` should be. [Hansen & Sokol (2014)](https://arxiv.org/abs/1304.0217) justify their definition of interventions on SDEs by showing that it equals the limit of interventions on the structural equation models built from the Euler scheme, as the step size goes to zero. The neat result: the graph-like picture is only exact in the limit where it stops being a graph you can draw. (The more principled route from equilibria of ODEs to SCMs is [Mooij, Janzing & Schölkopf (2013)](https://arxiv.org/abs/1304.7920). It avoids discretisation entirely by only looking at steady states.)

## How bad is "bad"?

Here are two of the systems from earlier posts, solved exactly and with forward Euler:

![Forward Euler versus exact solution for Lotka-Volterra and logistic growth](/tutorials/just-a-differential-equation-euler.png)

On the left is the [Lotka-Volterra model](/blog/lotka-volterra) with its original parameters. The exact solution is a closed orbit, because the system has a conserved quantity. Forward Euler doesn't know about it, so it adds a little energy every step and spirals outward, even at `h = 0.02`. Forward Euler on the plain harmonic oscillator `x'' = −ω²x` does the same: every step multiplies the amplitude by exactly `√(1 + h²ω²)`, so the orbit always grows, no matter how small you make `h`.

On the right is logistic growth with a step that is too large. The exact solution approaches 1 monotonically. The Euler version overshoots past the carrying capacity and never settles down.

```python
def euler(f, y0, h, n):
    ys = [np.asarray(y0, float)]
    for _ in range(n):
        ys.append(ys[-1] + h * f(ys[-1]))
    return np.array(ys)
```

Five lines, and you can see why everyone picks it.

## So what's the point?

Taking a beautiful continuous-time model and simulating it with the integrator from the first lecture sounds silly. To be fair, though, there are good reasons it keeps happening:

1. **There is often no "true" ODE.** In a ResNet or RNN the weights are trained _through_ the Euler steps. The network learns whatever discrete map works, Euler error included. Asking how accurately it approximates the ODE is like asking how accurately a recipe approximates the cake. The discrete map is the model, and the ODE is a lens for analysing it.
2. **One function evaluation per step.** Every extra RK stage is another forward pass through a large network, plus memory for backprop. If accuracy against an ODE isn't the goal, Euler is the cheapest map with the right structure (a skip connection plus an update).
3. **Gradients are trivial.** Backpropagating through `x + h·f(x)` is easy. Adaptive solvers with error control bring step-size logic, rejected steps and adjoint methods into the training loop.
4. **The ODE view earns its keep anyway.** Even when nobody integrates accurately, the continuous picture tells you _what can go wrong_: exploding and vanishing dynamics, stability regions, oversmoothing. Stability analysis is something numerical analysts have been doing since long before deep learning.

The punchline, though, is that whenever someone took the "it's a differential equation" claim seriously and swapped in a better integrator, things got better, or at least the integrator stopped getting in the way. AntisymmetricRNN had to patch around Euler's stability region. Linear state-space models like [S4 (Gu et al., 2022)](https://arxiv.org/abs/2111.00396) skip Euler entirely and use exact zero-order-hold or bilinear discretisations of a linear ODE.

So "it's just like a differential equation" is usually true and usually useful. It's just that the second half of the sentence, "...and we integrate it like it's 1768", tends to get left out.
