---
title: 'Stochastic gradient methods'
description: 'SGD, momentum and Nesterov, Adam and AdamW, learning-rate schedules, gradient clipping and batch size: the optimisers behind neural network training.'
aliases:
  [
    SGD,
    stochastic gradient descent,
    minibatch,
    momentum,
    heavy ball,
    Nesterov,
    Adam,
    AdamW,
    RMSProp,
    Adagrad,
    Lion,
    Shampoo,
    learning rate schedule,
    cosine schedule,
    warmup,
    one-cycle,
    gradient clipping,
    batch size,
    weight decay,
    variance reduction,
    SVRG
  ]
tags: [optimisation, machine learning, stochastic]
order: 6
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- For objectives that are sums over data, $\frac1N\sum_i\ell_i(\theta)$, use **minibatch gradients**: noisy but $N/B$ times cheaper per step.
- **Default for deep learning: AdamW** (lr ~ $10^{-3}$–$10^{-4}$, $\beta = (0.9, 0.999)$, decoupled weight decay) with **warmup + cosine decay**.
- **SGD + momentum (0.9)** with a step schedule often generalises slightly better for CNNs, but needs more tuning.
- **Learning rate** is the most important hyperparameter; then schedule, batch size, weight decay.
- **Gradient clipping** (global norm ~1) prevents occasional explosions (RNNs, transformers).
- Full-batch, smooth, small problems (e.g. fitting a modest ODE model)? Use L-BFGS instead (see [unconstrained local](/ref/optimise/unconstrained-local)).

## Decision table

| Optimiser                 | Update idea                            | Memory per parameter | Use when                                         |
| ------------------------- | -------------------------------------- | -------------------- | ------------------------------------------------ |
| SGD                       | $\theta \leftarrow \theta - \eta g$    | 0                    | convex, theory, with good schedules              |
| SGD + momentum / Nesterov | averaged direction                     | 1                    | CNNs, when tuned carefully                       |
| Adagrad                   | per-parameter $\eta/\sqrt{\sum g^2}$   | 1                    | sparse features (NLP embeddings, convex)         |
| RMSProp                   | per-parameter EMA of $g^2$             | 1                    | RNNs, RL                                         |
| Adam                      | momentum + RMSProp + bias correction   | 2                    | general default (without weight decay)           |
| AdamW                     | Adam + decoupled weight decay          | 2                    | **default** for transformers, most deep learning |
| Lion / Adafactor          | sign updates / factored second moments | 1 / below 1          | memory-limited large models                      |
| Shampoo / SOAP / Muon     | preconditioning with matrix statistics | more                 | large-scale training, faster convergence         |
| SVRG / SAGA               | variance-reduced gradients             | 1 / $N$              | finite sums, convex, need high accuracy          |
| L-BFGS (full batch)       | quasi-Newton                           | $2m$                 | small data, smooth, deterministic                |

## Variants

### SGD and momentum

- **SGD:** $\theta_{t+1} = \theta_t - \eta_tg_t$, $g_t$ = minibatch gradient. Converges (convex) with decaying $\eta_t$, $\sum\eta_t = \infty$, $\sum\eta_t^2 \lt \infty$.
- **Heavy ball:** $v_{t+1} = \mu v_t + g_t$, $\theta_{t+1} = \theta_t - \eta v_{t+1}$, $\mu \approx 0.9$. Effective step $\eta/(1 - \mu)$.
- **Nesterov:** evaluate the gradient at the look-ahead point $\theta_t - \eta\mu v_t$; better theoretical rates, small practical gain.

### Adam and AdamW

1. $m_t = \beta_1m_{t-1} + (1 - \beta_1)g_t$ (first moment).
2. $v_t = \beta_2v_{t-1} + (1 - \beta_2)g_t^2$ (second moment, elementwise).
3. Bias correction: $\hat m_t = m_t/(1 - \beta_1^t)$, $\hat v_t = v_t/(1 - \beta_2^t)$.
4. Update: $\theta_{t+1} = \theta_t - \eta\,\hat m_t/(\sqrt{\hat v_t} + \epsilon)$.
5. **AdamW:** additionally $\theta \leftarrow \theta - \eta\lambda\theta$ (decoupled), instead of adding $\lambda\theta$ to the gradient (which Adam's normalisation would distort).

- Per-parameter step sizes make it robust to scaling; $\epsilon$ (~$10^{-8}$) matters in low precision.

### Learning-rate schedules

- **Warmup:** linear increase over the first ~1–5% of steps; stabilises Adam's early second-moment estimates and large-batch training.
- **Cosine decay** to ~0 (or 10% of peak); **step decay** (×0.1 at milestones); **one-cycle**; **WSD** (warmup-stable-decay) allows continuing training.
- **LR range test:** increase lr exponentially over a short run, pick a value ~10× below where the loss diverges.

### Batch size

- Gradient noise scale: larger batches reduce variance; beyond a critical batch size, more data per step gives diminishing returns.
- **Linear scaling rule** (SGD): scale lr with batch size up to a point, with warmup. For Adam, roughly square-root scaling.
- Small batches add noise that can help generalisation; large batches use hardware efficiently.

### Clipping and stability

- **Global norm clipping:** if $\lVert g\rVert \gt c$, $g \leftarrow cg/\lVert g\rVert$. Prevents rare huge updates.
- Loss spikes: lower lr, longer warmup, check for bad data, use bf16 with fp32 master weights and loss scaling for fp16.

## Other lenses

- **Dynamics.** SGD with small steps approximates an SDE $d\theta = -\nabla L\,dt + \sqrt{\eta\Sigma}\,dW$: the noise temperature $\eta/B$ shapes which minima are found. Momentum = damped Hamiltonian dynamics.
- **Preconditioning.** Adam is a diagonal preconditioner from gradient statistics; Shampoo/K-FAC approximate the curvature with Kronecker factors; natural gradient uses the Fisher information (see [Fisher information](/ref/info/fisher-information-geometry)).
- **Convex optimisation.** For convex finite sums, variance-reduced methods (SVRG, SAGA) recover linear convergence; deep learning mostly ignores them because generalisation, not optimisation accuracy, is the goal.

## Pitfalls & diagnostics

- **Diverging loss:** lr too high (or missing warmup); exploding gradients (clip); numerical issues (log of 0, overflow in softmax).
- **Plateaus:** lr too low, bad initialisation, saturated activations, or dead ReLUs.
- **Weight decay with Adam:** use AdamW; L2 in the loss ≠ weight decay for adaptive methods.
- **Comparing optimisers** requires tuning each one's lr; a single shared lr comparison is meaningless.
- **Monitor:** training/validation loss, gradient norm, update-to-weight ratio (~$10^{-3}$ per step is a common healthy range), learning rate.

## See also

- [Unconstrained local optimisation](/ref/optimise/unconstrained-local)
- [Training in practice](/ref/learn/training-practice)
- [Neural network building blocks](/ref/learn/nn-building-blocks)
- [Neural ODEs & UDEs](/ref/optimise/neural-odes-udes)

**References:**

- Bottou, Curtis & Nocedal, "Optimization methods for large-scale machine learning", SIAM Review (2018).
- Kingma & Ba, "Adam: a method for stochastic optimization", ICLR (2015).
- Loshchilov & Hutter, "Decoupled weight decay regularization", ICLR (2019).
