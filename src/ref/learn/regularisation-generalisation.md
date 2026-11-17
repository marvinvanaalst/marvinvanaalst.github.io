---
title: 'Regularisation & generalisation'
description: 'Bias–variance, L1/L2 penalties and weight decay, dropout, early stopping, data augmentation, implicit regularisation and double descent: why models generalise and how to help them.'
aliases:
  [
    regularization,
    generalization,
    overfitting,
    underfitting,
    bias-variance tradeoff,
    L2 regularization,
    ridge,
    weight decay,
    L1,
    LASSO,
    dropout,
    early stopping,
    data augmentation,
    label smoothing,
    batch normalization,
    implicit regularization,
    double descent,
    benign overfitting,
    learning curves,
    validation curve,
    model capacity,
    ensembling
  ]
tags: [machine learning, statistics, generalisation]
order: 5
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Generalisation** = performance on new data from the same distribution. Measure it on held-out data, never on training data.
- **Classical view:** error = bias² + variance + noise; regularisation trades a little bias for less variance.
- **Toolbox, roughly in order of payoff:** more/better data and **augmentation**, **weight decay**, **early stopping**, dropout, ensembling, architectural inductive biases.
- **Modern overparameterised models** interpolate the training data yet generalise (double descent, implicit regularisation of SGD); still, explicit regularisation and validation-based model selection remain essential.
- **Diagnose with learning curves:** train vs validation loss as a function of data size, model size and training time.

## Decision table

| Technique                      | Mechanism                                           | Hyperparameter            | Use when                                                                  |
| ------------------------------ | --------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------- |
| L2 / ridge                     | shrink weights towards 0                            | $\lambda$ (CV)            | linear models, correlated features                                        |
| Weight decay (AdamW)           | decoupled shrinkage each step                       | ~0.01–0.1                 | neural networks (default)                                                 |
| L1 / LASSO                     | sparsity                                            | $\lambda$                 | feature selection, interpretable sparse models                            |
| Early stopping                 | stop at best validation loss                        | patience                  | iterative training (NNs, boosting)                                        |
| Dropout                        | randomly zero activations (ensemble of subnetworks) | rate 0.1–0.5              | MLPs, transformers with limited data                                      |
| Data augmentation              | encode invariances by transforming inputs           | transform set             | images, signals, any known invariance                                     |
| Label smoothing                | soften one-hot targets                              | 0.05–0.1                  | classification, overconfidence                                            |
| Ensembling                     | average independently trained models                | number of members         | best accuracy and calibrated uncertainty                                  |
| Smaller model / fewer features | reduce capacity                                     | size                      | small data                                                                |
| Priors (Bayesian)              | regularise via the prior; average over posterior    | prior scales              | small data, uncertainty needed                                            |
| Physics constraints            | restrict hypothesis space                           | architecture / loss terms | scientific ML (see [physics-informed ML](/ref/learn/physics-informed-ml)) |

## Variants

### Bias–variance decomposition

- For squared loss: $\mathbb E[(y - \hat f(x))^2] = \text{bias}^2 + \text{variance} + \sigma^2$.
- **Underfitting** (high bias): training and validation errors both high → more capacity, better features.
- **Overfitting** (high variance): low training, high validation error → more data, regularisation, smaller model.

### Penalties

- **L2:** $\min L(\theta) + \tfrac\lambda2\lVert\theta\rVert^2$; Bayesian view: Gaussian prior. For adaptive optimisers, use **decoupled weight decay** (AdamW) instead of adding L2 to the loss.
- **L1:** sparse solutions (Laplace prior); see [sparse recovery](/ref/approx/sparse-recovery).
- Penalise weights, not biases or normalisation parameters.

### Early stopping

- Track validation loss; keep the best checkpoint; stop after no improvement for $p$ epochs.
- For linear models with gradient descent, early stopping ≈ ridge with $\lambda \sim 1/(\eta t)$.

### Dropout and noise

- During training, zero each unit with probability $p$ and scale the rest by $1/(1 - p)$; at test time use the full network.
- Equivalent to training an ensemble of subnetworks; MC dropout at test time gives (rough) uncertainty estimates.
- Input noise / jitter is a Tikhonov-like regulariser on the function's smoothness.

### Data augmentation

- Apply label-preserving transformations (rotations, crops, time shifts, noise, mixup) to encode invariances; often the most effective regulariser.
- In scientific data: exploit physical symmetries (permutations of identical particles, rotations of molecules, time-reversal) rather than generic image transforms.

### Double descent and implicit regularisation

- Test error vs model size can decrease, rise near the interpolation threshold (#parameters ≈ #samples), then decrease again.
- SGD on overparameterised models prefers low-norm / flat solutions (implicit regularisation); large learning rates and small batches add noise that favours flatter minima.

## Other lenses

- **Statistics.** Regularisation = priors; cross-validation chooses the strength; ridge and LASSO are MAP estimates; effective degrees of freedom quantify complexity (see [model selection](/ref/freq/model-selection)).
- **Linear algebra.** Ridge filters small singular values ($\sigma^2/(\sigma^2 + \lambda)$), exactly Tikhonov regularisation of ill-posed problems (see [least squares](/ref/solve/least-squares)).
- **Learning theory.** Generalisation bounds via VC dimension, Rademacher complexity or PAC-Bayes; classical bounds are vacuous for deep nets, PAC-Bayes/compression bounds less so.

## Pitfalls & diagnostics

- **Validation leakage:** tuning many hyperparameters on one validation set overfits it; keep a final test set or use nested CV.
- **Distribution shift:** no regulariser protects against test data from a different distribution; validate on data that resemble deployment.
- **Regularising the wrong thing:** weight decay on normalisation layers or embeddings can hurt.
- **Calibration** degrades with overfitting even when accuracy doesn't; check reliability diagrams.
- **Learning curves** tell you whether more data or more capacity will help; plot them before scaling either.

## See also

- [Classical supervised learning](/ref/learn/classical-supervised)
- [Training in practice](/ref/learn/training-practice)
- [Resampling](/ref/freq/resampling)
- [Sparse recovery](/ref/approx/sparse-recovery)

**References:**

- Hastie, Tibshirani & Friedman, _The Elements of Statistical Learning_, ch. 3, 7.
- Belkin, Hsu, Ma & Mandal, "Reconciling modern machine-learning practice and the classical bias–variance trade-off", PNAS (2019).
- Zhang et al., "Understanding deep learning (still) requires rethinking generalization", CACM (2021).
