---
title: 'Training neural networks in practice'
description: 'Losses, data pipelines, learning-rate finding, mixed precision, gradient checkpointing and accumulation, debugging recipes and what to monitor.'
aliases:
  [
    training recipe,
    loss functions,
    cross-entropy,
    MSE,
    Huber loss,
    data loading,
    normalization of inputs,
    learning rate finder,
    mixed precision,
    bf16,
    fp16,
    loss scaling,
    gradient checkpointing,
    gradient accumulation,
    distributed training,
    data parallel,
    FSDP,
    checkpointing,
    reproducibility,
    overfit a single batch,
    debugging neural networks,
    monitoring,
    hyperparameter search
  ]
tags: [deep learning, training, engineering]
order: 6
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Start simple:** small model, standardised inputs, AdamW, warmup + cosine schedule, a known-good baseline. Add complexity only when it helps.
- **Debug by overfitting one batch:** if the loss can't go to ~0 on a handful of samples, there is a bug.
- **Loss** matches the noise model: MSE (Gaussian), cross-entropy (classification), Huber (outliers), NLL with predicted variance (heteroscedastic).
- **Memory:** mixed precision (**bf16**) halves activation memory and speeds up tensor cores; **gradient checkpointing** trades compute for memory; **gradient accumulation** simulates larger batches.
- **Monitor** losses, gradient norms, learning rate and validation metrics; keep checkpoints and configurations for reproducibility.

## Decision table

| Task / issue          | Choice                                                            | Notes                                                        |
| --------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------ |
| Regression loss       | MSE; Huber / L1 with outliers; Gaussian NLL with learned variance | scale targets to $O(1)$                                      |
| Classification loss   | cross-entropy (from logits); focal loss for imbalance             | use the fused log-softmax for stability                      |
| Positive targets      | log-transform targets or Poisson/Gamma NLL                        | multiplicative noise                                         |
| Optimiser             | AdamW                                                             | see [stochastic gradient](/ref/optimise/stochastic-gradient) |
| Learning rate         | LR range test, then warmup + cosine                               | most important hyperparameter                                |
| Precision             | bf16 autocast with fp32 master weights                            | fp16 needs loss scaling                                      |
| Out of memory         | smaller batch + accumulation, checkpointing, bf16, FSDP/ZeRO      | in that order roughly                                        |
| Multi-GPU             | data parallel (DDP); FSDP for large models                        | scale lr with global batch size                              |
| Hyperparameter search | random search / Bayesian optimisation over a few key parameters   | grid search wastes budget                                    |
| Reproducibility       | seeds, deterministic kernels, logged configs and data versions    | bitwise determinism costs speed                              |

## Variants

### Data

- Standardise inputs (per feature or per channel), fit statistics on training data only.
- Shuffle each epoch; balance or weight classes if needed; augmentations on the fly.
- Keep the data pipeline faster than the model (prefetching, multiple workers); profile if GPU utilisation is low.

### The debugging recipe

1. Check data: visualise inputs and labels after preprocessing.
2. Verify the loss at initialisation (e.g. $\ln C$ for $C$ balanced classes).
3. Overfit a single batch (loss → ~0); if not, find the bug.
4. Train the simple baseline end-to-end; compare with a trivial predictor.
5. Scale up model/data; tune lr first, then regularisation.

### Learning-rate finding

- LR range test: increase lr exponentially over a few hundred steps; plot loss vs lr; choose ~1/10 of the lr where loss starts to blow up.
- Re-tune when changing batch size, architecture width or optimiser.

### Memory and speed

- **Mixed precision:** compute in bf16/fp16, keep fp32 master weights and fp32 accumulations; fp16 needs dynamic loss scaling to avoid gradient underflow.
- **Gradient checkpointing:** store only some activations, recompute the rest in the backward pass (~30% more compute for large memory savings).
- **Gradient accumulation:** sum gradients over $k$ micro-batches before stepping (effective batch $k\times$).
- **Compilation** (torch.compile, XLA/JAX jit) fuses kernels; often 1.5–2× speedups.

### Distributed training

- **Data parallel:** replicate the model, split batches, all-reduce gradients.
- **Sharded (FSDP/ZeRO):** shard parameters, gradients and optimiser states across devices for models that don't fit on one.
- **Tensor/pipeline parallelism** for very large models.

### Monitoring

- Training and validation loss (log scale), task metrics, gradient norm, learning rate, update/weight norm ratio, throughput.
- Watch for: loss spikes (lr/instability), divergence between train and validation (overfitting), flat loss (lr too low, bug, dead units).

## Other lenses

- **Optimisation.** Most training problems are optimisation problems in disguise: conditioning (normalisation helps), step size (lr), noise (batch size) (see [stochastic gradient](/ref/optimise/stochastic-gradient)).
- **Numerics.** Mixed precision is a floating-point accuracy budget; accumulate in higher precision, avoid catastrophic cancellation in losses (log-sum-exp) (see [floating-point](/ref/reliable/floating-point)).
- **AD.** Checkpointing is the same memory/recompute trade-off as in adjoint ODE sensitivities (see [automatic differentiation](/ref/diff/automatic-differentiation)).

## Pitfalls & diagnostics

- **Silent bugs:** wrong label alignment, data leakage, augmentations applied to validation data, normalisation statistics computed on all data.
- **Numerics:** `log(softmax(x))` instead of `log_softmax`, NaNs from division by zero or overflow in fp16.
- **Non-determinism:** different results between runs are normal (GPU kernels, data order); report mean ± sd over seeds for comparisons.
- **Comparing configurations unfairly:** tune each candidate's lr; equal compute budgets.
- **Checkpoint everything** needed to resume: model, optimiser state, scheduler, RNG states, data position.

## See also

- [Stochastic gradient methods](/ref/optimise/stochastic-gradient)
- [Neural network building blocks](/ref/learn/nn-building-blocks)
- [Regularisation & generalisation](/ref/learn/regularisation-generalisation)
- [Floating-point arithmetic](/ref/reliable/floating-point)

**References:**

- Karpathy, "A recipe for training neural networks" (blog, 2019).
- Google Research, "Deep learning tuning playbook" (2023).
- Micikevicius et al., "Mixed precision training", ICLR (2018).
