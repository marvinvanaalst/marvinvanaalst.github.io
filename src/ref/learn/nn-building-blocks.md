---
title: 'Neural network building blocks'
description: 'MLPs, activations, initialisation, normalisation layers, residual connections, convolutions, pooling and recurrent layers: what each component does and the standard choices.'
aliases:
  [
    neural network,
    MLP,
    multilayer perceptron,
    activation function,
    ReLU,
    GELU,
    SiLU,
    swish,
    tanh,
    sigmoid,
    softmax,
    weight initialization,
    Xavier,
    Glorot,
    He initialization,
    batch normalization,
    layer normalization,
    RMSNorm,
    residual connection,
    skip connection,
    ResNet,
    convolution,
    CNN,
    pooling,
    dropout,
    embedding,
    RNN,
    LSTM,
    GRU,
    universal approximation
  ]
tags: [deep learning, neural networks, architectures]
order: 3
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *MLP:* affine layers + nonlinearities; universal approximators, the default for small vector inputs (and inside larger models).
- *Activations:* ReLU/GELU/SiLU in hidden layers; smooth activations (tanh, softplus, GELU) when derivatives of the network matter (neural ODEs, PINNs).
- *Initialisation:* He (ReLU-family) or Xavier/Glorot (tanh) to keep activations and gradients at stable scales.
- *Normalisation + residual connections* make deep networks trainable: LayerNorm/RMSNorm in transformers, BatchNorm in CNNs.
- *Architecture by data:* grids/images → convolutions; sequences → transformers (or state-space models / RNNs for long streams); sets/graphs → permutation-invariant / message-passing networks.
- *Inductive bias beats size for small data:* encode symmetries and structure.

## Decision table

| Component | Standard choice | Alternatives / when |
| --- | --- | --- |
| Hidden activation | GELU / SiLU (transformers), ReLU (CNNs) | tanh/softplus for smooth function approximation, SwiGLU in LLM MLPs |
| Output activation | identity (regression), softmax (multi-class), sigmoid (multi-label) | softplus/exp for positive outputs |
| Initialisation | He normal (ReLU), Xavier (tanh) | zero-init last layer of residual branches / UDE outputs |
| Normalisation | LayerNorm / RMSNorm (pre-norm) in transformers; BatchNorm in CNNs | GroupNorm for small batches; none for small MLPs |
| Skip connections | residual $`x + f(x)`$ | dense connections (DenseNet), U-Net skips |
| Regularisation | weight decay, dropout (0.1), data augmentation | see [regularisation](/ref/learn/regularisation-generalisation) |
| Images / grids | convolutions (3×3), ResNet blocks, or ViT for large data | U-Net for dense prediction |
| Sequences | transformer | RNN/LSTM/GRU for streaming; SSMs (S4, Mamba) for very long sequences |
| Sets | Deep Sets (sum/mean pooling) | set transformers |
| Graphs / molecules | message-passing GNNs | graph transformers, equivariant GNNs for 3D |
| Categorical inputs | embeddings | one-hot for few categories |

## Variants

### MLP

- $`h_{l+1} = \sigma(W_lh_l + b_l)`$; width and depth trade off; 2–4 hidden layers of 64–512 units cover most small problems.
- Universal approximation: one hidden layer suffices in principle; depth makes it efficient for compositional functions.

### Activations

- *ReLU* $`\max(0, x)`$: cheap, sparse, non-smooth; dead units if biases/learning rates are bad.
- *GELU / SiLU (swish)*: smooth ReLU-like; standard in modern architectures.
- *tanh / sigmoid:* saturate (vanishing gradients) in deep networks; fine for small or smooth models and gates.
- *Gated linear units* (GLU, SwiGLU): $`\sigma(xW)\odot(xV)`$, used in transformer MLPs.

### Initialisation

- Variance-preserving: $`\operatorname{Var}(W) = 2/\text{fan-in}`$ (He) for ReLU; $`2/(\text{fan-in} + \text{fan-out})`$ (Xavier) for tanh/linear.
- Residual networks: scale or zero-initialise the last layer of each residual branch so the network starts near identity.

### Normalisation

- *BatchNorm:* normalise each feature over the batch; strong in CNNs; behaves differently in training vs inference; problematic for small batches and sequences.
- *LayerNorm / RMSNorm:* normalise over features per sample; batch-independent; standard in transformers (pre-norm placement trains more stably).

### Residual connections

- $`h_{l+1} = h_l + f_l(h_l)`$: gradients flow through the identity path; enables 100+ layers.
- Equivalent to explicit Euler steps of an ODE (see [neural ODEs](/ref/optimise/neural-odes-udes)).

### Convolutions and pooling

- Local filters with shared weights: translation equivariance, few parameters. Stacking grows the receptive field; dilations grow it faster.
- Pooling / strided convolutions downsample; global average pooling before the classifier.
- 1D convolutions for signals and sequences; 3D for volumes.

### Recurrent layers

- RNN $`h_t = \sigma(Wh_{t-1} + Ux_t)`$ suffers vanishing/exploding gradients; *LSTM/GRU* gates fix much of it.
- Largely replaced by transformers for parallel training, but useful for streaming/online and small data; modern SSMs combine recurrent inference with parallel training.

## Other lenses

- *Dynamics.* ResNets ≈ ODE discretisations, RNNs ≈ discrete dynamical systems; stability of signal propagation is an eigenvalue/Lyapunov question (see [It's just like a differential equation](/blog/2026-10-just-a-differential-equation)).
- *Signal processing.* Convolutions are learned FIR filters; pooling is decimation; aliasing considerations apply (see [filtering & convolution](/ref/signals/filtering-convolution)).
- *Symmetry.* Architectures encode invariances: CNNs (translation), GNNs/Deep Sets (permutation), equivariant networks (rotations). The right symmetry reduces data needs (see [physics-informed ML](/ref/learn/physics-informed-ml)).

## Pitfalls & diagnostics

- *Exploding/vanishing activations:* monitor activation and gradient statistics per layer; fix initialisation, normalisation, learning rate.
- *Train/eval mode mix-ups* with BatchNorm and dropout.
- *Unscaled inputs/outputs:* standardise inputs; scale targets to $`O(1)`$.
- *Overfitting small data* with large networks: start small, add capacity only when the training loss can't go down.
- *Smoothness needs:* ReLU networks have zero second derivatives almost everywhere, which breaks PINN losses and some UDE uses.

## See also

- [Attention & transformers](/ref/learn/attention-transformers)
- [Training in practice](/ref/learn/training-practice)
- [Stochastic gradient methods](/ref/optimise/stochastic-gradient)
- [Regularisation & generalisation](/ref/learn/regularisation-generalisation)

*References:*

- Goodfellow, Bengio & Courville, _Deep Learning_.
- Prince, _Understanding Deep Learning_ (free online).
- He et al., "Deep residual learning for image recognition", CVPR (2016).
