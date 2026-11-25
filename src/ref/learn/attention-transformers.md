---
title: 'Attention & transformers'
description: 'Scaled dot-product attention, multi-head attention, positional encodings (sinusoidal, RoPE), the transformer block, KV caching, and efficient attention for long sequences.'
aliases:
  [
    attention,
    self-attention,
    cross-attention,
    scaled dot-product attention,
    multi-head attention,
    transformer,
    encoder,
    decoder,
    causal mask,
    positional encoding,
    RoPE,
    rotary embeddings,
    ALiBi,
    KV cache,
    FlashAttention,
    linear attention,
    sparse attention,
    grouped-query attention,
    GQA,
    MQA,
    mixture of experts,
    tokenization,
    context length
  ]
tags: [deep learning, transformers, sequence models]
order: 4
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Attention:* each position forms a query, compares it with all keys, and returns a softmax-weighted average of values: $`\operatorname{softmax}(QK^\top/\sqrt{d_k})V`$.
- *Transformer block:* (pre-norm) multi-head self-attention + MLP, each with a residual connection. Stack $`L`$ blocks.
- *Cost:* $`O(n^2d)`$ time and (naively) $`O(n^2)`$ memory in sequence length $`n`$; *FlashAttention* removes the memory blow-up (exact, IO-aware tiling).
- *Positions:* attention is permutation-invariant, so add positional information: *RoPE* (rotary) is the modern default.
- *Autoregressive generation:* causal masking during training; *KV cache* at inference ($`O(n)`$ per new token, memory $`\propto`$ layers × heads × $`n`$ × $`d`$; GQA/MQA shrink it).
- Very long sequences: sliding-window/sparse attention, linear attention, or state-space models.

## Decision table

| Design choice | Options | Default / note |
| --- | --- | --- |
| Architecture | encoder-only (BERT), decoder-only (GPT), encoder–decoder (T5) | decoder-only for generation; encoder for embeddings/classification |
| Attention computation | naive, FlashAttention, memory-efficient attention | FlashAttention (exact, much less memory) |
| Positional encoding | sinusoidal, learned absolute, RoPE, ALiBi | RoPE (good extrapolation with scaling tricks) |
| Normalisation | post-norm, pre-norm, RMSNorm | pre-norm RMSNorm |
| MLP | GELU MLP, SwiGLU (×4 or ×8/3 width) | SwiGLU |
| Heads | multi-head, grouped-query, multi-query | GQA for inference efficiency |
| Long context | sliding window, sparse/block, linear attention, SSM hybrids | depends on task; check quality on long-range tasks |
| Scaling capacity | dense, mixture of experts (MoE) | MoE for more parameters at fixed compute |
| Inputs | tokens (BPE), patches (ViT), points/atoms | patchify continuous data |

## Variants

### Scaled dot-product attention

1. Project inputs $`X \in\mathbb R^{n\times d}`$: $`Q = XW_Q`$, $`K = XW_K`$, $`V = XW_V`$.
2. Scores $`S = QK^\top/\sqrt{d_k}`$ (scaling keeps the softmax out of saturation).
3. Mask (causal: $`-\infty`$ above the diagonal; padding).
4. $`A = \operatorname{softmax}(S)`$ row-wise; output $`AV`$.

- *Multi-head:* $`h`$ heads of dimension $`d/h`$ in parallel, concatenated and projected: different heads attend to different relations.
- *Cross-attention:* queries from one sequence, keys/values from another (encoder–decoder, conditioning).

### Transformer block (pre-norm)

- $`h \leftarrow h + \operatorname{Attn}(\operatorname{Norm}(h))`$
- $`h \leftarrow h + \operatorname{MLP}(\operatorname{Norm}(h))`$
- Parameters per block ≈ $`12d^2`$ (4$`d^2`$ attention, 8$`d^2`$ MLP with 4× expansion); FLOPs per token ≈ $`2\times`$ parameters (forward) + attention term $`\propto nd`$.

### Positional information

- *Sinusoidal / learned absolute:* added to embeddings; limited length generalisation.
- *RoPE:* rotate query/key pairs by position-dependent angles so scores depend on relative position; extend context with frequency scaling.
- *ALiBi:* linear distance penalty added to scores.

### Efficient attention

- *FlashAttention:* tile $`Q, K, V`$ into on-chip memory, compute softmax online (running max and sum); exact result, $`O(n)`$ memory, large speedups.
- *KV cache:* store keys/values of past tokens during generation; GQA/MQA share K/V across heads to reduce cache size; paged attention manages cache memory.
- *Approximate:* sliding window (local), sparse patterns, low-rank/kernel (linear) attention; trade exactness for $`O(n)`$ cost.

## Other lenses

- *Kernels.* Softmax attention is a normalised kernel smoother (Nadaraya–Watson) with a learned similarity; linear attention replaces it with feature-map kernels.
- *Dynamics.* A transformer layer is an explicit Euler step of interacting-particle dynamics (tokens as particles); residual stacks as ODE discretisations (see [It's just like a differential equation](/blog/2026-10-just-a-differential-equation)).
- *Associative memory.* Attention is equivalent to the update of modern Hopfield networks: retrieval of stored patterns by similarity.
- *Graphs.* Self-attention is message passing on a complete graph with learned edge weights; masks define sparser graphs.

## Pitfalls & diagnostics

- *Quadratic memory* with naive attention; use fused kernels.
- *Training instability* at scale: loss spikes from attention logit growth; use pre-norm, QK-normalisation, warmup, lower learning rates, bf16 with care.
- *Length generalisation:* models often fail beyond the training context length unless positional schemes are designed/scaled for it.
- *Small data:* transformers have weak inductive biases; for small scientific datasets, CNNs, GNNs or classical models often win.
- *Tokenisation artefacts* affect numeric and chemical sequences; check how inputs are split.

## See also

- [Neural network building blocks](/ref/learn/nn-building-blocks)
- [Training in practice](/ref/learn/training-practice)
- [Generative models](/ref/learn/generative-models)
- [Low-rank approximation](/ref/approx/low-rank)

*References:*

- Vaswani et al., "Attention is all you need", NeurIPS (2017).
- Dao et al., "FlashAttention: fast and memory-efficient exact attention with IO-awareness", NeurIPS (2022).
- Su et al., "RoFormer: enhanced transformer with rotary position embedding" (2021).
