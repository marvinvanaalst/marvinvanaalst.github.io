---
title: 'Generative models'
description: 'VAEs, GANs, normalising flows, diffusion and flow-matching models, and autoregressive models: how each learns a distribution and the trade-offs between them.'
aliases:
  [
    generative model,
    VAE,
    variational autoencoder,
    GAN,
    generative adversarial network,
    normalizing flow,
    RealNVP,
    Glow,
    neural spline flow,
    continuous normalizing flow,
    diffusion model,
    DDPM,
    score-based model,
    score matching,
    SDE diffusion,
    flow matching,
    rectified flow,
    autoregressive model,
    energy-based model,
    latent diffusion,
    classifier-free guidance,
    likelihood,
    sample quality
  ]
tags: [deep learning, generative models, probability]
order: 7
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Diffusion / flow matching:* best sample quality and stable training for images, molecules, proteins; sampling needs many network evaluations (reduced by better ODE solvers, distillation).
- *Normalising flows:* exact likelihoods and fast sampling; less expressive per parameter; good for density estimation in moderate dimensions and for SBI/VI.
- *VAEs:* stable, fast, with a useful latent space; samples blurrier; good for representation learning and as the encoder in latent diffusion.
- *GANs:* sharp samples, fast sampling, unstable training, mode collapse; largely superseded except for speed-critical uses.
- *Autoregressive models:* exact likelihoods, excellent for sequences (text, code); sequential sampling.
- Choose by what you need: likelihoods, sampling speed, sample quality, latent representations, conditioning.

## Decision table

| Model | Likelihood | Sampling speed | Training stability | Strengths | Weaknesses |
| --- | --- | --- | --- | --- | --- |
| VAE | lower bound (ELBO) | fast (one pass) | good | latent space, amortised inference | blurry samples, posterior collapse |
| GAN | none | fast (one pass) | poor | sharp samples | mode collapse, no likelihood |
| Normalising flow | exact | fast | good | exact density, invertible | architectural constraints, memory |
| Continuous NF (CNF) | exact (ODE trace) | ODE solve | moderate | flexible architectures | expensive training (unless flow matching) |
| Diffusion (DDPM / score SDE) | bound / ODE-exact | slow (10–1000 steps) | very good | state-of-the-art quality, conditioning | sampling cost |
| Flow matching / rectified flow | exact via ODE | moderate (few–50 steps) | very good | simple regression objective, straight paths | still multi-step |
| Autoregressive | exact | sequential, $`O(n)`$ steps | very good | sequences, exact likelihood | slow generation, ordering needed |
| Energy-based models | unnormalised | MCMC | hard | flexible | sampling and training via MCMC |

## Variants

### VAEs

- Encoder $`q_\phi(z\mid x)`$, decoder $`p_\theta(x\mid z)`$, prior $`p(z) = \mathcal N(0, I)`$.
- Maximise the ELBO $`\mathbb E_q[\log p_\theta(x\mid z)] - \mathrm{KL}(q_\phi(z\mid x)\Vert p(z))`$ with the reparametrisation trick (see [variational inference](/ref/bayes/variational-inference)).
- β-VAE weights the KL term (disentanglement vs reconstruction).

### GANs

- Generator $`G(z)`$ vs discriminator $`D(x)`$ in a min-max game; the original objective minimises JS divergence, WGAN uses Wasserstein-1 with a Lipschitz critic (see [distribution distances](/ref/info/distribution-distances)).
- Stabilisers: spectral normalisation, gradient penalties, two-timescale learning rates.

### Normalising flows

- Invertible maps $`x = f(z)`$ with tractable Jacobians: $`\log p(x) = \log p(z) - \log\lvert\det\partial f/\partial z\rvert`$.
- Coupling layers (RealNVP, Glow), autoregressive flows (MAF, IAF), neural spline flows.
- *Continuous NFs:* $`\dot z = v_\theta(z, t)`$ with $`\frac{d}{dt}\log p = -\operatorname{tr}(\partial v/\partial z)`$ (Hutchinson trace estimator) (see [neural ODEs](/ref/optimise/neural-odes-udes)).

### Diffusion models

- *Forward process:* gradually add Gaussian noise, $`x_t = \alpha_tx_0 + \sigma_t\epsilon`$.
- *Training:* a network predicts the noise $`\epsilon`$ (or the score $`\nabla\log p_t`$, or $`x_0`$) from $`x_t`$ and $`t`$ with a simple MSE loss (denoising score matching).
- *Sampling:* integrate the reverse SDE or the probability-flow ODE from noise to data; better ODE solvers (DPM-Solver, Heun) need 10–50 steps; distillation and consistency models reduce to 1–4.
- *Conditioning:* classifier-free guidance mixes conditional and unconditional predictions.

### Flow matching

- Choose a probability path between noise and data (e.g. straight lines $`x_t = (1 - t)z + tx`$) and regress a velocity field $`v_\theta(x_t, t)`$ onto the path's velocity $`x - z`$.
- Simulation-free training of CNFs; straighter paths mean fewer ODE steps at sampling time.

### Autoregressive

- Factorise $`p(x) = \prod_ip(x_i\mid x_{\lt i})`$; transformers with causal masks (see [attention & transformers](/ref/learn/attention-transformers)).

## Other lenses

- *SDEs and ODEs.* Diffusion models are reverse-time SDEs; sampling is numerical integration, so ODE solver theory (order, stiffness, step control) directly improves sampling (see [SDEs](/ref/integrate/sdes), [integrate ODEs](/ref/integrate/ode-initial-value)).
- *Optimal transport.* Rectified flow and OT-conditional flow matching approximate optimal transport maps between noise and data.
- *Bayesian inference.* Flows and diffusion models serve as flexible posteriors in VI and SBI (see [likelihood-free inference](/ref/bayes/likelihood-free)).
- *Science.* Generative models for molecules, proteins, materials and cell states; equivariant architectures encode physical symmetries (see [physics-informed ML](/ref/learn/physics-informed-ml)).

## Pitfalls & diagnostics

- *Evaluation:* likelihood, sample quality and diversity disagree; use several metrics (NLL, FID-like distances, precision/recall for distributions, domain-specific validity).
- *Memorisation:* check nearest training neighbours of samples, especially with small datasets.
- *Mode collapse* (GANs) and *posterior collapse* (VAEs with strong decoders).
- *Scientific validity:* generated molecules/structures must satisfy physical constraints; check them explicitly.
- *Compute:* diffusion training is stable but data- and compute-hungry; start from pretrained models where possible.

## See also

- [Variational inference](/ref/bayes/variational-inference)
- [SDEs](/ref/integrate/sdes)
- [Neural ODEs & UDEs](/ref/optimise/neural-odes-udes)
- [Distribution distances](/ref/info/distribution-distances)
- [Sampling from distributions](/ref/random/sampling-distributions)

*References:*

- Kingma & Welling, "Auto-encoding variational Bayes", ICLR (2014).
- Song et al., "Score-based generative modeling through stochastic differential equations", ICLR (2021).
- Lipman et al., "Flow matching for generative modeling", ICLR (2023).
