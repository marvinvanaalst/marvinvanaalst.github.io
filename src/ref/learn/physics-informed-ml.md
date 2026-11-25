---
title: 'Physics-informed machine learning'
description: 'PINNs, neural operators (DeepONet, FNO), equivariant networks and hard constraints: when embedding physics into learning helps and when a classical solver is better.'
aliases:
  [
    physics-informed neural networks,
    PINN,
    scientific machine learning,
    neural operator,
    DeepONet,
    Fourier neural operator,
    FNO,
    operator learning,
    equivariant neural networks,
    E(3) equivariance,
    symmetry,
    hard constraints,
    soft constraints,
    Hamiltonian neural networks,
    Lagrangian neural networks,
    inductive bias,
    hybrid modelling,
    surrogate PDE solvers
  ]
tags: [scientific ML, PDE, neural networks, symmetry]
order: 8
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *PINNs:* fit a network $`u_\theta(x, t)`$ by minimising PDE residuals at collocation points + data + boundary terms. Mesh-free and good for inverse problems with sparse data; usually *slower and less accurate than classical solvers* for forward problems.
- *Neural operators* (DeepONet, FNO): learn the map from inputs (coefficients, initial/boundary conditions) to solutions across many instances; fast surrogates once trained, need lots of solver data.
- *Symmetry and structure:* equivariant networks (rotations, permutations), Hamiltonian/Lagrangian networks (energy conservation), hard-coded conservation laws. Usually the biggest win for small data.
- *Hybrid models* (UDEs): keep known mechanisms, learn only the unknown terms (see [neural ODEs & UDEs](/ref/optimise/neural-odes-udes)).
- Hard constraints (built into the architecture) beat soft constraints (penalty terms) when you can implement them.

## Decision table

| Approach | Learns | Data needed | Strength | Weakness |
| --- | --- | --- | --- | --- |
| Classical solver (FEM/FV/spectral) | nothing | none | accurate, convergent, trusted | slow for many queries; needs full model |
| PINN (forward) | one solution $`u(x, t)`$ | equations only | mesh-free, simple to set up | slow training, failure modes, low accuracy |
| PINN (inverse) | solution + unknown parameters/fields | sparse observations | combines data and physics flexibly | balancing loss terms, identifiability |
| DeepONet | operator (input function → solution) | many solved instances | flexible inputs, any query point | data hungry |
| Fourier neural operator | operator on grids | many solved instances | fast, resolution-transferable, global kernels | regular grids, periodic-ish settings |
| Graph neural simulators | time-stepping on meshes/particles | trajectories | unstructured meshes, particles | error accumulation over rollouts |
| Equivariant networks | functions respecting symmetries | less (symmetry built in) | data efficiency, physical consistency | architectural complexity |
| Hamiltonian / Lagrangian NNs | energy function | trajectories | conserve energy by construction | only conservative systems |
| UDE / hybrid | missing terms in known equations | modest | interpretable, extrapolates better | needs partial mechanistic knowledge |

## Variants

### PINNs

- Loss $`= \lambda_r\sum_i\lVert\mathcal N[u_\theta](x_i)\rVert^2 + \lambda_b\sum_j\lVert\mathcal B[u_\theta](x_j)\rVert^2 + \lambda_d\sum_k\lVert u_\theta(x_k) - y_k\rVert^2`$, derivatives via AD.
- *Practical recipe:*
  1. Nondimensionalise; smooth activations (tanh, sine, GELU).
  2. Adam, then L-BFGS for the final accuracy.
  3. Adaptive loss weights or residual-based resampling of collocation points.
  4. Hard-code boundary conditions via output transforms where possible.
- *Failure modes:* spectral bias (slow learning of high frequencies), stiff/advective problems, causality violations in time; curricula and time-marching help.

### Neural operators

- *DeepONet:* branch net encodes the input function (sampled at sensors), trunk net encodes query coordinates; output = dot product.
- *FNO:* layers apply a learned linear transform on the lowest Fourier modes plus a pointwise linear map, then nonlinearity; global receptive field, $`O(n\log n)`$ per layer.
- Train on solver-generated pairs; test generalisation across parameters and resolutions; physics losses can augment data.

### Symmetry and conservation

- *Equivariance:* $`f(gx) = gf(x)`$ for group elements $`g`$ (rotations, translations, permutations). Implement with group convolutions, tensor-product layers (E(3)-equivariant GNNs) or canonicalisation.
- *Hamiltonian NNs:* learn $`H_\theta(q, p)`$ and integrate $`\dot q = \partial H/\partial p`$, $`\dot p = -\partial H/\partial q`$ with a symplectic integrator (see [geometric integrators](/ref/integrate/geometric-integrators)).
- *Conservation laws* by construction: predict fluxes and apply the stoichiometric matrix / divergence operator so totals are conserved exactly.

## Other lenses

- *Numerical analysis.* PINNs are a collocation method with a neural basis and a nonconvex solver; classical error analysis (consistency, stability) mostly doesn't transfer, which explains their unpredictability (see [method of lines](/ref/integrate/pdes-method-of-lines)).
- *Surrogates.* Neural operators are surrogates for parametric PDEs, competing with reduced-order models and GPs (see [surrogates](/ref/approx/surrogates), [timescale reduction](/ref/dynamics/timescale-reduction)).
- *Representation theory.* Equivariant architectures are built from group representations (spherical harmonics, Clebsch–Gordan products for SO(3)).

## Pitfalls & diagnostics

- *Benchmark against a classical solver* on cost and accuracy; many published PINN wins don't survive a fair comparison for forward problems.
- *Loss balance:* residual terms with different units/scales dominate each other; nondimensionalise and monitor each term.
- *Extrapolation* in time or parameters is where learned models fail; test explicitly.
- *Rollout error accumulation* for learned time-steppers; train with multi-step losses or noise injection.
- *Data generation cost* for operators is often the dominant expense; account for it.

## See also

- [Neural ODEs & UDEs](/ref/optimise/neural-odes-udes)
- [Method of lines](/ref/integrate/pdes-method-of-lines)
- [Surrogate models](/ref/approx/surrogates)
- [Neural network building blocks](/ref/learn/nn-building-blocks)

*References:*

- Raissi, Perdikaris & Karniadakis, "Physics-informed neural networks", J. Comput. Phys. (2019).
- Kovachki et al., "Neural operator: learning maps between function spaces", JMLR (2023).
- Karniadakis et al., "Physics-informed machine learning", Nat. Rev. Phys. (2021).
