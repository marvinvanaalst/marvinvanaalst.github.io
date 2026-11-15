# Reference (`/ref`) backlog

Task-oriented quick-refresher pages on numerical methods and surrounding maths. This is not a textbook: the focus is on variants and on **which one is more efficient when**, with theory only where it helps. This file is the single source of truth for what's left. It is not deployed.

## Conventions

**Location:** `src/ref/<section>/<slug>.md`. Section ids and titles are in `src/lib/ref.ts`, and an unknown folder fails the build. A page goes live once its file exists with `published: true`.

**Frontmatter:**

```yaml
---
title: 'Solve Ax = b'
description: 'One-line TL;DR shown under the title and in listings.'
aliases: [LU, Cholesky, GMRES] # technique names, so the /ref filter finds this page
tags: [linear algebra]
order: 1 # position within the section
updated: '2026-11-15'
published: true
layout: ref
---
```

**Page template** (skip empty blocks):

1. `## TL;DR`: 2–3 lines, the default choice.
2. `## Decision table`: variant | use when | cost (flops / memory / evals) | order or convergence | pitfalls.
3. `## Variants`: per method, the core formula or update rule, a short Julia snippet and its costs.
4. `## Other lenses`: the same idea in LA / probability / optimisation / CT language, kept short.
5. `## Pitfalls & diagnostics`
6. `## See also` + 1–2 canonical references.

No library or "in practice" section: pages stay library-agnostic.

**Syntax gotchas** (mdsvex → Svelte):

- Math: `$…$` inline and `$$…$$` display, on their own lines with blank lines around them. It is rendered by KaTeX at build time, and invalid TeX fails the build. Commutative diagrams use `\begin{CD}`.
- Display math inside list items renders inline. Use `$\displaystyle …$` there, or move it out of the list.
- No raw `{ }` or `<` in prose (Svelte parses them). Put them inside math, inline code or code blocks, or use `≲`/`≤`.
- No `|` inside math in tables (it breaks the table). Use `\lvert x\rvert`.
- Cross-references to pages that don't exist yet stay plain text, not links (strict prerender fails on dead links). Turn them into links once the target exists.
- Pseudocode in Julia.

## Pages

★ = pilot page (template review pending). `[x]` = published.

### 1. reliable: Compute reliably

- [ ] floating-point: IEEE 754 formats, ulp / machine eps, rounding modes, cancellation, FMA, subnormals, NaN semantics
- [ ] summation-reductions: pairwise, Kahan / Neumaier, compensated dot products, reproducible reductions
- [ ] conditioning-stability: condition numbers, forward vs backward error, scaling / equilibration
- [ ] interval-affine-arithmetic: interval arithmetic, dependency problem, affine arithmetic, validated numerics

### 2. solve: Solve equations

- [x] ★ linear-systems: LU, Cholesky, LDLᵀ, QR, banded / Thomas, Woodbury, mixed-precision refinement
- [ ] sparse-matrices: COO/CSR/CSC, fill-in, AMD / nested dissection / RCM, symbolic vs numeric factorisation
- [ ] krylov-preconditioning: CG, MINRES, GMRES (restarts), BiCGSTAB; Jacobi, ILU, AMG, block preconditioners
- [ ] least-squares: normal equations vs QR vs SVD, Tikhonov, weighted / generalised / total LS, NNLS
- [ ] nonlinear-roots: bisection, Brent, Newton, secant, Broyden, Newton–Krylov, globalisation, homotopy
- [ ] eigenproblems: power / inverse iteration, Rayleigh quotient, QR algorithm, Lanczos / Arnoldi, shift-invert, generalised
- [ ] matrix-functions: expm (Padé scaling-squaring), Krylov expm·v, sqrtm / logm, φ-functions
- [ ] multigrid-domain-decomposition: geometric vs algebraic multigrid (smoothers, V/W/F-cycles), as solver vs preconditioner, Schwarz / Schur-complement DD
- [ ] hierarchical-matrices-fmm: fast multipole, H-/H²-matrices, HSS, low-rank off-diagonal blocks, O(n log n) kernel matrices (GPs, N-body, BEM)

### 3. diff: Differentiate

- [ ] finite-differences: forward / central / higher order, optimal step size, complex step, Richardson
- [ ] automatic-differentiation: dual numbers, reverse tape, checkpointing, Jacobian colouring, HVPs; symbolic + CSE as a lens
- [ ] ode-sensitivities: forward sensitivities, continuous vs discrete adjoint, interpolating / backsolve / checkpointed, cost vs #params
- [ ] matrix-calculus: numerator vs denominator layout, common identities (trace, logdet, inverse)

### 4. integrate: Integrate & evolve

- [ ] quadrature: Newton–Cotes, Gauss, Clenshaw–Curtis, adaptive (Gauss–Kronrod), singularities, high-dim → MC / QMC
- [x] ★ ode-initial-value: explicit RK, Adams, BDF, Radau, Rosenbrock, SDIRK, stiffness, stability functions
- [ ] step-control-events: embedded error estimates, PI controllers, initial step, dense output, event location
- [ ] daes: index, mass matrices, consistent initialisation, index reduction (Pantelides), BDF / Radau
- [ ] geometric-integrators: Verlet / leapfrog, symplectic Euler, splitting (Strang), energy drift, Lie-group methods
- [ ] sdes: Euler–Maruyama, Milstein, SRI / SRA, Itô vs Stratonovich, strong vs weak order
- [ ] pdes-method-of-lines: FD / FV / FE / spectral, CFL, upwinding, IMEX, operator splitting
- [ ] stochastic-simulation: Gillespie SSA, next reaction, τ-leaping, chemical Langevin; DDE note
- [ ] parallel-in-time-exponential: parareal, PFASST, exponential integrators (ETD, Krylov φ-functions), IMEX-RK

### 5. dynamics: Analyse dynamics

- [ ] steady-states-stability: root solve vs time-stepping, Jacobian eigenvalues, Lyapunov functions
- [ ] bifurcation-continuation: natural vs pseudo-arclength, fold / Hopf / branch-point detection
- [ ] timescale-reduction: QSSA, Tikhonov, slow manifolds, POD, balanced truncation, DMD
- [ ] global-sensitivity: local vs global, Morris, Sobol indices (Saltelli), FAST, MCA control coefficients
- [ ] stoichiometric-analysis: null space, conservation laws, elementary modes, FBA / FVA as LP
- [ ] periodic-orbits-chaos: shooting for limit cycles, Floquet multipliers, Poincaré maps, Lyapunov exponents
- [ ] stochastic-processes: discrete / continuous-time Markov chains, Poisson processes, chemical master equation, Fokker–Planck, stationary distributions, mixing times
- [ ] reaction-network-theory: CRNT, deficiency zero / one theorems, multistationarity, toric steady states, Gröbner bases / symbolic steady-state solving

### 6. approx: Approximate functions

- [ ] interpolation: barycentric Lagrange, Runge, Chebyshev nodes, cubic / B-splines, PCHIP / Akima
- [ ] chebyshev-spectral: Chebyshev series, Clenshaw, spectral convergence, Chebfun-style workflows
- [ ] fourier-fft: DFT / FFT, aliasing, windowing, FFT convolution, real FFTs, NUFFT
- [ ] low-rank: SVD, truncated / randomised SVD, PCA, NMF, CP / Tucker / tensor train
- [ ] gaussian-processes: kernels, Cholesky cost, inducing points / sparse GPs, hyperparameter fitting
- [ ] surrogates: RBF, polynomial chaos, NN surrogates, active learning
- [ ] high-dimensional-numerics: curse of dimensionality, sparse grids (Smolyak), tensor-train methods, when MC / QMC beats grids
- [ ] sparse-recovery: compressed sensing, LASSO / basis pursuit, OMP, RIP / coherence, proximal solvers (ISTA / FISTA), link to SINDy

### 7. optimise: Optimise & fit

- [ ] unconstrained-local: GD, Newton, BFGS / L-BFGS, nonlinear CG; Wolfe line search vs trust region
- [ ] nonlinear-least-squares: Gauss–Newton, Levenberg–Marquardt, TRF, residual Jacobians
- [ ] constrained: LP (simplex / IPM), QP, SQP, augmented Lagrangian, barrier, projection, bound transforms
- [ ] convex-duality: Lagrangian, KKT, duality gap, proximal operators, ADMM
- [ ] global-derivative-free: Nelder–Mead, CMA-ES, DE, PSO, simulated annealing, multistart, Bayesian optimisation
- [ ] stochastic-gradient: SGD, momentum / Nesterov, Adam / AdamW, schedules, clipping, batch size
- [ ] fitting-odes: single vs multiple shooting, collocation, log-parameters, scaling, local minima
- [ ] neural-odes-udes: NODE training, adjoint choice, UDEs, SINDy / symbolic regression, stiffness in training
- [ ] optimal-control-mpc: Pontryagin, direct vs indirect methods, direct transcription / collocation, multiple shooting, MPC
- [ ] mixed-integer-sat-smt: branch & bound, cutting planes, MILP formulation tricks (big-M, indicator constraints), SAT (CDCL), SMT
- [ ] manifold-optimisation: Riemannian gradients, retractions, Stiefel / SPD / SO(3), Lie-group integrators, reparametrisation vs manifold methods

### 8. random: Generate randomness

- [ ] prngs: LCG, xorshift / xoshiro, PCG, MT19937, Philox / Threefry, streams / splitting, JAX-style keys
- [ ] sampling-distributions: inverse CDF, Box–Muller / Ziggurat, rejection, alias method, Marsaglia–Tsang gamma, MVN via Cholesky, Gumbel-max
- [ ] quasi-monte-carlo: Sobol, Halton, LHS, scrambling, discrepancy, error rates
- [ ] variance-reduction: antithetic, control variates, importance sampling, stratification, MLMC

### 9. bayes: Infer (Bayesian)

- [ ] foundations-priors: Bayes' rule, conjugacy, prior choice, hierarchical models, non-centred parametrisation
- [ ] mcmc: Metropolis–Hastings, Gibbs, adaptive MH, slice, ensemble samplers; R-hat / ESS
- [ ] hmc-nuts: leapfrog, mass matrix, step-size adaptation, divergences
- [ ] sequential-monte-carlo: SMC samplers, tempering, PMCMC
- [ ] variational-inference: ELBO, mean-field, ADVI, normalising flows, reparametrisation trick; Laplace / INLA
- [ ] likelihood-free: ABC rejection, SMC-ABC, synthetic likelihood, neural SBI (NPE / NLE)
- [ ] model-comparison: evidence, Bayes factors, nested sampling, thermodynamic integration, WAIC, PSIS-LOO

### 10. freq: Infer (frequentist)

- [ ] estimation-mle: MLE, Fisher information, Cramér–Rao, sandwich estimator, M-estimators
- [ ] confidence-intervals: Wald, profile likelihood, bootstrap percentile / BCa
- [ ] identifiability: structural vs practical, FIM rank, profile likelihood, sloppiness
- [ ] hypothesis-tests: t, χ², LRT / Wald / score, nonparametric, permutation, power
- [ ] multiple-testing: Bonferroni, Holm, Benjamini–Hochberg FDR
- [ ] regression-glms: OLS assumptions, GLMs / IRLS, mixed models, robust regression
- [ ] resampling: bootstrap, jackknife, cross-validation
- [ ] model-selection: AIC, BIC, nested LRT, CV
- [ ] experimental-design: FIM-based D / A / E-optimality, Bayesian OED (expected information gain), sequential / adaptive design
- [ ] conformal-prediction: split / full / jackknife+ conformal, exchangeability, conformalised quantile regression, coverage guarantees

### 11. signals: Estimate state & signals

- [ ] kalman-filters: KF, EKF, UKF, EnKF, RTS smoother
- [ ] particle-filters: bootstrap PF, resampling schemes, degeneracy
- [ ] filtering-convolution: FIR / IIR, Savitzky–Golay, convolution theorem
- [ ] spectral-estimation: periodogram, Welch, Lomb–Scargle, autocorrelation
- [ ] time-series-models: AR / ARMA, state-space form

### 12. info: Information & distances

- [ ] entropy-kl-mi: entropy, cross-entropy, KL, JS, mutual information estimators
- [ ] fisher-information-geometry: FIM, natural gradient, Jeffreys prior
- [ ] distribution-distances: TV, Hellinger, Wasserstein / OT / Sinkhorn, MMD, two-sample tests

### 13. learn: Learn from data

- [ ] classical-supervised: linear / logistic, kernels / SVM, trees / RF / GBM, kNN
- [ ] unsupervised: k-means, GMM / EM, hierarchical, DBSCAN, t-SNE / UMAP
- [ ] nn-building-blocks: MLP, activations, initialisation, normalisation, residuals, convolutions
- [ ] attention-transformers: attention, positional encodings, KV cache, complexity
- [ ] regularisation-generalisation: L1 / L2, dropout, early stopping, bias–variance, double descent
- [ ] training-practice: losses, LR finding, mixed precision, gradient checkpointing
- [ ] generative-models: VAE, GAN, flows, diffusion
- [ ] physics-informed-ml: PINNs, DeepONet / FNO, equivariance

### 14. explain: Explain & reason causally

- [ ] feature-attribution: permutation importance, PDP / ICE, SHAP variants, LIME, integrated gradients, saliency
- [ ] causal-graphs-scms: DAGs, d-separation, SCMs, do-operator, backdoor / frontdoor, IVs
- [ ] causal-effect-estimation: adjustment, IPW, doubly robust, matching, DiD, RD
- [ ] counterfactuals: abduction–action–prediction, counterfactual explanations
- [ ] causal-discovery: PC, GES, NOTEARS, LiNGAM

### 15. space: Transform space

- [ ] vectors-tensors-einsum: index notation, einsum, broadcasting, strides / memory layout, contraction order
- [ ] rotations: matrices, Euler angles / gimbal lock, axis-angle, quaternions, slerp, SO(3) exp / log
- [ ] homogeneous-projections: affine 4×4, view / perspective matrices, normal matrix, reverse-Z
- [ ] geometry-primitives: barycentric coordinates, intersections, robust orientation predicates

### 16. programs: Structure programs

- [x] ★ category-theory-core: categories, functors, natural transformations, universal properties, adjunctions
- [ ] monads-effects: monads, Kleisli, applicatives, free monads, algebraic effects / handlers
- [ ] algebraic-data-types: F-algebras, initial algebras, catamorphisms / recursion schemes, codata
- [ ] algebraic-structures: monoids, semirings, tropical semiring (parallel reductions, path problems)
- [ ] lambda-calculus-types: untyped / STLC, System F, Curry–Howard, variance, dependent types (brief)
- [ ] type-inference-unification: unification, Hindley–Milner / algorithm W, bidirectional typing
- [ ] lattices-fixpoints-dataflow: posets, Kleene / Tarski, dataflow analysis, abstract interpretation, widening
- [ ] term-rewriting-egraphs: confluence, termination, e-graphs / equality saturation
- [ ] ssa-ir: SSA, dominators, φ-nodes, CPS correspondence
- [ ] parsing-automata: regex → NFA → DFA, CFGs, LL / LR, Pratt, PEG, GLR
- [ ] polyhedral-loop-transforms: affine loop nests, dependence analysis, tiling / fusion / skewing / interchange legality, scheduling
- [ ] staging-partial-evaluation: multi-stage programming, partial evaluation, Futamura projections, specialisation, supercompilation
- [ ] probabilistic-programming: PPL semantics (traces, measures), effect handlers for inference, inference compilation, gradient-based inference

### 17. graphs: Search graphs & index space

- [ ] graph-algorithms: BFS / DFS, topological sort, SCCs (Tarjan), shortest paths (Dijkstra, Bellman–Ford, A*), MST, max-flow / min-cut, matchings, union-find
- [ ] spatial-indexes: k-d trees, octrees, BVH, R-trees, uniform grids / cell lists, Morton / Hilbert ordering, nearest-neighbour and range queries

## Open items

- [ ] Review the three ★ pilot pages and adjust the template before batching.
- [ ] Turn plain-text "See also" entries into links as their targets get published.
- [ ] Pagefind full-text search (post-build index) once there are enough pages.
- [ ] Mobile: consider collapsing the "On this page" TOC (`<details>`) on long pages.
- [ ] Maybe: printable / offline view.
