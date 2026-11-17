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
3. `## Variants`: per method, the core formula or update rule and its costs. Write algorithms as numbered steps with formulas, not code.
4. `## Other lenses`: the same idea in LA / probability / optimisation / CT language, kept short.
5. `## Pitfalls & diagnostics`
6. `## See also` + 1–2 canonical references.

No library or "in practice" section: pages stay library-agnostic.

**Structure:**

- No code blocks. Formulas go in KaTeX, algorithms in numbered steps. Short inline code is fine for names (`atol`, `map`, LAPACK routines).
- Use lists, not paragraphs strung together from **Bold:** fragments. Each labelled point is its own bullet, with sub-bullets for details. Keep prose paragraphs for one- or two-sentence intros.
- See also and References are lists too.

**Syntax gotchas** (mdsvex → Svelte):

- Math: `$…$` inline and `$$…$$` display, on their own lines with blank lines around them. It is rendered by KaTeX at build time, and invalid TeX fails the build. Commutative diagrams use `\begin{CD}`.
- Display math inside list items renders inline. Use `$\displaystyle …$` there, or move it out of the list.
- No raw `{ }` or `<` in prose (Svelte parses them). Put them inside math or inline code, or use `≲`/`≤`.
- No `|` inside math in tables (it breaks the table). Use `\lvert x\rvert`.
- Cross-references are links (`/ref/<section>/<slug>`). Strict prerendering fails the build on dead links, so a reference to a page that doesn't exist yet stays plain text until it does.
- Inline math must not start or end with a space inside the dollars: `$x = $` is not parsed as math, and its braces then break Svelte.
- KaTeX lacks some LaTeX macros (e.g. `\bigsqcap`); the build fails and names the macro.
- `src/ref/` is excluded from Prettier (`.prettierignore`): it doesn't understand math and rewrites `x^*` … `y^*` as emphasis.

## Pages

★ = pilot page. `[x]` = published. All planned pages are written; add new topics at the end of their section (and to `src/lib/ref.ts` for new sections).

### 1. reliable: Compute reliably

- [x] floating-point: IEEE 754 formats, ulp / machine eps, rounding modes, cancellation, FMA, subnormals, NaN semantics
- [x] summation-reductions: pairwise, Kahan / Neumaier, compensated dot products, reproducible reductions
- [x] conditioning-stability: condition numbers, forward vs backward error, scaling / equilibration
- [x] interval-affine-arithmetic: interval arithmetic, dependency problem, affine arithmetic, validated numerics

### 2. solve: Solve equations

- [x] ★ linear-systems: LU, Cholesky, LDLᵀ, QR, banded / Thomas, Woodbury, mixed-precision refinement
- [x] sparse-matrices: COO/CSR/CSC, fill-in, AMD / nested dissection / RCM, symbolic vs numeric factorisation
- [x] krylov-preconditioning: CG, MINRES, GMRES (restarts), BiCGSTAB; Jacobi, ILU, AMG, block preconditioners
- [x] least-squares: normal equations vs QR vs SVD, Tikhonov, weighted / generalised / total LS, NNLS
- [x] nonlinear-roots: bisection, Brent, Newton, secant, Broyden, Newton–Krylov, globalisation, homotopy
- [x] eigenproblems: power / inverse iteration, Rayleigh quotient, QR algorithm, Lanczos / Arnoldi, shift-invert, generalised
- [x] matrix-functions: expm (Padé scaling-squaring), Krylov expm·v, sqrtm / logm, φ-functions
- [x] multigrid-domain-decomposition: geometric vs algebraic multigrid (smoothers, V/W/F-cycles), as solver vs preconditioner, Schwarz / Schur-complement DD
- [x] hierarchical-matrices-fmm: fast multipole, H-/H²-matrices, HSS, low-rank off-diagonal blocks, O(n log n) kernel matrices (GPs, N-body, BEM)

### 3. diff: Differentiate

- [x] finite-differences: forward / central / higher order, optimal step size, complex step, Richardson
- [x] automatic-differentiation: dual numbers, reverse tape, checkpointing, Jacobian colouring, HVPs; symbolic + CSE as a lens
- [x] ode-sensitivities: forward sensitivities, continuous vs discrete adjoint, interpolating / backsolve / checkpointed, cost vs #params
- [x] matrix-calculus: numerator vs denominator layout, common identities (trace, logdet, inverse)

### 4. integrate: Integrate & evolve

- [x] quadrature: Newton–Cotes, Gauss, Clenshaw–Curtis, adaptive (Gauss–Kronrod), singularities, high-dim → MC / QMC
- [x] ★ ode-initial-value: explicit RK, Adams, BDF, Radau, Rosenbrock, SDIRK, stiffness, stability functions
- [x] step-control-events: embedded error estimates, PI controllers, initial step, dense output, event location
- [x] daes: index, mass matrices, consistent initialisation, index reduction (Pantelides), BDF / Radau
- [x] geometric-integrators: Verlet / leapfrog, symplectic Euler, splitting (Strang), energy drift, Lie-group methods
- [x] sdes: Euler–Maruyama, Milstein, SRI / SRA, Itô vs Stratonovich, strong vs weak order
- [x] pdes-method-of-lines: FD / FV / FE / spectral, CFL, upwinding, IMEX, operator splitting
- [x] stochastic-simulation: Gillespie SSA, next reaction, τ-leaping, chemical Langevin; DDE note
- [x] parallel-in-time-exponential: parareal, PFASST, exponential integrators (ETD, Krylov φ-functions), IMEX-RK

### 5. dynamics: Analyse dynamics

- [x] steady-states-stability: root solve vs time-stepping, Jacobian eigenvalues, Lyapunov functions
- [x] bifurcation-continuation: natural vs pseudo-arclength, fold / Hopf / branch-point detection
- [x] timescale-reduction: QSSA, Tikhonov, slow manifolds, POD, balanced truncation, DMD
- [x] global-sensitivity: local vs global, Morris, Sobol indices (Saltelli), FAST, MCA control coefficients
- [x] stoichiometric-analysis: null space, conservation laws, elementary modes, FBA / FVA as LP
- [x] periodic-orbits-chaos: shooting for limit cycles, Floquet multipliers, Poincaré maps, Lyapunov exponents
- [x] stochastic-processes: discrete / continuous-time Markov chains, Poisson processes, chemical master equation, Fokker–Planck, stationary distributions, mixing times
- [x] reaction-network-theory: CRNT, deficiency zero / one theorems, multistationarity, toric steady states, Gröbner bases / symbolic steady-state solving

### 6. approx: Approximate functions

- [x] interpolation: barycentric Lagrange, Runge, Chebyshev nodes, cubic / B-splines, PCHIP / Akima
- [x] chebyshev-spectral: Chebyshev series, Clenshaw, spectral convergence, Chebfun-style workflows
- [x] fourier-fft: DFT / FFT, aliasing, windowing, FFT convolution, real FFTs, NUFFT
- [x] low-rank: SVD, truncated / randomised SVD, PCA, NMF, CP / Tucker / tensor train
- [x] gaussian-processes: kernels, Cholesky cost, inducing points / sparse GPs, hyperparameter fitting
- [x] surrogates: RBF, polynomial chaos, NN surrogates, active learning
- [x] high-dimensional-numerics: curse of dimensionality, sparse grids (Smolyak), tensor-train methods, when MC / QMC beats grids
- [x] sparse-recovery: compressed sensing, LASSO / basis pursuit, OMP, RIP / coherence, proximal solvers (ISTA / FISTA), link to SINDy

### 7. optimise: Optimise & fit

- [x] unconstrained-local: GD, Newton, BFGS / L-BFGS, nonlinear CG; Wolfe line search vs trust region
- [x] nonlinear-least-squares: Gauss–Newton, Levenberg–Marquardt, TRF, residual Jacobians
- [x] constrained: LP (simplex / IPM), QP, SQP, augmented Lagrangian, barrier, projection, bound transforms
- [x] convex-duality: Lagrangian, KKT, duality gap, proximal operators, ADMM
- [x] global-derivative-free: Nelder–Mead, CMA-ES, DE, PSO, simulated annealing, multistart, Bayesian optimisation
- [x] stochastic-gradient: SGD, momentum / Nesterov, Adam / AdamW, schedules, clipping, batch size
- [x] fitting-odes: single vs multiple shooting, collocation, log-parameters, scaling, local minima
- [x] neural-odes-udes: NODE training, adjoint choice, UDEs, SINDy / symbolic regression, stiffness in training
- [x] optimal-control-mpc: Pontryagin, direct vs indirect methods, direct transcription / collocation, multiple shooting, MPC
- [x] mixed-integer-sat-smt: branch & bound, cutting planes, MILP formulation tricks (big-M, indicator constraints), SAT (CDCL), SMT
- [x] manifold-optimisation: Riemannian gradients, retractions, Stiefel / SPD / SO(3), Lie-group integrators, reparametrisation vs manifold methods

### 8. random: Generate randomness

- [x] prngs: LCG, xorshift / xoshiro, PCG, MT19937, Philox / Threefry, streams / splitting, JAX-style keys
- [x] sampling-distributions: inverse CDF, Box–Muller / Ziggurat, rejection, alias method, Marsaglia–Tsang gamma, MVN via Cholesky, Gumbel-max
- [x] quasi-monte-carlo: Sobol, Halton, LHS, scrambling, discrepancy, error rates
- [x] variance-reduction: antithetic, control variates, importance sampling, stratification, MLMC

### 9. bayes: Infer (Bayesian)

- [x] foundations-priors: Bayes' rule, conjugacy, prior choice, hierarchical models, non-centred parametrisation
- [x] mcmc: Metropolis–Hastings, Gibbs, adaptive MH, slice, ensemble samplers; R-hat / ESS
- [x] hmc-nuts: leapfrog, mass matrix, step-size adaptation, divergences
- [x] sequential-monte-carlo: SMC samplers, tempering, PMCMC
- [x] variational-inference: ELBO, mean-field, ADVI, normalising flows, reparametrisation trick; Laplace / INLA
- [x] likelihood-free: ABC rejection, SMC-ABC, synthetic likelihood, neural SBI (NPE / NLE)
- [x] model-comparison: evidence, Bayes factors, nested sampling, thermodynamic integration, WAIC, PSIS-LOO

### 10. freq: Infer (frequentist)

- [x] estimation-mle: MLE, Fisher information, Cramér–Rao, sandwich estimator, M-estimators
- [x] confidence-intervals: Wald, profile likelihood, bootstrap percentile / BCa
- [x] identifiability: structural vs practical, FIM rank, profile likelihood, sloppiness
- [x] hypothesis-tests: t, χ², LRT / Wald / score, nonparametric, permutation, power
- [x] multiple-testing: Bonferroni, Holm, Benjamini–Hochberg FDR
- [x] regression-glms: OLS assumptions, GLMs / IRLS, mixed models, robust regression
- [x] resampling: bootstrap, jackknife, cross-validation
- [x] model-selection: AIC, BIC, nested LRT, CV
- [x] experimental-design: FIM-based D / A / E-optimality, Bayesian OED (expected information gain), sequential / adaptive design
- [x] conformal-prediction: split / full / jackknife+ conformal, exchangeability, conformalised quantile regression, coverage guarantees

### 11. signals: Estimate state & signals

- [x] kalman-filters: KF, EKF, UKF, EnKF, RTS smoother
- [x] particle-filters: bootstrap PF, resampling schemes, degeneracy
- [x] filtering-convolution: FIR / IIR, Savitzky–Golay, convolution theorem
- [x] spectral-estimation: periodogram, Welch, Lomb–Scargle, autocorrelation
- [x] time-series-models: AR / ARMA, state-space form

### 12. info: Information & distances

- [x] entropy-kl-mi: entropy, cross-entropy, KL, JS, mutual information estimators
- [x] fisher-information-geometry: FIM, natural gradient, Jeffreys prior
- [x] distribution-distances: TV, Hellinger, Wasserstein / OT / Sinkhorn, MMD, two-sample tests

### 13. learn: Learn from data

- [x] classical-supervised: linear / logistic, kernels / SVM, trees / RF / GBM, kNN
- [x] unsupervised: k-means, GMM / EM, hierarchical, DBSCAN, t-SNE / UMAP
- [x] nn-building-blocks: MLP, activations, initialisation, normalisation, residuals, convolutions
- [x] attention-transformers: attention, positional encodings, KV cache, complexity
- [x] regularisation-generalisation: L1 / L2, dropout, early stopping, bias–variance, double descent
- [x] training-practice: losses, LR finding, mixed precision, gradient checkpointing
- [x] generative-models: VAE, GAN, flows, diffusion
- [x] physics-informed-ml: PINNs, DeepONet / FNO, equivariance

### 14. explain: Explain & reason causally

- [x] feature-attribution: permutation importance, PDP / ICE, SHAP variants, LIME, integrated gradients, saliency
- [x] causal-graphs-scms: DAGs, d-separation, SCMs, do-operator, backdoor / frontdoor, IVs
- [x] causal-effect-estimation: adjustment, IPW, doubly robust, matching, DiD, RD
- [x] counterfactuals: abduction–action–prediction, counterfactual explanations
- [x] causal-discovery: PC, GES, NOTEARS, LiNGAM

### 15. space: Transform space

- [x] vectors-tensors-einsum: index notation, einsum, broadcasting, strides / memory layout, contraction order
- [x] rotations: matrices, Euler angles / gimbal lock, axis-angle, quaternions, slerp, SO(3) exp / log
- [x] homogeneous-projections: affine 4×4, view / perspective matrices, normal matrix, reverse-Z
- [x] geometry-primitives: barycentric coordinates, intersections, robust orientation predicates

### 16. programs: Structure programs

- [x] ★ category-theory-core: categories, functors, natural transformations, universal properties, adjunctions
- [x] monads-effects: monads, Kleisli, applicatives, free monads, algebraic effects / handlers
- [x] algebraic-data-types: F-algebras, initial algebras, catamorphisms / recursion schemes, codata
- [x] algebraic-structures: monoids, semirings, tropical semiring (parallel reductions, path problems)
- [x] lambda-calculus-types: untyped / STLC, System F, Curry–Howard, variance, dependent types (brief)
- [x] type-inference-unification: unification, Hindley–Milner / algorithm W, bidirectional typing
- [x] lattices-fixpoints-dataflow: posets, Kleene / Tarski, dataflow analysis, abstract interpretation, widening
- [x] term-rewriting-egraphs: confluence, termination, e-graphs / equality saturation
- [x] ssa-ir: SSA, dominators, φ-nodes, CPS correspondence
- [x] parsing-automata: regex → NFA → DFA, CFGs, LL / LR, Pratt, PEG, GLR
- [x] polyhedral-loop-transforms: affine loop nests, dependence analysis, tiling / fusion / skewing / interchange legality, scheduling
- [x] staging-partial-evaluation: multi-stage programming, partial evaluation, Futamura projections, specialisation, supercompilation
- [x] probabilistic-programming: PPL semantics (traces, measures), effect handlers for inference, inference compilation, gradient-based inference

### 17. graphs: Search graphs & index space

- [x] graph-algorithms: BFS / DFS, topological sort, SCCs (Tarjan), shortest paths (Dijkstra, Bellman–Ford, A*), MST, max-flow / min-cut, matchings, union-find
- [x] spatial-indexes: k-d trees, octrees, BVH, R-trees, uniform grids / cell lists, Morton / Hilbert ordering, nearest-neighbour and range queries

## Open items

- [x] Review the three ★ pilot pages and adjust the template before batching (no code blocks; lists instead of bold-label paragraphs).
- [x] Turn plain-text "See also" entries into links as their targets get published.
- [ ] Pagefind full-text search (post-build index): now worth doing with 114 pages.
- [ ] Mobile: consider collapsing the "On this page" TOC (`<details>`) on long pages.
- [ ] Maybe: printable / offline view.
