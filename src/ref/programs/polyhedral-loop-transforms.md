---
title: 'Polyhedral model & loop transformations'
description: 'Affine loop nests as polyhedra, dependence analysis, legality of interchange, tiling, fusion and skewing, affine scheduling, and how this maps onto cache, SIMD and parallel hardware.'
aliases:
  [
    polyhedral model,
    polyhedral compilation,
    affine loop nest,
    iteration domain,
    dependence analysis,
    dependence vector,
    distance vector,
    direction vector,
    loop interchange,
    loop tiling,
    blocking,
    loop fusion,
    loop fission,
    loop skewing,
    wavefront parallelism,
    unimodular transformations,
    affine scheduling,
    Pluto,
    isl,
    Polly,
    Halide,
    scheduling language,
    roofline,
    cache blocking,
    vectorization
  ]
tags: [compilers, performance, loops, parallelism]
order: 11
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- Loop nests with affine bounds and array indices (stencils, linear algebra, many simulations) can be modelled exactly: iterations are integer points of a *polyhedron*, dependences are affine relations between them.
- A transformation (interchange, skewing, tiling, fusion) is *legal* iff it preserves the order of all dependent iterations, i.e. every dependence distance stays lexicographically positive.
- *Why bother:* performance is about data movement. Tiling for cache reuse, interchange for contiguous access, fusion to avoid intermediate arrays, skewing to expose parallel wavefronts.
- Automatic polyhedral optimisers (Pluto, Polly, isl-based tools) find good schedules; *scheduling languages* (Halide, TVM, MLIR affine/linalg) let you separate the algorithm from its schedule and choose it yourself.
- Use the roofline model to decide whether a kernel is memory- or compute-bound before optimising.

## Decision table

| Transformation | Effect | Legal when | Benefit |
| --- | --- | --- | --- |
| Interchange | swap loop order | no dependence becomes lexicographically negative | contiguous (unit-stride) access, vectorisation |
| Tiling (blocking) | iterate over blocks, then within blocks | loops in the band are fully permutable | cache/register reuse (GEMM, stencils) |
| Fusion | merge adjacent loops | no dependence reversed between the bodies | fewer passes over memory, no temporaries |
| Fission (distribution) | split a loop body | no dependence cycle across the split | enable vectorisation/parallelism per part |
| Skewing | shear the iteration space | always (unimodular), used to make tiling legal | wavefront parallelism for stencils |
| Reversal | iterate backwards | no carried dependence in that loop (or adjusted) | enable fusion/interchange |
| Unrolling / unroll-and-jam | replicate bodies | always (with remainder handling) | ILP, register reuse |
| Vectorisation | process $`w`$ iterations per instruction | innermost loop has no carried dependence (or reduction) | SIMD throughput |
| Parallelisation | iterations on different cores | loop carries no dependence | multicore/GPU |

## Variants

### Iteration domains and dependences

- Iteration domain: $`\{(i, j) : 0 \le i \lt N,\ 0 \le j \le i\}`$ — integer points of a polyhedron.
- *Dependence* from iteration $`s`$ to $`t`$ if both access the same memory location, at least one writes, and $`s`$ executes before $`t`$ (flow/RAW, anti/WAR, output/WAW).
- *Distance vectors* $`t - s`$ (when constant), e.g. $`(1, 0)`$ = carried by the outer loop. Direction vectors abstract signs.
- Exact dependence analysis = integer linear feasibility (Omega test, isl); approximations (GCD, Banerjee tests) are cheaper.

### Legality

- A schedule $`\theta`$ (mapping iterations to time) is legal iff $`\theta(t) \succ \theta(s)`$ for every dependence $`s \to t`$.
- For loop permutations: legal iff every distance vector permuted stays lexicographically positive.
- Anti and output dependences can often be removed by *privatisation / renaming* (array expansion, scalar replacement), freeing more transformations.

### Tiling

- Requires a band of fully permutable loops (all distance components non-negative in those dimensions); skewing first if necessary (e.g. time-iterated stencils).
- Tile sizes from cache capacities (and register blocking inside); multi-level tiling for L1/L2/L3.
- Matrix multiply: the textbook example, from $`O(n^3)`$ memory traffic to $`O(n^3/\sqrt{M})`$ for cache size $`M`$.

### Affine scheduling (Pluto)

- Find affine schedule hyperplanes $`\theta(i) = c^\top i + c_0`$ satisfying all dependences, minimising dependence distances (for locality) and maximising outer parallelism; solved as a sequence of ILPs.
- Then tile the permutable bands and parallelise outer loops / vectorise inner ones; generate code by polyhedra scanning (CLooG, isl AST generation).

### Scheduling languages

- *Halide / TVM / MLIR:* write the computation as a pure function of indices; specify the schedule separately (tile, vectorise, parallelise, compute\_at, store\_at). Autotuners and learned cost models search schedule spaces.
- Keeps algorithms readable and makes trying schedules cheap.

## Other lenses

- *Linear algebra & geometry.* Loop transformations are unimodular (integer, determinant ±1) or affine maps of iteration polyhedra; legality is a linear-inequality question (Farkas' lemma) (see [constrained optimisation](/ref/optimise/constrained)).
- *Static analysis.* Dependence analysis is a specialised abstract interpretation on array indices; polyhedra are also an abstract domain (see [lattices & dataflow](/ref/programs/lattices-fixpoints-dataflow)).
- *Numerics.* Reordering loops reorders floating-point operations; reductions change results in the last bits unless reassociation is disallowed (see [summation](/ref/reliable/summation-reductions)).
- *Array programming.* einsum expressions compile to loop nests; tensor compilers choose loop orders and tiling (see [vectors, tensors & einsum](/ref/space/vectors-tensors-einsum)).

## Pitfalls & diagnostics

- *Non-affine code* (indirect indexing, data-dependent bounds, function calls, pointer aliasing) falls outside the model; restructure or annotate (restrict, assume).
- *Measure, don't guess:* profile, check the roofline position (arithmetic intensity vs bandwidth), count cache misses.
- *Over-tiling* small problems adds overhead; tile sizes are hardware-specific (autotune).
- *Aliasing* prevents compilers from proving independence; use local copies, restrict pointers or languages with clear aliasing rules.
- *Compilers already do much of this* for simple loops; check vectorisation reports before hand-optimising.

## See also

- [Vectors, tensors & einsum](/ref/space/vectors-tensors-einsum)
- [SSA & intermediate representations](/ref/programs/ssa-ir)
- [Lattices, fixpoints & dataflow](/ref/programs/lattices-fixpoints-dataflow)
- [Staging & partial evaluation](/ref/programs/staging-partial-evaluation)

*References:*

- Bondhugula et al., "A practical automatic polyhedral parallelizer and locality optimizer" (Pluto), PLDI (2008).
- Ragan-Kelley et al., "Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines", PLDI (2013).
- Kennedy & Allen, _Optimizing Compilers for Modern Architectures_.
