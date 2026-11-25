---
title: 'Algebraic structures for computing'
description: 'Semigroups, monoids, groups, rings and semirings (including the tropical semiring) and lattices: the structure that makes parallel reductions, path problems and dataflow work.'
aliases:
  [
    semigroup,
    monoid,
    group,
    abelian group,
    ring,
    field,
    semiring,
    tropical semiring,
    max-plus algebra,
    min-plus,
    boolean semiring,
    lattice,
    join,
    meet,
    associativity,
    commutativity,
    identity element,
    parallel reduction,
    prefix sum,
    scan,
    foldMap,
    path algebra,
    Kleene star,
    GraphBLAS,
    homomorphism
  ]
tags: [algebra, programming languages, parallel computing, graphs]
order: 4
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Monoid* (associative operation + identity) is the key structure for computing: any monoid reduction can be *parallelised* (tree reduction) and *incrementalised* (prefix sums / scans, segment trees).
- *Commutativity* additionally allows reordering (unordered parallel reductions, out-of-order accumulation); groups (inverses) allow sliding windows by subtraction.
- *Semirings* (two monoids with distributivity) generalise linear algebra: swap $`(+, \times)`$ for $`(\min, +)`$ to get shortest paths, $`(\lor, \land)`$ for reachability, $`(\max, \times)`$ for most-probable paths. One algorithm, many problems.
- *Lattices* (join/meet) underlie dataflow analysis, abstract interpretation and CRDTs (merges converge because join is associative, commutative, idempotent).
- Floating-point addition is only approximately associative: parallel sums differ in the last bits (see [summation](/ref/reliable/summation-reductions)).

## Decision table

| Structure | Axioms | Computational payoff | Examples |
| --- | --- | --- | --- |
| Semigroup | associative $`\cdot`$ | tree reduction (non-empty), parallel `reduce` | max, min, concatenation of non-empty lists |
| Monoid | + identity $`e`$ | reductions of empty collections, `foldMap`, scans | $`(+, 0)`$, $`(\times, 1)`$, strings, maps with merge |
| Commutative monoid | + commutativity | unordered/concurrent accumulation, histograms | counts, sums, sets under union |
| Group | + inverses | sliding windows, undo, differences of prefix sums | integers under $`+`$, invertible matrices |
| Semiring | $`(\oplus, 0)`$ comm. monoid, $`(\otimes, 1)`$ monoid, distributive, $`0`$ annihilates | generalised matrix products, path problems, dynamic programming | $`(+,\times)`$, $`(\min,+)`$, $`(\max,\times)`$, $`(\lor,\land)`$ |
| Ring / field | + additive inverses (/ multiplicative inverses) | Gaussian elimination, linear algebra | $`\mathbb R`$, $`\mathbb Z_p`$, $`\mathrm{GF}(2)`$ |
| Lattice / semilattice | join (and meet): associative, commutative, idempotent | fixpoint iteration, CRDT merges | sets, intervals, max over a partial order |

| Semiring $`(\oplus, \otimes, 0, 1)`$ | Matrix "product" computes |
| --- | --- |
| $`(+, \times, 0, 1)`$ | number of paths / ordinary linear algebra |
| $`(\min, +, \infty, 0)`$ (tropical) | shortest path lengths |
| $`(\max, +, -\infty, 0)`$ | longest paths in DAGs, critical paths, Viterbi in log space |
| $`(\max, \times, 0, 1)`$ | most probable paths |
| $`(\lor, \land, \text{false}, \text{true})`$ | reachability (transitive closure) |
| $`(\max, \min, 0, \infty)`$ | widest / bottleneck paths |
| (log-sum-exp, $`+`$, $`-\infty`$, 0) | forward algorithm, partition functions in log space |

## Variants

### Monoid reductions and scans

- *Parallel reduce:* combine pairs in a balanced tree, $`O(\log n)`$ depth; requires associativity only.
- *Scan (prefix sums):* all partial reductions $`x_1, x_1\oplus x_2, \dots`$ in $`O(\log n)`$ depth (Blelloch scan); building block for stream compaction, radix sort, parallel parsing.
- *Monoid homomorphisms* ($`h(a\cdot b) = h(a)\cdot h(b)`$) are exactly the functions computable by divide-and-conquer: map-reduce = foldMap.
- *Clever monoids:* (count, mean, M2) for parallel variance (Chan's formula); (max, argmax); sorted runs for merges; matrices for linear recurrences (parallel scan of $`x_{t+1} = A_tx_t + b_t`$, as in parallel state-space models).

### Semirings and path problems

- Weighted adjacency matrix $`W`$ over a semiring; $`W^k`$ gives best $`k`$-step paths; the closure $`W^* = I \oplus W \oplus W^2 \oplus \dots`$ gives best paths overall.
- Algorithms (Floyd–Warshall, Bellman–Ford, Gauss–Jordan elimination) generalise to closed semirings; Dijkstra needs a selective (min-like) semiring with non-negative weights.
- *GraphBLAS:* sparse matrix operations over user-chosen semirings express graph algorithms (BFS = boolean semiring matvec).
- *Dynamic programming* recurrences (Viterbi, CYK parsing, sequence alignment) are semiring computations; swapping the semiring changes "best" into "sum over all" (forward algorithm) or "count".

### Lattices

- Partial order with joins (least upper bounds) and meets.
- *Fixpoint iteration* over finite-height lattices terminates (dataflow analysis) (see [lattices & fixpoints](/ref/programs/lattices-fixpoints-dataflow)).
- *CRDTs:* replicated state as a join-semilattice; merges commute and are idempotent, so replicas converge regardless of message order.

## Other lenses

- *Category theory.* A monoid is a one-object category; monoids in the category of endofunctors are monads; semiring modules generalise vector spaces (see [category theory](/ref/programs/category-theory-core)).
- *Linear algebra.* Max-plus linear algebra has eigenvalues (cycle means) describing periodic regimes of scheduling and discrete-event systems.
- *Probability.* Semiring parsing / message passing: sum-product (marginals) vs max-product (MAP) belief propagation are the same algorithm over different semirings.

## Pitfalls & diagnostics

- *Assumed associativity* that doesn't hold (floating point, averaging means directly, non-associative user operations) gives scheduling-dependent results; use the right monoid (e.g. (sum, count) for means) or compensated arithmetic.
- *Missing identity:* reductions over empty inputs need an identity or an option type.
- *Negative cycles* break shortest-path semirings (no closure); detect them (Bellman–Ford).
- *Non-commutative operations* reduced with unordered parallel primitives give wrong answers; use ordered scans/reductions.

## See also

- [Summation & reductions](/ref/reliable/summation-reductions)
- [Graph algorithms](/ref/graphs/graph-algorithms)
- [Lattices, fixpoints & dataflow](/ref/programs/lattices-fixpoints-dataflow)
- [Algebraic data types](/ref/programs/algebraic-data-types)

*References:*

- Gondran & Minoux, _Graphs, Dioids and Semirings_.
- Blelloch, "Prefix sums and their applications" (1990).
- Kepner & Gilbert (eds.), _Graph Algorithms in the Language of Linear Algebra_.
