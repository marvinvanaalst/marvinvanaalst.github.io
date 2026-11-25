---
title: 'SSA & intermediate representations'
description: 'Control-flow graphs, dominators and dominance frontiers, SSA construction with φ-nodes, SSA-based optimisations, out-of-SSA translation, and how SSA relates to CPS and ANF.'
aliases:
  [
    SSA,
    static single assignment,
    phi node,
    phi function,
    control flow graph,
    CFG,
    basic blocks,
    dominators,
    dominator tree,
    dominance frontier,
    Cytron algorithm,
    Braun SSA construction,
    out of SSA,
    copy propagation,
    SCCP,
    global value numbering,
    GVN,
    dead code elimination,
    loop invariant code motion,
    CPS,
    ANF,
    basic block arguments,
    MLIR,
    LLVM IR,
    sea of nodes
  ]
tags: [compilers, programming languages, intermediate representations]
order: 9
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *CFG:* basic blocks (straight-line code) connected by branches. Most compiler analyses run on it.
- *SSA:* every variable is assigned exactly once; merges use *φ-nodes* (or block arguments). Def-use chains become explicit, making many optimisations simple and sparse.
- *Construction:* Cytron et al. (dominance frontiers for φ placement + renaming) or Braun et al. (on-the-fly, directly from the AST, no dominance computation).
- *Optimisations on SSA:* constant propagation (SCCP), global value numbering, dead code elimination, copy propagation, LICM, partial redundancy elimination.
- *Out of SSA:* replace φ-nodes with copies on edges (split critical edges), then coalesce copies during register allocation.
- *SSA ≈ functional programming:* blocks with arguments are local functions; SSA, ANF and CPS are equivalent views (MLIR and Swift SIL use block arguments).

## Decision table

| IR / technique | Representation | Strength | Used in |
| --- | --- | --- | --- |
| AST | syntax tree | close to source, desugaring, type checking | front ends |
| CFG of three-address code | blocks of simple instructions | general dataflow analysis | classic back ends |
| SSA with φ-nodes | single assignment + φ at merges | sparse analyses, simple optimisations | LLVM IR, GCC GIMPLE |
| SSA with block arguments | φ as parameters of blocks | cleaner semantics, functional correspondence | MLIR, Swift SIL, Cranelift |
| Sea of nodes | graph of data + control dependencies | aggressive global code motion | HotSpot C2, V8 TurboFan (historically), Graal |
| ANF | let-bound intermediate results | functional compilers | functional language compilers |
| CPS | every call passes its continuation | explicit control, first-class continuations | Scheme/ML compilers, effect handlers |
| Multi-level IR (MLIR dialects) | progressive lowering of domain ops | domain-specific optimisation (tensors, loops) | ML compilers, HPC DSLs |

## Variants

### CFG and dominators

- $`d`$ *dominates* $`n`$ if every path from entry to $`n`$ passes through $`d`$; immediate dominators form the *dominator tree*.
- *Dominance frontier* of $`d`$: nodes where $`d`$'s dominance ends (join points reached from both dominated and non-dominated paths); where φ-nodes are needed.
- Computation: Cooper–Harvey–Kennedy iterative algorithm (simple, fast in practice) or Lengauer–Tarjan (near-linear).
- Natural loops: back edges $`n \to h`$ where $`h`$ dominates $`n`$.

### SSA construction (Cytron et al.)

1. Compute dominators and dominance frontiers.
2. *Place φ-nodes:* for each variable, insert φ at the iterated dominance frontier of its definition sites (pruned/semi-pruned variants use liveness to avoid dead φs).
3. *Rename:* walk the dominator tree with a stack of current definitions per variable, giving each definition a fresh name and rewriting uses.

- *Braun et al.:* build SSA directly during AST lowering with on-demand variable lookup, inserting φ lazily and removing trivial ones; no dominance frontiers needed.

### Optimisations that SSA makes easy

- *SCCP:* sparse conditional constant propagation over SSA edges plus executable CFG edges; finds constants and dead branches together.
- *GVN:* assign value numbers to expressions; equal numbers = redundant computations (hash-consing over SSA values).
- *DCE:* remove definitions with no uses (iteratively, or mark-and-sweep from side effects).
- *LICM:* hoist loop-invariant computations to preheaders; *PRE / GVN-PRE* removes partial redundancies.
- Strength reduction and induction variable analysis (scalar evolution) on SSA loops.

### Out of SSA

- Replace each φ with copies at the end of predecessor blocks; split *critical edges* (edges from a block with several successors to one with several predecessors) first.
- *Lost-copy and swap problems* require care (parallel copy semantics).
- Register allocators (graph colouring on SSA is efficient: SSA interference graphs are chordal) coalesce most copies away.

### SSA, ANF and CPS

- A block with φ-nodes ≈ a local function whose parameters are the φ-variables; a jump ≈ a tail call with arguments (block arguments make this explicit).
- ANF names every intermediate result (like SSA within a block); CPS additionally makes return points explicit continuations.
- Consequences: functional-language optimisations (inlining, contification) and SSA optimisations are the same ideas.

## Other lenses

- *Graph algorithms.* Dominators, SCCs, loop nesting forests and reverse postorder traversals underpin every pass (see [graph algorithms](/ref/graphs/graph-algorithms)).
- *Dataflow analysis.* SSA turns dense, per-program-point analyses into sparse ones on def-use chains (see [lattices & dataflow](/ref/programs/lattices-fixpoints-dataflow)).
- *Rewriting.* Peephole and algebraic simplifications on SSA graphs; e-graphs as an alternative IR for optimisation (see [term rewriting & e-graphs](/ref/programs/term-rewriting-egraphs)).
- *AD.* Reverse-mode AD on SSA IR (Enzyme, Zygote) differentiates after optimisation; φ-nodes become accumulations in the adjoint (see [automatic differentiation](/ref/diff/automatic-differentiation)).

## Pitfalls & diagnostics

- *Memory isn't in SSA* automatically: loads/stores need mem2reg / scalar replacement or memory SSA to benefit.
- *Critical edges* must be split before out-of-SSA or code motion.
- *φ semantics:* all φ-nodes in a block read their inputs simultaneously (parallel copy); sequentialising them naively causes bugs.
- *Exceptions and irreducible control flow* complicate dominance-based algorithms; handle explicitly or restructure.
- *Verify IR invariants* after every pass (dominance of definitions over uses), as LLVM's verifier does.

## See also

- [Lattices, fixpoints & dataflow](/ref/programs/lattices-fixpoints-dataflow)
- [Polyhedral loop transforms](/ref/programs/polyhedral-loop-transforms)
- [Staging & partial evaluation](/ref/programs/staging-partial-evaluation)
- [Graph algorithms](/ref/graphs/graph-algorithms)

*References:*

- Cytron et al., "Efficiently computing static single assignment form and the control dependence graph", TOPLAS (1991).
- Braun et al., "Simple and efficient construction of static single assignment form", CC (2013).
- Rastello & Bouchez Tichadou (eds.), _SSA-based Compiler Design_.
