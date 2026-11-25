---
title: 'Parsing & automata'
description: 'Regular expressions to NFA to DFA, lexers, context-free grammars, LL and LR parsing, Pratt parsing for expressions, PEGs, GLR/Earley for ambiguous grammars, and error recovery.'
aliases:
  [
    parsing,
    lexing,
    tokenizer,
    regular expressions,
    finite automata,
    NFA,
    DFA,
    Thompson construction,
    subset construction,
    DFA minimization,
    backtracking regex,
    context-free grammar,
    CFG,
    BNF,
    LL parsing,
    recursive descent,
    LR parsing,
    LALR,
    Pratt parsing,
    precedence climbing,
    operator precedence,
    PEG,
    packrat,
    GLR,
    Earley,
    CYK,
    ambiguity,
    error recovery,
    incremental parsing,
    tree-sitter
  ]
tags: [compilers, automata, programming languages]
order: 10
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Lexing:* regular expressions → NFA (Thompson) → DFA (subset construction) → minimised DFA; linear time per input character. Avoid backtracking regex engines for untrusted input (catastrophic backtracking).
- *Parsing a programming language:* a hand-written *recursive descent* parser with *Pratt parsing* for expressions is the pragmatic default: fast, flexible, good error messages.
- *Generated parsers:* LR(1)/LALR (yacc, bison, Menhir) handle more grammars than LL; LL(_) / ALL(_) (ANTLR) is friendlier.
- *PEG / packrat:* ordered choice, no ambiguity by construction, linear time with memoisation; but ordered choice can silently hide alternatives.
- *Ambiguous or arbitrary CFGs* (natural language, some DSLs): Earley or GLR, $`O(n^3)`$ worst case.
- *Editors:* incremental, error-tolerant parsing (tree-sitter) matters more than grammar class.

## Decision table

| Technique | Grammar class | Time | Errors / ergonomics | Use when |
| --- | --- | --- | --- | --- |
| DFA lexer | regular | $`O(n)`$ | simple | tokens |
| Backtracking regex (PCRE) | regular + backrefs | exponential worst case | convenient | trusted inputs, small patterns |
| Recursive descent (LL(k)) | LL(k), no left recursion | $`O(n)`$ | excellent, hand-tuned errors | *hand-written language parsers* |
| Pratt / precedence climbing | operator expressions | $`O(n)`$ | easy precedence/associativity | expressions inside recursive descent |
| LALR(1) / LR(1) generators | LR(1) (deterministic CFGs) | $`O(n)`$ | conflicts need grammar surgery; errors harder | large grammars, guaranteed determinism |
| ALL(\*) (ANTLR) | most unambiguous CFGs | $`O(n)`$ typical, $`O(n^4)`$ worst | good tooling | DSLs, quick development |
| PEG / packrat | PEGs (ordered choice) | $`O(n)`$ with memoisation (memory heavy) | intuitive, but hidden shadowing | DSLs, scannerless parsing |
| Earley | all CFGs | $`O(n^3)`$, $`O(n^2)`$ unambiguous, $`O(n)`$ for most LR | handles ambiguity | ambiguous grammars, grammar prototyping |
| GLR | all CFGs | $`O(n^3)`$ worst, near-linear in practice | parse forests | C++-like ambiguity, language workbenches |
| CYK | CFG in Chomsky normal form | $`O(n^3)`$ | simple DP | theory, probabilistic grammars |
| Incremental (tree-sitter) | GLR-based | proportional to edit | error-tolerant | editors, syntax highlighting |

## Variants

### Regular expressions and automata

1. *Thompson's construction:* regex → ε-NFA with $`O(m)`$ states.
2. *Subset construction:* NFA → DFA (states = sets of NFA states); exponential worst case, usually fine; or simulate the NFA directly in $`O(nm)`$.
3. *Minimisation:* Hopcroft's algorithm, $`O(k\log k)`$.
4. Lexer: longest match + priority among token rules.

- Linear-time engines (RE2, Rust `regex`) guarantee no catastrophic backtracking; backreferences make matching NP-hard.

### Context-free grammars

- Productions in BNF; derivations produce parse trees; *ambiguity* = multiple trees for one string (dangling else, operator precedence).
- Remove ambiguity by grammar restructuring or precedence/associativity declarations.

### Recursive descent and Pratt parsing

- One function per nonterminal; lookahead tokens choose the alternative. Left recursion must be rewritten as iteration.
- *Pratt parsing* for expressions: each token has a prefix handler (nud) and an infix handler (led) with a binding power:
  1. Parse a prefix expression for the current token (literal, unary operator, parenthesis).
  2. While the next token is an infix operator whose left binding power exceeds the current minimum: consume it and parse the right operand with its right binding power.
  3. Associativity via left vs right binding powers.

### LR parsing

- Shift-reduce automaton built from LR(0)/LR(1) item sets; LALR(1) merges states with equal cores (smaller tables, occasional spurious conflicts).
- *Conflicts* (shift/reduce, reduce/reduce) signal ambiguity or insufficient lookahead; resolve by precedence declarations or grammar changes.
- Handles left recursion naturally; error recovery via error productions or automatic repair (Menhir's tools help).

### PEGs and packrat

- Ordered choice `A / B`: try A, use B only if A fails. Unambiguous by definition but can mask alternatives (`"a" / "ab"` never matches `ab` fully).
- Packrat memoisation of (rule, position) gives linear time; left recursion needs special support.

### General CFG parsing

- *Earley:* chart parsing with predict/scan/complete; handles any CFG, efficient on unambiguous grammars.
- *GLR:* LR with forking stacks on conflicts and graph-structured stack sharing; produces shared packed parse forests.

### Error recovery

- Panic mode (skip to synchronising tokens like `;`, `}`), phrase-level repairs, error productions; for editors, produce partial trees with error nodes.
- Good messages: expected-token sets, matching delimiter tracking, hand-written diagnostics for common mistakes.

## Other lenses

- *Algebra.* Regular languages ↔ finite automata ↔ regular expressions (Kleene's theorem); regex operations form a Kleene algebra; CYK/Earley are semiring dynamic programs (see [algebraic structures](/ref/programs/algebraic-structures)).
- *Coalgebra.* Automata are coalgebras (state → output × next state); Brzozowski derivatives of regexes give automata directly (see [algebraic data types](/ref/programs/algebraic-data-types)).
- *Probability / ML.* Probabilistic CFGs and HMMs use the same chart algorithms with probability semirings (inside–outside, Viterbi); tokenisers for language models are learned lexers.

## Pitfalls & diagnostics

- *Catastrophic regex backtracking* (ReDoS) with nested quantifiers on backtracking engines.
- *Left recursion* in recursive descent / naive PEGs loops forever.
- *Hidden ambiguity* in PEGs (ordered choice) and in grammars with precedence declarations; test with tricky inputs.
- *Error messages* are a product feature; generated parsers need deliberate effort to produce good ones.
- *Lexer/parser interaction* (context-sensitive tokens, significant whitespace, nested comments) often needs lexer modes or a scannerless approach.

## See also

- [Algebraic data types](/ref/programs/algebraic-data-types)
- [Term rewriting & e-graphs](/ref/programs/term-rewriting-egraphs)
- [Algebraic structures](/ref/programs/algebraic-structures)
- [SSA & intermediate representations](/ref/programs/ssa-ir)

*References:*

- Aho, Lam, Sethi & Ullman, _Compilers: Principles, Techniques, and Tools_ (2nd ed.).
- Grune & Jacobs, _Parsing Techniques: A Practical Guide_ (2nd ed.).
- Cox, "Regular expression matching can be simple and fast" (2007).
