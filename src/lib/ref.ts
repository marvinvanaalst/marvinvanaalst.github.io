import type { RefSection } from './types';

/** Reference sections, ordered roughly by dependency. Ids are the src/ref/<id>/ folders. */
export const sections: RefSection[] = [
	{
		id: 'reliable',
		title: 'Compute reliably',
		blurb: 'Floating point, conditioning, stable summation, interval arithmetic.'
	},
	{
		id: 'solve',
		title: 'Solve equations',
		blurb:
			'Linear systems, sparse, iterative and fast solvers, least squares, roots, eigenproblems.'
	},
	{
		id: 'diff',
		title: 'Differentiate',
		blurb: 'Finite differences, automatic differentiation, ODE sensitivities, matrix calculus.'
	},
	{
		id: 'integrate',
		title: 'Integrate & evolve',
		blurb:
			'Quadrature, ODEs, DAEs, SDEs, PDEs via method of lines, stochastic and parallel-in-time.'
	},
	{
		id: 'dynamics',
		title: 'Analyse dynamics',
		blurb:
			'Steady states, bifurcations, model reduction, sensitivity, stochastic processes, reaction networks.'
	},
	{
		id: 'approx',
		title: 'Approximate functions',
		blurb:
			'Interpolation, Chebyshev and Fourier, low-rank, GPs, surrogates, sparsity, high dimensions.'
	},
	{
		id: 'optimise',
		title: 'Optimise & fit',
		blurb:
			'Local, constrained, global, stochastic and integer optimisation; control; fitting ODEs and NODEs.'
	},
	{
		id: 'random',
		title: 'Generate randomness',
		blurb: 'PRNGs, sampling distributions, quasi-Monte Carlo, variance reduction.'
	},
	{
		id: 'bayes',
		title: 'Infer (Bayesian)',
		blurb: 'Priors, MCMC, HMC, SMC, variational and likelihood-free inference, model comparison.'
	},
	{
		id: 'freq',
		title: 'Infer (frequentist)',
		blurb:
			'MLE, confidence intervals, identifiability, tests, regression, experimental design, conformal.'
	},
	{
		id: 'signals',
		title: 'Estimate state & signals',
		blurb: 'Kalman and particle filters, filtering, spectral estimation, time-series models.'
	},
	{
		id: 'info',
		title: 'Information & distances',
		blurb: 'Entropy, KL, mutual information, Fisher information, distances between distributions.'
	},
	{
		id: 'learn',
		title: 'Learn from data',
		blurb: 'Classical ML, neural network building blocks, transformers, generative models.'
	},
	{
		id: 'explain',
		title: 'Explain & reason causally',
		blurb: 'Feature attribution, structural causal models, effect estimation, discovery.'
	},
	{
		id: 'space',
		title: 'Transform space',
		blurb: 'Tensors and einsum, rotations and quaternions, projections, geometric predicates.'
	},
	{
		id: 'programs',
		title: 'Structure programs',
		blurb: 'Category theory, types, fixpoints, rewriting, IRs, loop optimisation, staging, parsing.'
	},
	{
		id: 'graphs',
		title: 'Search graphs & index space',
		blurb:
			'Traversal, shortest paths, flows, union-find; k-d trees, BVHs and other spatial indexes.'
	}
];

export function findSection(id: string): RefSection | undefined {
	return sections.find((section) => section.id === id);
}
