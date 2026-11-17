---
title: 'Unsupervised learning'
description: 'k-means, Gaussian mixtures and EM, hierarchical and density-based clustering, and dimensionality reduction for visualisation (PCA, t-SNE, UMAP).'
aliases:
  [
    clustering,
    k-means,
    k-means++,
    Gaussian mixture model,
    GMM,
    expectation maximization,
    EM algorithm,
    hierarchical clustering,
    agglomerative clustering,
    linkage,
    dendrogram,
    DBSCAN,
    HDBSCAN,
    spectral clustering,
    Leiden,
    Louvain,
    dimensionality reduction,
    PCA,
    t-SNE,
    UMAP,
    silhouette score,
    elbow method,
    anomaly detection,
    autoencoder
  ]
tags: [machine learning, clustering, dimensionality reduction]
order: 2
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Clustering:**
  - Roughly spherical, similar-size clusters, $k$ known: **k-means** (k-means++ init, several restarts).
  - Elliptical clusters, soft assignments, model selection by BIC: **Gaussian mixture** (EM).
  - Arbitrary shapes, noise points, unknown $k$: **HDBSCAN**.
  - Nested structure, small $n$: **hierarchical** clustering (Ward linkage).
  - Graph / single-cell data: **Leiden** on a kNN graph.
- **Dimensionality reduction:** PCA for linear structure and preprocessing; **UMAP** or **t-SNE** for 2D visualisation only (distances between clusters and cluster sizes are not meaningful).
- Clustering always returns clusters; validate stability and meaning.

## Decision table

| Method                       | Cost                              | Cluster shape              | $k$ needed      | Noise handling          | Use when                                   |
| ---------------------------- | --------------------------------- | -------------------------- | --------------- | ----------------------- | ------------------------------------------ |
| k-means                      | $O(nkd)$ per iteration            | convex, similar size       | yes             | no                      | large data, quick partition                |
| Mini-batch k-means           | cheaper                           | as k-means                 | yes             | no                      | huge data                                  |
| Gaussian mixture (EM)        | $O(nkd^2)$ per iteration          | ellipsoidal                | yes (BIC)       | no (or extra component) | soft assignments, density estimation       |
| Hierarchical (Ward, average) | $O(n^2)$ memory, $O(n^2\log n)$   | depends on linkage         | no (cut tree)   | no                      | small $n$, dendrograms                     |
| DBSCAN                       | $O(n\log n)$ with index           | arbitrary                  | no              | yes                     | density-separated clusters, single density |
| HDBSCAN                      | $O(n\log n)$                      | arbitrary, varying density | no              | yes                     | **default for unknown structure**          |
| Spectral clustering          | eigenvectors of a graph Laplacian | non-convex                 | yes             | no                      | manifold/graph structure, moderate $n$     |
| Leiden / Louvain             | near-linear                       | graph communities          | no (resolution) | no                      | kNN graphs (single-cell), networks         |
| PCA                          | $O(nd\min(n, d))$                 | —                          | —               | —                       | linear reduction, denoising, preprocessing |
| t-SNE                        | $O(n\log n)$ (Barnes–Hut / FFT)   | local structure            | —               | —                       | 2D visualisation of local neighbourhoods   |
| UMAP                         | ~$O(n^{1.14})$                    | local + some global        | —               | —                       | visualisation, faster than t-SNE           |
| Autoencoders                 | network training                  | nonlinear                  | —               | —                       | large data, nonlinear compression          |

## Variants

### k-means

1. Initialise centres with k-means++ (sample proportional to squared distance from existing centres).
2. Assign each point to the nearest centre.
3. Move centres to the means of their points.
4. Repeat until assignments stop changing; run several initialisations, keep the lowest within-cluster sum of squares.

- Minimises within-cluster variance; NP-hard globally, Lloyd's algorithm finds local optima.

### Gaussian mixtures and EM

- **E-step:** responsibilities $r_{ik} = \pi_k\mathcal N(x_i\mid\mu_k, \Sigma_k)/\sum_j\pi_j\mathcal N(x_i\mid\mu_j, \Sigma_j)$.
- **M-step:** update $\pi_k$, $\mu_k$, $\Sigma_k$ as responsibility-weighted averages.
- Monotonically increases the likelihood; local optima and degenerate components (variance → 0) need restarts and regularisation of covariances.
- Choose $k$ by BIC; covariance structure (full, diagonal, tied) trades flexibility vs parameters.

### Hierarchical clustering

- Agglomerative: start with singletons, repeatedly merge the closest pair. Linkage defines "closest": single (chaining), complete, average, **Ward** (minimum variance increase, usually best).
- Cut the dendrogram at a height or number of clusters.

### Density-based

- **DBSCAN:** core points have ≥ minPts neighbours within $\varepsilon$; clusters are connected core regions; others are noise. Single $\varepsilon$ struggles with varying density.
- **HDBSCAN:** builds a hierarchy over all $\varepsilon$ and extracts stable clusters; main parameter is the minimum cluster size.

### Visualisation embeddings

- **t-SNE:** matches neighbour probabilities (Gaussian in high-D, Student-t in 2D); perplexity ~30 sets the neighbourhood size; preserves local structure.
- **UMAP:** fuzzy simplicial set / kNN graph optimised in low dimensions; parameters n_neighbors (local vs global) and min_dist.
- Run on PCA-reduced data (e.g. top 50 PCs) for speed and noise reduction.

## Other lenses

- **Linear algebra.** PCA is an SVD; spectral clustering uses Laplacian eigenvectors; k-means is a low-rank (indicator) matrix factorisation (see [low-rank](/ref/approx/low-rank)).
- **Probability.** GMMs are latent-variable models; EM is coordinate ascent on the ELBO with exact posteriors (see [variational inference](/ref/bayes/variational-inference)).
- **Graphs.** Community detection (modularity, Leiden) and random-walk methods cluster networks; kNN graphs connect point clouds to graph algorithms (see [graph algorithms](/ref/graphs/graph-algorithms)).

## Pitfalls & diagnostics

- **Scaling:** distance-based methods depend on feature scales; standardise or use meaningful units.
- **Interpreting t-SNE/UMAP:** cluster sizes, distances between clusters and apparent continuity are artefacts of the method and parameters; never cluster on 2D embeddings for final conclusions.
- **Choosing $k$:** elbow plots are subjective; use BIC (GMM), silhouette, gap statistic, and most of all stability under resampling and domain meaning.
- **High dimensions:** distances concentrate; reduce dimension first (PCA) and use cosine distances where appropriate.
- **Batch effects** dominate clusters in experimental data; correct or check before interpreting.

## See also

- [Low-rank approximation](/ref/approx/low-rank)
- [Classical supervised learning](/ref/learn/classical-supervised)
- [Graph algorithms](/ref/graphs/graph-algorithms)
- [Spatial indexes](/ref/graphs/spatial-indexes) (neighbour search)

**References:**

- Bishop, _Pattern Recognition and Machine Learning_, ch. 9.
- McInnes, Healy & Melville, "UMAP: uniform manifold approximation and projection" (2018).
- Campello, Moulavi & Sander, "Density-based clustering based on hierarchical density estimates", PAKDD (2013).
