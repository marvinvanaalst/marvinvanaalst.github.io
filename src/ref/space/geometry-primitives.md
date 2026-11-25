---
title: 'Geometric primitives & predicates'
description: 'Barycentric coordinates, ray–triangle and other intersections, distances, orientation and in-circle predicates, robustness with exact arithmetic, and polygon basics.'
aliases:
  [
    computational geometry,
    barycentric coordinates,
    ray-triangle intersection,
    Moller-Trumbore,
    ray-box intersection,
    slab method,
    point in polygon,
    winding number,
    orientation predicate,
    orient2d,
    orient3d,
    incircle,
    robust predicates,
    exact arithmetic,
    Shewchuk,
    adaptive precision,
    signed area,
    convex hull,
    Delaunay triangulation,
    closest point,
    signed distance function,
    epsilon tolerance
  ]
tags: [geometry, graphics, robustness]
order: 4
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Barycentric coordinates* describe points relative to a triangle/simplex: interpolation, inside tests, ray hits.
- *Orientation predicate* (sign of a determinant = signed area/volume) is the core of convex hulls, Delaunay triangulation, segment intersection and point-in-polygon. Evaluate it *robustly* (Shewchuk's adaptive exact predicates), not with an epsilon.
- *Ray–triangle:* Möller–Trumbore; ray–box: slab method. Accelerate many queries with BVHs (see [spatial indexes](/ref/graphs/spatial-indexes)).
- *Point in polygon:* winding number (handles non-simple polygons); ray-crossing for simple ones.
- Use tolerances only for _modelling_ decisions (snapping), never to decide combinatorial predicates.

## Decision table

| Task | Method | Cost | Robustness notes |
| --- | --- | --- | --- |
| Which side of a line / plane | orient2d / orient3d determinant | $`O(1)`$ | exact adaptive predicates |
| Point in circle / sphere | incircle / insphere determinant | $`O(1)`$ | exact adaptive predicates (Delaunay) |
| Ray–triangle | Möller–Trumbore (or watertight Woop et al.) | $`O(1)`$ | watertight variant avoids cracks between triangles |
| Ray–axis-aligned box | slab method | $`O(1)`$ | handle division by zero with IEEE infinities |
| Segment–segment intersection | orientation tests | $`O(1)`$ | exact predicates; compute point separately |
| Point in polygon | winding number / crossing number | $`O(n)`$ | points on edges need a policy |
| Closest point on triangle | region classification via barycentrics | $`O(1)`$ | Ericson's routine |
| Polygon area / centroid | shoelace formula | $`O(n)`$ | signed area gives orientation |
| Convex hull (2D / 3D) | Andrew's monotone chain / quickhull | $`O(n\log n)`$ | relies on exact orientation |
| Delaunay triangulation | Bowyer–Watson / divide and conquer | $`O(n\log n)`$ | relies on exact incircle |
| Distance to complex shapes | signed distance functions, BVH queries | varies | — |

## Variants

### Barycentric coordinates

- $`p = \lambda_0a + \lambda_1b + \lambda_2c`$ with $`\sum\lambda_i = 1`$; inside iff all $`\lambda_i \ge 0`$.
- $`\lambda_i`$ = ratios of signed sub-triangle areas (or volumes in 3D).
- Interpolate any vertex attribute linearly: $`f(p) = \sum\lambda_if_i`$ (finite elements, texture coordinates, normals). Perspective-correct interpolation in rendering divides by $`w`$.

### Orientation and in-circle predicates

- $`\operatorname{orient2d}(a, b, c) = \det\begin{pmatrix} a_x - c_x & a_y - c_y \\ b_x - c_x & b_y - c_y\end{pmatrix}`$: positive = counterclockwise.
- $`\operatorname{incircle}(a, b, c, d)`$: sign of a $`3\times3`$ (lifted) determinant; positive = $`d`$ inside the circumcircle of counterclockwise $`a, b, c`$.
- *Robustness:* near-degenerate inputs make floating-point signs wrong, which breaks algorithms (inconsistent decisions → crashes or invalid meshes).
- *Shewchuk's adaptive predicates:* compute in floating point with an error bound; if the result is too close to zero, refine with exact expansion arithmetic. Usually as fast as naive code.
- *Simulation of simplicity* handles exactly degenerate cases consistently by symbolic perturbation.

### Ray–triangle (Möller–Trumbore)

1. Edges $`e_1 = b - a`$, $`e_2 = c - a`$; $`p = d\times e_2`$; $`\det = e_1\cdot p`$ (≈ 0 → parallel).
2. $`s = o - a`$; $`u = (s\cdot p)/\det`$; reject if outside $`[0, 1]`$.
3. $`q = s\times e_1`$; $`v = (d\cdot q)/\det`$; reject if $`v \lt 0`$ or $`u + v \gt 1`$.
4. $`t = (e_2\cdot q)/\det`$; hit if $`t \gt t_{\min}`$.

- Watertight variants guarantee rays hitting a shared edge register exactly one hit.

### Ray–box (slab method)

- For each axis, $`t_{1,2} = (\text{min}_i - o_i)/d_i`$, $`(\text{max}*i - o_i)/d_i`$; intersect intervals $`[t*{\text{near}}, t_{\text{far}}]`$ over axes; hit if $`t_{\text{near}} \le t_{\text{far}}`$ and $`t_{\text{far}} \ge 0`$. Precompute $`1/d`$.

### Polygons

- *Shoelace:* $`A = \tfrac12\sum_i(x_iy_{i+1} - x_{i+1}y_i)`$ (signed: positive for counterclockwise).
- *Winding number:* sum of signed crossings of a ray (or total angle); non-zero = inside; correct for self-intersecting polygons.

## Other lenses

- *Linear algebra.* Predicates are determinant signs; barycentric coordinates solve small linear systems; Delaunay = lower convex hull of points lifted to a paraboloid.
- *Floating point.* Robust geometry is a showcase for exact and adaptive-precision arithmetic, error-free transformations and interval filters (see [interval arithmetic](/ref/reliable/interval-affine-arithmetic), [summation](/ref/reliable/summation-reductions)).
- *Meshing and FEM.* Delaunay triangulations and quality meshes underpin finite elements; barycentric/shape functions are the FE basis (see [method of lines](/ref/integrate/pdes-method-of-lines)).

## Pitfalls & diagnostics

- *Epsilon comparisons* in combinatorial decisions create inconsistencies (A left of B, B left of C, C left of A); use exact predicates.
- *Self-intersection* due to rounding of constructed points (intersection coordinates are not exactly representable); keep predicates on input points or use exact constructions where it matters.
- *Degenerate inputs* (collinear, cocircular points, zero-area triangles) are common in real data (grids, CAD); test them explicitly.
- *Units and scale:* absolute tolerances break when coordinates span many orders of magnitude.

## See also

- [Spatial indexes](/ref/graphs/spatial-indexes)
- [Homogeneous coordinates & projections](/ref/space/homogeneous-projections)
- [Interval & affine arithmetic](/ref/reliable/interval-affine-arithmetic)
- [Floating-point arithmetic](/ref/reliable/floating-point)

*References:*

- de Berg, Cheong, van Kreveld & Overmars, _Computational Geometry: Algorithms and Applications_ (3rd ed.).
- Shewchuk, "Adaptive precision floating-point arithmetic and fast robust geometric predicates", Discrete Comput. Geom. (1997).
- Ericson, _Real-Time Collision Detection_.
