---
title: 'Homogeneous coordinates & projections'
description: 'Affine transforms as 4×4 matrices, model/view/projection pipelines, perspective and orthographic projections, normal matrices, depth precision and reverse-Z.'
aliases:
  [
    homogeneous coordinates,
    affine transformation,
    4x4 matrix,
    model matrix,
    view matrix,
    projection matrix,
    MVP,
    look-at,
    perspective projection,
    orthographic projection,
    perspective divide,
    NDC,
    normalized device coordinates,
    viewport transform,
    normal matrix,
    depth buffer,
    z-fighting,
    reverse-Z,
    camera intrinsics,
    pinhole camera,
    projective geometry,
    transform hierarchy,
    scene graph
  ]
tags: [geometry, graphics, computer vision]
order: 3
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Homogeneous coordinates** $(x, y, z, w)$ make translations, rotations, scales and projections all **4×4 matrix products**; points have $w = 1$, directions $w = 0$.
- **Pipeline:** object → world (model $M$) → camera (view $V$) → clip (projection $P$) → perspective divide → NDC → viewport. Combined $PVM$, applied right to left (column-vector convention).
- **Normals** transform with the inverse transpose of the upper-left 3×3: $(M^{-1})^\top$.
- **Depth precision:** perspective depth is hyperbolic; use **reverse-Z** with a floating-point depth buffer (and an infinite far plane) to avoid z-fighting.
- **Computer vision** uses the same projective geometry with camera intrinsics $K$ and extrinsics $[R\mid t]$.

## Decision table

| Transform               | Matrix (column vectors)                                                    | Notes                                                               |
| ----------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Translation by $t$      | $\begin{pmatrix} I & t \\ 0 & 1\end{pmatrix}$                              | needs $w = 1$; doesn't affect directions                            |
| Rotation $R$            | $\begin{pmatrix} R & 0 \\ 0 & 1\end{pmatrix}$                              | see [rotations](/ref/space/rotations)                               |
| Scale $S$               | $\operatorname{diag}(s_x, s_y, s_z, 1)$                                    | non-uniform scale breaks normals                                    |
| Rigid motion            | $\begin{pmatrix} R & t \\ 0 & 1\end{pmatrix}$                              | inverse: $\begin{pmatrix} R^\top & -R^\top t \\ 0 & 1\end{pmatrix}$ |
| View (look-at)          | inverse of camera's rigid pose                                             | world → camera                                                      |
| Orthographic projection | scale + translate box to $[-1, 1]^3$                                       | CAD, shadows maps, 2D                                               |
| Perspective projection  | maps frustum to clip space, $w_{\text{clip}} = -z_{\text{eye}}$            | divide by $w$ gives foreshortening                                  |
| Normal matrix           | $(M_{3\times3}^{-1})^\top$                                                 | equals $M_{3\times3}$ for rotations only                            |
| Camera intrinsics $K$   | $\begin{pmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1\end{pmatrix}$ | pixel coordinates from camera rays                                  |

## Variants

### Homogeneous coordinates

- Point $(x, y, z) \mapsto (x, y, z, 1)$; any $(x, y, z, w)$ with $w \ne 0$ represents $(x/w, y/w, z/w)$.
- Directions/vectors use $w = 0$, so translations don't affect them.
- Composition = matrix product; order matters. With column vectors $p' = M_3M_2M_1p$ applies $M_1$ first. (Row-vector conventions transpose everything.)

### Perspective projection

- Symmetric frustum with vertical field of view $\phi$, aspect $a$, near $n$, far $f$ (OpenGL-style, right-handed, looking down $-z$):
  - $x_c = \frac{\cot(\phi/2)}{a}x_e$, $y_c = \cot(\phi/2)\,y_e$, $w_c = -z_e$.
  - $z_c$ maps $[n, f]$ to $[-1, 1]$ (OpenGL) or $[0, 1]$ (Direct3D/Vulkan/Metal) after division.
- **Perspective divide** $x_{\text{ndc}} = x_c/w_c$ etc.; clipping happens before the divide in clip space.
- **Viewport:** NDC → pixel coordinates.

### Depth precision and reverse-Z

- Stored depth $\propto 1/z$: most precision is near the near plane; distant surfaces z-fight.
- **Reverse-Z:** map near → 1, far → 0 with a float32 depth buffer; float's denser spacing near 0 cancels the $1/z$ distribution, giving nearly uniform relative precision. Allows an infinite far plane.
- Also: push the near plane as far out as possible; it matters far more than the far plane.

### Normals

- Normals must stay perpendicular to surfaces: transform with $(M^{-1})^\top$ and renormalise.
- For rotations + uniform scale, $M$ itself works (up to normalisation).

### Pinhole camera model (vision)

- $\lambda\,(u, v, 1)^\top = K[R\mid t]\,(X, Y, Z, 1)^\top$.
- Calibration estimates $K$ and lens distortion; pose estimation (PnP) finds $R, t$; triangulation and epipolar geometry (fundamental/essential matrices) relate multiple views.

## Other lenses

- **Projective geometry.** Homogeneous coordinates represent points at infinity ($w = 0$) and make projective transformations linear; lines and planes are covectors (they transform with the inverse transpose, like normals).
- **Lie groups.** Rigid motions form $SE(3)$; optimisation over camera poses uses Lie-algebra increments (see [manifold optimisation](/ref/optimise/manifold-optimisation)).
- **Floating point.** Depth buffer behaviour is a floating-point distribution story; large world coordinates lose precision too (camera-relative rendering) (see [floating-point](/ref/reliable/floating-point)).

## Pitfalls & diagnostics

- **Convention soup:** handedness, row vs column vectors, NDC depth range $[-1, 1]$ vs $[0, 1]$, matrix storage order (column-major in GLSL). Check each when porting.
- **Non-uniform scale and normals:** lighting looks wrong if the normal matrix isn't used.
- **Precision far from the origin** (planetary scales, large CAD models): use camera-relative coordinates or double precision on the CPU.
- **Inverting general 4×4 matrices** when a rigid-motion inverse ($R^\top$, $-R^\top t$) suffices: slower and less accurate.

## See also

- [Rotations](/ref/space/rotations)
- [Geometry primitives](/ref/space/geometry-primitives)
- [Vectors, tensors & einsum](/ref/space/vectors-tensors-einsum)
- [Floating-point arithmetic](/ref/reliable/floating-point)

**References:**

- Akenine-Möller et al., _Real-Time Rendering_ (4th ed.).
- Hartley & Zisserman, _Multiple View Geometry in Computer Vision_ (2nd ed.).
- Reed, "Depth precision visualized" (NVIDIA developer blog, 2015).
