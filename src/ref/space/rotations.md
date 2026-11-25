---
title: 'Rotations'
description: 'Rotation matrices, Euler angles and gimbal lock, axis-angle and Rodrigues, quaternions and slerp, and SO(3) exp/log maps for interpolation, integration and optimisation.'
aliases:
  [
    rotation matrix,
    orthogonal matrix,
    SO(3),
    Euler angles,
    Tait-Bryan,
    gimbal lock,
    axis-angle,
    rotation vector,
    Rodrigues formula,
    quaternion,
    unit quaternion,
    slerp,
    nlerp,
    Lie algebra,
    so(3),
    exponential map,
    logarithm map,
    angular velocity,
    Kabsch algorithm,
    Procrustes,
    rotation averaging,
    6D rotation representation,
    dual quaternions
  ]
tags: [geometry, graphics, robotics, Lie groups]
order: 2
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Store and compose* with unit quaternions (4 numbers, cheap composition, easy renormalisation) or rotation matrices (9 numbers, direct to apply, needed by GPUs/BLAS).
- *Interpolate* with slerp (quaternions) or $`R_0\exp(t\log(R_0^\top R_1))`$.
- *Optimise / integrate* with local updates in the Lie algebra: $`R \leftarrow R\exp(\widehat{\omega})`$, $`\omega \in\mathbb R^3`$. Avoid Euler angles in numerics (gimbal lock).
- *Euler angles* only for human input/output, with the convention stated (order, intrinsic vs extrinsic).
- *Best rotation aligning point sets:* Kabsch / orthogonal Procrustes via SVD.
- *Neural network outputs:* continuous 6D representation (two columns + Gram–Schmidt) rather than quaternions or Euler angles.

## Decision table

| Representation | Parameters | Compose | Apply to vector | Singularities / ambiguity | Use for |
| --- | --- | --- | --- | --- | --- |
| Rotation matrix | 9 (3 DOF) | matmul (27 mult.) | matvec (9 mult.) | none; drift off $`SO(3)`$ | applying to many vectors, GPU, linear algebra |
| Unit quaternion | 4 | 16 mult. | \~30 mult. (or convert) | double cover ($`q`$ and $`-q`$) | storage, composition, interpolation |
| Axis-angle / rotation vector | 3 | via conversion | Rodrigues | angle $`\pi`$ ambiguity, 0 axis undefined | optimisation increments, angular velocity |
| Euler angles | 3 | via conversion | via conversion | gimbal lock | human-readable input/output only |
| 6D (two columns) | 6 | via conversion | via conversion | none (continuous) | neural network regression targets |
| Dual quaternions | 8 | fast | fast | double cover | rigid motions (rotation + translation), skinning |

## Variants

### Rotation matrices

- $`R^\top R = I`$, $`\det R = 1`$. Compose by multiplication (not commutative).
- *Re-orthonormalise* after many updates: polar decomposition / SVD ($`R \leftarrow UV^\top`$) or Gram–Schmidt.
- Elementary rotations about axes for constructing from Euler angles.

### Euler angles and gimbal lock

- Three successive rotations about axes (e.g. ZYX yaw–pitch–roll). 12 conventions × intrinsic/extrinsic.
- *Gimbal lock:* when the middle angle is ±90°, two axes align and one degree of freedom is lost; derivatives blow up, interpolation and optimisation misbehave.

### Axis-angle and Rodrigues

- Rotation vector $`\omega = \theta\hat n`$.
- *Rodrigues:* $`R = I + \sin\theta\,K + (1 - \cos\theta)K^2`$ with $`K = \widehat{\hat n}`$ (skew-symmetric cross-product matrix).
- *Log map:* $`\theta = \arccos\big((\operatorname{tr}R - 1)/2\big)`$, axis from $`R - R^\top`$; numerically careful near $`\theta = 0`$ (Taylor expansions) and $`\theta = \pi`$ (use the symmetric part).

### Quaternions

- $`q = (w, x, y, z) = (\cos\tfrac\theta2, \hat n\sin\tfrac\theta2)`$, $`\lVert q\rVert = 1`$; rotate $`v`$ by $`qvq^*`$.
- Composition = Hamilton product; inverse = conjugate. $`q`$ and $`-q`$ represent the same rotation.
- *Slerp:* $`q(t) = \dfrac{\sin((1 - t)\Omega)}{\sin\Omega}q_0 + \dfrac{\sin(t\Omega)}{\sin\Omega}q_1`$ with $`\cos\Omega = q_0\cdot q_1`$; flip $`q_1 \to -q_1`$ if the dot product is negative (shortest path); fall back to normalised lerp for small $`\Omega`$.

### Lie group view

- $`SO(3)`$ is a Lie group; its algebra $`\mathfrak{so}(3)`$ = skew-symmetric matrices ≅ $`\mathbb R^3`$.
- $`\exp`$ = Rodrigues, $`\log`$ = inverse. Local perturbations $`R\exp(\widehat{\delta})`$ give unconstrained 3-parameter increments for Gauss–Newton, Kalman filters (error-state EKF) and integrators.
- *Kinematics:* $`\dot R = R\widehat{\omega}`$ (body angular velocity); integrate with $`R_{k+1} = R_k\exp(h\widehat{\omega})`$ to stay on $`SO(3)`$ (see [geometric integrators](/ref/integrate/geometric-integrators)).

### Alignment: Kabsch / Procrustes

1. Centre both point sets.
2. Cross-covariance $`H = \sum_ip_iq_i^\top`$.
3. SVD $`H = U\Sigma V^\top`$; $`R = V\operatorname{diag}(1, 1, \det(VU^\top))U^\top`$ (the determinant fixes reflections).

- Used for RMSD in structural biology, point cloud registration (inner step of ICP).

## Other lenses

- *Manifold optimisation.* Rotations live on a curved manifold; Riemannian methods with retractions are the principled way to optimise over them (see [manifold optimisation](/ref/optimise/manifold-optimisation)).
- *Matrix functions.* $`\exp`$ and $`\log`$ of skew-symmetric matrices are special cases of matrix exponentials/logarithms with closed forms in 3D (see [matrix functions](/ref/solve/matrix-functions)).
- *Algebra.* Quaternions form a division algebra; unit quaternions $`SU(2)`$ double-cover $`SO(3)`$; geometric algebra (rotors) generalises this to any dimension.

## Pitfalls & diagnostics

- *Convention mismatches* (active vs passive, row vs column vectors, quaternion component order $`wxyz`$ vs $`xyzw`$, intrinsic vs extrinsic Euler) cause most rotation bugs. Test with known 90° rotations.
- *Averaging rotations* by averaging matrices or quaternion components is wrong in general; use chordal mean + projection, or iterative Karcher mean.
- *Quaternion sign flips* break interpolation and filters; enforce a consistent hemisphere.
- *Drift:* renormalise quaternions / re-orthonormalise matrices periodically.
- *Small-angle numerics* in $`\log`$/$`\exp`$ and slerp: use series expansions.

## See also

- [Homogeneous coordinates & projections](/ref/space/homogeneous-projections)
- [Manifold optimisation](/ref/optimise/manifold-optimisation)
- [Geometric integrators](/ref/integrate/geometric-integrators)
- [Vectors, tensors & einsum](/ref/space/vectors-tensors-einsum)

*References:*

- Diebel, "Representing attitude: Euler angles, unit quaternions, and rotation vectors" (2006).
- Solà, "Quaternion kinematics for the error-state Kalman filter" (2017).
- Zhou et al., "On the continuity of rotation representations in neural networks", CVPR (2019).
