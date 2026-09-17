# Gravity-Aligned Descriptors in OKVIS2-X

How OKVIS2-X orients BRISK descriptors along gravity, and why the orientation is a per-keypoint $2\times2$ affine warp rather than a scalar angle.

## 1. The problem

A rotation-invariant descriptor estimates its own orientation from local image gradients and normalises it away. That works, but it is both fragile (the gradient direction is noisy on weak or symmetric patches) and wasteful (orientation is thrown away, so a descriptor can no longer distinguish a patch from its rotated copy).

A visual-inertial system does not need to guess. The IMU gives an absolute attitude, so the direction of gravity in the camera frame is known before the image is even processed. Anchoring every descriptor to that direction gives a canonical orientation that is *shared across frames, across cameras and across sessions*, without spending any image information on it.

The complication: gravity is a single 3D direction, and a descriptor needs a 2D orientation *in the image*. For a narrow-FoV pinhole those are almost the same thing. In general they are not — where "down" points in pixels depends on **which pixel**, because the projection of a fixed 3D direction varies over the image. So the orientation must be recomputed per keypoint, from the camera model. Once that machinery exists, it costs nothing extra to let it also absorb lens distortion, which is what OKVIS does.

## 2. Notation

| Symbol | Meaning | Code |
|---|---|---|
| $\pi:\mathbb R^3\to\mathbb R^2$ | Camera projection, $\mathbf u=\pi(\mathbf x)$, $\mathbf x$ in camera frame | `CameraBase::project` |
| $J(\mathbf x)=\partial\pi/\partial\mathbf x\in\mathbb R^{2\times3}$ | Projection Jacobian | `imageJacobians_` |
| $\mathbf d\in S^2$ | Unit bearing of the keypoint in the camera frame | `rays_` |
| $\mathbf g\in S^2$ | Gravity direction in the camera frame | `extractionDirection_` |
| $\mathbf p_k=(x_k,y_k)$, $\sigma_k$ | BRISK pattern point and its blur radius, canonical pixel units | `patternPoints_` |
| $f$ | Virtual focal length | `virtualFocalLength_` |
| $P_{\mathbf d}=I-\mathbf d\mathbf d^\top$ | Orthogonal projector onto $\mathbf d^\perp$ | — |
| $W\in\mathbb R^{2\times2}$ | Pattern warp | `warp` |

## 3. Getting gravity into the camera frame

### 3.1 Attitude from IMU propagation

Detection needs a pose *before* the frame has been processed, so the estimator is not consulted. [ThreadedSlam.cpp:598-610](okvis_multisensor_processing/src/ThreadedSlam.cpp#L598-L610) takes the last optimised state and integrates forward:

```cpp
T_WS = lastOptimisedState_.T_WS;
speedAndBias.head<3>()     = lastOptimisedState_.v_W;
speedAndBias.segment<3>(3) = lastOptimisedState_.b_g;
speedAndBias.tail<3>()     = lastOptimisedState_.b_a;
ceres::ImuError::propagation(imuMeasurementDeque_, parameters_.imu, T_WS, speedAndBias,
                             lastOptimisedState_.timestamp, multiFrame->timestamp());
```

with `T_WC = T_WS * T_SC(i)` per camera. Only the *rotation* of this prediction is used, so propagation drift over one frame interval is irrelevant — attitude is anchored by gravity in the accelerometer and does not drift the way position does.

Bootstrap cases: before the first optimised state, attitude comes from `ImuError::initPose` (averaged accelerometer); with `imu.use == false` the camera is simply assumed upright.

### 3.2 Rotating into the camera frame

[Frontend.cpp:233-238](okvis_frontend/src/Frontend.cpp#L233-L238):

```cpp
Eigen::Vector3d g_in_W(0, 0, -1);
Eigen::Vector3d extractionDir = T_WC.inverse().C() * g_in_W;
briskExtractor->setExtractionDirection(cv::Vec3f(extractionDir[0], extractionDir[1], extractionDir[2]));
```

i.e. $\mathbf g = C_{CW}\,\mathbf g_W$ with $\mathbf g_W=(0,0,-1)$, since the OKVIS world frame has gravity along $-z$. This is set once per camera per frame, immediately before `detect()` / `describe()`.

The default if it were never set is `(0,1,0)` ([brisk-descriptor-extractor.h:199](external/brisk/include/brisk/brisk-descriptor-extractor.h#L199)) — the camera's own down-axis. So the fallback behaviour is body-upright alignment, not "no alignment".

### 3.3 Camera-awareness maps

Two per-pixel lookup tables are precomputed once per camera in [PinholeCamera.hpp:171-198](okvis_cv/include/okvis/cameras/implementation/PinholeCamera.hpp#L171-L198):

```cpp
backProject(Eigen::Vector2d(u,v), &ray);  ray.normalize();
rays_.at<cv::Vec3f>(v,u) = ray;                        // CV_32FC3
project(ray, &pt, &jacobian);
imageJacobians_.at<cv::Vec6f>(v,u) = jacobian;         // CV_32FC(6), row-major 2x3
```

`rays_` is the backprojected unit bearing per pixel; `imageJacobians_` is $J$ at that bearing. They are handed to BRISK by `setCameraProperties(rays, imageJacobians, focalLengthU)` — note the virtual focal length is the camera's *actual* $f_u$, not the 500 default.

## 4. The math

### 4.1 The exact fact that makes a $2\times2$ warp sufficient

For a central camera, $\pi(\lambda\mathbf x)=\pi(\mathbf x)$ for all $\lambda>0$. Differentiating in $\lambda$ at $\lambda=1$:

$$J(\mathbf d)\,\mathbf d = \mathbf 0 .$$

So $J$ has rank 2 with null space $\operatorname{span}(\mathbf d)$, and its restriction

$$J|_{\mathbf d^\perp}:\;\mathbf d^\perp \longrightarrow \mathbb R^2$$

is an isomorphism (for a non-degenerate camera). The $2\times3$ Jacobian carries no information beyond its action on the two-dimensional tangent space $\mathbf d^\perp = T_{\mathbf d}S^2$. Representing it as a $2\times2$ matrix in a basis of $\mathbf d^\perp$ is a change of representation, **not** an approximation.

### 4.2 The virtual tangent plane, and where the warp comes from

Model the descriptor pattern as living on a virtual pinhole image plane of focal length $f$ whose optical axis is the keypoint's own viewing ray $\mathbf d$. Given an orthonormal basis $\{\mathbf e_u,\mathbf e_v\}$ of $\mathbf d^\perp$, pattern coordinate $(x,y)$ names the 3D direction

$$\mathbf x(x,y)\;=\;\mathbf d+\frac{x}{f}\,\mathbf e_u+\frac{y}{f}\,\mathbf e_v .$$

Project that into the *real* image and expand to first order about $\mathbf d$:

$$
\mathbf u(x,y)=\pi\big(\mathbf x(x,y)\big)
=\underbrace{\pi(\mathbf d)}_{\mathbf u_{kp}}
+J(\mathbf d)\Big[\tfrac{x}{f}\mathbf e_u+\tfrac{y}{f}\mathbf e_v\Big]
+O(\lVert\mathbf p\rVert^2).
$$

Hence

$$
\boxed{\;\Delta\mathbf u=W\mathbf p,\qquad
W=\frac1f\big[\;J\mathbf e_u\;\;\;J\mathbf e_v\;\big]\in\mathbb R^{2\times2}.\;}
$$

That is [brisk-descriptor-extractor.cc:712](external/brisk/src/brisk-descriptor-extractor.cc#L712), `warp = J * eu_ev`, with the $1/f$ already folded into the basis vectors at [line 689-690](external/brisk/src/brisk-descriptor-extractor.cc#L689-L690). It is applied to every pattern point at [line 367](external/brisk/src/brisk-descriptor-extractor.cc#L367):

```cpp
briskPoint1.x = warp[0]*briskPoint0.x + warp[1]*briskPoint0.y;
briskPoint1.y = warp[2]*briskPoint0.x + warp[3]*briskPoint0.y;
```

The affine truncation is the *only* approximation in the scheme. The discarded $O(\lVert\mathbf p\rVert^2)$ term is also why using $J$ at the nearest-neighbour pixel (rather than interpolated) is acceptable, while the ray $\mathbf d$ *is* bilinearly interpolated — the ray sets the orientation and deserves subpixel accuracy, the Jacobian only sets a local stretch.

### 4.3 The roll degree of freedom — all that `extractionDirection_` does

Any two orthonormal bases of $\mathbf d^\perp$ differ by a rotation about $\mathbf d$. Choosing the basis chooses the descriptor's roll, and nothing else. The gravity anchor is [line 685](external/brisk/src/brisk-descriptor-extractor.cc#L685) — the only use of `extractionDirection_` anywhere:

$$\mathbf e_u=\frac{\mathbf g\times\mathbf d}{\lVert\mathbf g\times\mathbf d\rVert},
\qquad
\mathbf e_v=\mathbf d\times\mathbf e_u .$$

Expanding the second with the BAC–CAB identity:

$$\mathbf d\times(\mathbf g\times\mathbf d)=\mathbf g(\mathbf d\cdot\mathbf d)-\mathbf d(\mathbf d\cdot\mathbf g)=\big(I-\mathbf d\mathbf d^\top\big)\mathbf g=P_{\mathbf d}\,\mathbf g,$$

and since $\lVert\mathbf g\times\mathbf d\rVert=\sin\angle(\mathbf g,\mathbf d)=\lVert P_{\mathbf d}\mathbf g\rVert$,

$$\boxed{\;\mathbf e_v=\frac{P_{\mathbf d}\,\mathbf g}{\lVert P_{\mathbf d}\,\mathbf g\rVert}\;}$$

**The descriptor's $y$-axis is gravity orthogonally projected into the tangent plane of the viewing ray, normalised; its $x$-axis is the local horizon.**

Frame checks. $\mathbf e_u\perp\mathbf g$ and $\mathbf e_u\perp\mathbf d$, hence $\mathbf e_u\perp P_{\mathbf d}\mathbf g\parallel\mathbf e_v$, so the basis is orthonormal. Handedness:

$$\mathbf e_u\times\mathbf e_v=\mathbf e_u\times(\mathbf d\times\mathbf e_u)
=\mathbf d(\mathbf e_u\!\cdot\!\mathbf e_u)-\mathbf e_u(\mathbf e_u\!\cdot\!\mathbf d)=\mathbf d,$$

so $(\mathbf e_u,\mathbf e_v,\mathbf d)$ is right-handed with the viewing ray as third axis.

Because $\mathbf d$ varies across the image, two keypoints in the same frame get **different** image-plane orientations from the same $\mathbf g$. That is the behaviour a scalar angle cannot express, and it is what matters on wide-FoV lenses.

The reported keypoint orientation is the image direction of the second column of $W$ ([line 722](external/brisk/src/brisk-descriptor-extractor.cc#L722)):

$$\texttt{kp.angle}=\operatorname{atan2}(W_{11},W_{01})=\angle\big(J\mathbf e_v\big),$$

i.e. the projected gravity direction in pixels.

### 4.4 Reference case: $W=I$

Ideal pinhole, keypoint at the principal point, camera upright — $\mathbf d=(0,0,1)$, $\mathbf g=(0,1,0)$:

$$\mathbf e_u=\mathbf g\times\mathbf d=(1,0,0),\qquad
\mathbf e_v=(0,1,0),\qquad
J(\mathbf d)=\begin{bmatrix}f&0&0\\0&f&0\end{bmatrix}$$

$$W=\frac1f\begin{bmatrix}f&0&0\\0&f&0\end{bmatrix}
\begin{bmatrix}1&0\\0&1\\0&0\end{bmatrix}=I_2 .$$

The unmodified BRISK pattern. The $1/f$ normalisation exists precisely to make this the fixed point: $W$ departs from identity only by the amount the true geometry departs from an upright, centred, distortion-free pinhole. This also shows the warp is doing **two jobs in one matrix** — gravity alignment *and* camera-model compensation — so descriptors are comparable across cameras with different lenses without ever rectifying the image.

### 4.5 Conditioning, and the $\sim6°$ gate

The frame degenerates as $\mathbf d\to\pm\mathbf g$ (looking straight up or down): $P_{\mathbf d}\mathbf g\to\mathbf 0$ and the roll becomes undefined. Quantitatively, perturb the attitude estimate, $\mathbf g\to\mathbf g+\boldsymbol\delta$. Then $\delta(P_{\mathbf d}\mathbf g)=P_{\mathbf d}\boldsymbol\delta$, and the induced roll $\psi$ about $\mathbf d$ is the component of that change along $\mathbf e_u$ divided by the length being rotated:

$$\boxed{\;\delta\psi=\frac{\mathbf e_u^\top P_{\mathbf d}\boldsymbol\delta}{\lVert P_{\mathbf d}\mathbf g\rVert}
=\frac{\mathbf e_u^\top\boldsymbol\delta}{\sin\angle(\mathbf g,\mathbf d)}\;}$$

Attitude error is amplified into descriptor-orientation error by $1/\sin\angle$. The guard at [line 686](external/brisk/src/brisk-descriptor-extractor.cc#L686):

```cpp
if (eu[0]*eu[0]+eu[1]*eu[1]+eu[2]*eu[2] > 0.01) { directional = true; }
```

tests $\lVert\mathbf g\times\mathbf d\rVert^2=\sin^2\angle>0.01$, i.e. $\angle>\arcsin(0.1)=5.74°$, capping the amplification at $10\times$.

When the test fails, `directional` stays false and BRISK falls back to its classic orientation from long-pair intensity gradients (or $\theta=0$ if rotation invariance is off). Note the camera-model warp is **still applied** in that case — only the gravity anchoring is dropped, and the discretised rotation index $\theta$ comes back into play.

### 4.6 Blur scaling

An affine map $W$ turns an isotropic Gaussian of width $\sigma$ into an anisotropic one with covariance $\sigma^2WW^\top$, but the integral-image box approximation can only do isotropic blur. The code collapses it to a scalar at [line 718](external/brisk/src/brisk-descriptor-extractor.cc#L718):

$$\sigma'_k=\tfrac12\big(|\lambda_1|+|\lambda_2|\big)\,\sigma_k .$$

Caveat: $W$ is not symmetric in general, and `cv::eigen` is documented for symmetric input. The principled scalar would be the mean singular value $\tfrac12(s_1+s_2)$, or $\sqrt{|\det W|}$ for an area-preserving match. For mild distortion $W$ is close to a scaled rotation and all three agree, so this is a heuristic that happens to be tight in the regime it runs in.

### 4.7 What is finally sampled

$$I_k=\big(I*G_{\sigma'_k}\big)\big(\mathbf u_{kp}+W\,\mathbf p_k\big),
\qquad
\sigma'_k=\tfrac12(|\lambda_1|+|\lambda_2|)\,\sigma_k$$

with the rotation index forced to zero ([lines 794, 800](external/brisk/src/brisk-descriptor-extractor.cc#L794)):

```cpp
SmoothedIntensity<...>(image, _integral, x, y, scale, directional ? 0 : theta, i, warpptr, sigmaScale);
```

and the descriptor bits are the usual BRISK short-pair comparisons $b_{ij}=\mathbb 1[I_i>I_j]$. On the gravity-aligned path the precomputed rotation lookup table is never consulted: $W$ supplies orientation, distortion compensation and anisotropic scale in a single matrix.

## 5. Code map

| Step | Location |
|---|---|
| Attitude prediction $T_{WS}$ by IMU propagation | [ThreadedSlam.cpp:598-610](okvis_multisensor_processing/src/ThreadedSlam.cpp#L598-L610) |
| $\mathbf g=C_{CW}\mathbf g_W$, `setExtractionDirection` | [Frontend.cpp:233-238](okvis_frontend/src/Frontend.cpp#L233-L238) |
| `rays_`, `imageJacobians_` precomputation | [PinholeCamera.hpp:171-198](okvis_cv/include/okvis/cameras/implementation/PinholeCamera.hpp#L171-L198) |
| `setCameraProperties`, `setExtractionDirection` | [brisk-descriptor-extractor.h:112-122](external/brisk/include/brisk/brisk-descriptor-extractor.h#L112-L122) |
| Ray interpolation, tangent basis, warp, angle | [brisk-descriptor-extractor.cc:658-724](external/brisk/src/brisk-descriptor-extractor.cc#L658-L724) |
| Degeneracy gate | [brisk-descriptor-extractor.cc:686](external/brisk/src/brisk-descriptor-extractor.cc#L686) |
| Warp applied to pattern points | [brisk-descriptor-extractor.cc:365-371](external/brisk/src/brisk-descriptor-extractor.cc#L365-L371) |
| Sampling with `theta` bypassed | [brisk-descriptor-extractor.cc:790-801](external/brisk/src/brisk-descriptor-extractor.cc#L790-L801) |

## 6. Practical consequences

- **Descriptors are not rotation invariant, by design.** Roll information is retained rather than normalised away, so two patches differing only by roll remain distinguishable. This raises discriminative power, at the cost of a hard dependency on a usable attitude estimate.
- **Matching across cameras and across sessions works without a common image orientation.** Two cameras with different mountings and different lenses produce descriptors in the same gravity-anchored, virtual-pinhole frame. This is also what makes the DBoW2 loop-closure vocabulary transferable between setups.
- **Attitude errors enter linearly, amplified by $1/\sin\angle(\mathbf g,\mathbf d)$.** In normal forward-looking operation $\angle\approx90°$ and the amplification is $\approx1$; the failure mode is genuinely upward- or downward-looking keypoints, which is exactly what the gate excludes.
- **Camera-awareness maps are $O(\text{pixels})$ memory per camera** — `CV_32FC3` plus `CV_32FC(6)`, i.e. 36 bytes per pixel — built once at startup, indexed per keypoint at extraction time.
- **The rotation lookup table is still built and still used** by the non-directional fallback path, so the memory cost of BRISK's `n_rot_` discretisation is not recovered.
