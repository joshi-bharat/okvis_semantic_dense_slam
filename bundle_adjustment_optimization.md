# Bundle Adjustment: Nonlinear Optimization, Schur Complement, and QR

These notes follow the derivation from pixel residuals to the sparse linear systems used in SfM and visual SLAM. Intrinsics are assumed known unless stated otherwise. Sections 1–9 are generic; [§10](#10-how-okvis2-x-implements-this) maps every step onto the OKVIS2-X implementation and lists where it deviates.

## 1. Notation and residuals

Use different counts for landmarks and observations: one landmark can be observed in many images.

| Symbol | Meaning | Dimension |
|---|---|---|
| $m$ | Number of camera poses | Scalar |
| $n$ | Number of 3D landmarks | Scalar |
| $N$ | Number of observed camera–landmark pairs | Scalar |
| $\mathbf X_j$ | Landmark in world coordinates | $3$ |
| $R_i,\mathbf t_i$ | World-to-camera pose | 6 local degrees of freedom |
| $\mathbf u_{ij}$ | Measured pixel | $2$ |
| $\mathbf r_{ij}$ | Predicted minus measured pixel | $2$ |

The camera-coordinate point and predicted pixel are

$$
\mathbf p_{ij}=R_i\mathbf X_j+\mathbf t_i,
\qquad
\hat{\mathbf u}_{ij}=\pi(K_i,\mathbf p_{ij}).
$$

For $\mathbf p=[x,y,z]^\top$, without distortion,

$$
\pi(K,\mathbf p)=
\begin{bmatrix}f_xx/z+c_x\\f_yy/z+c_y\end{bmatrix}.
$$

Each observation contributes two scalar residuals:

$$
\mathbf r_{ij}=\hat{\mathbf u}_{ij}-\mathbf u_{ij}.
$$

Stack observations into $\mathbf r\in\mathbb R^{2N}$. The objective is

$$
F(\boldsymbol\theta)=\frac12\mathbf r^\top\mathbf r
=\frac12\sum_{(i,j)\in\mathcal O}\|\mathbf r_{ij}\|^2.
$$

For example, residuals $[3,-2]^\top$ and $[-1,4]^\top$ stack into $[3,-2,-1,4]^\top$, giving $F=15$.

**LM minimizes a sum of squared residuals.** A mean squared error differs by a constant scaling and has the same minimizer. A least-squares solver normally receives individual residuals and their derivatives, rather than only a scalar MSE. Never average signed residuals before squaring: they can cancel. Scaling the objective can change LM steps if damping is not scaled consistently.

## 2. Linearization and the LM step

Around the current estimate,

$$
\mathbf r(\boldsymbol\theta\boxplus\Delta\boldsymbol\theta)
\approx\mathbf r+J\Delta\boldsymbol\theta.
$$

Here $\boxplus$ denotes an appropriate parameter update; rotations must stay on their manifold. For the concrete choice of $\boxplus$ used here — $\mathbb R^3\times SO(3)$ for poses, and homogeneous coordinates for landmarks — see [§10.2](#102-the-variables).

Gauss–Newton solves

$$
\min_{\Delta\boldsymbol\theta}\frac12\|\mathbf r+J\Delta\boldsymbol\theta\|^2.
$$

Differentiating with respect to the update gives

$$
J^\top(\mathbf r+J\Delta\boldsymbol\theta)=0,
$$

and therefore

$$
J^\top J\Delta\boldsymbol\theta=-J^\top\mathbf r.
$$

A common LM formulation adds damping:

$$
\boxed{(J^\top J+\lambda I)\Delta\boldsymbol\theta=-J^\top\mathbf r.}
$$

Some implementations use $\lambda D^\top D$ for parameter scaling instead. Solve the system rather than explicitly computing its inverse. LM evaluates trial steps against the nonlinear objective and adjusts damping; linearization and solution repeat until convergence.

Damping is not the only trust-region mechanism. A dogleg strategy solves the undamped Gauss–Newton system once per iteration and chooses the step along the path between the Cauchy point and the GN point, clipped to a trust-region radius — one factorization per iteration instead of a damping sweep. OKVIS2-X uses dogleg; see [§10.5](#105-the-linear-solve).

$J^\top J$ is the Gauss–Newton Hessian approximation. The exact Hessian also contains $\sum_k r_k\nabla^2r_k$.

## 3. Joint camera and landmark structure

Order the updates as cameras first, points second:

$$
\Delta\boldsymbol\theta=
\begin{bmatrix}\Delta\mathbf c\\\Delta\mathbf X\end{bmatrix},
\quad
\Delta\mathbf c=\begin{bmatrix}\Delta\boldsymbol\xi_1\\\vdots\\\Delta\boldsymbol\xi_m\end{bmatrix},
\quad
\Delta\mathbf X=\begin{bmatrix}\Delta\mathbf X_1\\\vdots\\\Delta\mathbf X_n\end{bmatrix}.
$$

The dimensions are $6m$, $3n$, and $6m+3n$, before removing fixed variables.

For observation $(i,j)$ define

$$
A_{ij}=\frac{\partial\mathbf r_{ij}}{\partial\boldsymbol\xi_i}\in\mathbb R^{2\times6},
\qquad
B_{ij}=\frac{\partial\mathbf r_{ij}}{\partial\mathbf X_j}\in\mathbb R^{2\times3}.
$$

Then

$$
\mathbf r_{ij}^{\mathrm{new}}\approx
\mathbf r_{ij}+A_{ij}\Delta\boldsymbol\xi_i+B_{ij}\Delta\mathbf X_j.
$$

**Each observation depends on exactly one camera and one landmark.** All other blocks in its Jacobian row are zero.

For two cameras, where camera 1 sees points 1 and 2 and camera 2 sees points 1, 2, and 3:

$$
J=\left[\begin{array}{cc|ccc}
A_{11}&0&B_{11}&0&0\\
A_{12}&0&0&B_{12}&0\\
0&A_{21}&B_{21}&0&0\\
0&A_{22}&0&B_{22}&0\\
0&A_{23}&0&0&B_{23}
\end{array}\right].
$$

Each displayed block row represents two scalar rows. This is a $10\times21$ matrix. The example illustrates structure, not a sufficiently constrained reconstruction; point 3 has only one observation.

Partition $J=[J_c\;J_p]$:

$$
J^\top J=\begin{bmatrix}U&W\\W^\top&V\end{bmatrix},
\qquad
\mathbf b=-J^\top\mathbf r=
\begin{bmatrix}\mathbf b_c\\\mathbf b_p\end{bmatrix}.
$$

| Block | Definition | Size | Structure for standard reprojection BA |
|---|---|---|---|
| $U$ | $J_c^\top J_c$ | $6m\times6m$ | Block diagonal, $6\times6$ per camera |
| $V$ | $J_p^\top J_p$ | $3n\times3n$ | Block diagonal, $3\times3$ per point |
| $W$ | $J_c^\top J_p$ | $6m\times3n$ | Nonzero camera–point observation blocks |

No residual involves two cameras or two points, explaining the diagonal block structure of $U$ and $V$. The cameras and points are coupled through $W$.

## 4. Assembling blocks without building the full Jacobian

We still compute the mathematical equivalent of $J^\top J$, but use small products and direct accumulation.

For each observation:

$$
\begin{aligned}
U_{ii}&\mathrel{+}=A_{ij}^\top A_{ij}, &V_{jj}&\mathrel{+}=B_{ij}^\top B_{ij},\\
W_{ij}&\mathrel{+}=A_{ij}^\top B_{ij},\\
\mathbf b_{c,i}&\mathrel{+}=-A_{ij}^\top\mathbf r_{ij}, &
\mathbf b_{p,j}&\mathrel{+}=-B_{ij}^\top\mathbf r_{ij}.
\end{aligned}
$$

For example, a point seen by three cameras has

$$
V_{jj}=B_{1j}^\top B_{1j}+B_{2j}^\top B_{2j}+B_{3j}^\top B_{3j}.
$$

Conceptual C++ pseudocode:

```cpp
// All blocks start at zero; W.block(i,j) accesses a sparse block.
for (const auto& obs : observations) {
    const int i = obs.camera_id;
    const int j = obs.point_id;
    computeResidualAndJacobians(obs, r, A, B);

    U[i]          += A.transpose() * A;
    V[j]          += B.transpose() * B;
    W.block(i, j) += A.transpose() * B;
    bc[i]         -= A.transpose() * r;
    bp[j]         -= B.transpose() * r;
}
// Add damping to the camera and point diagonal blocks.
```

If an observation has information matrix $\Omega=\Sigma^{-1}$, use $A^\top\Omega A$, $B^\top\Omega B$, $A^\top\Omega B$, and the corresponding weighted right-hand sides. Equivalently, choose $L$ with $L^\top L=\Omega$ and whiten $r,A,B$ by multiplying each by $L$.

Robust losses are commonly handled with iterative weighting. In a basic IRLS approximation to $\frac12\rho(s)$, where $s=r^\top\Omega r$, scale the whitened residual and Jacobians by $\sqrt{\rho'(s)}$, holding that weight fixed during the linear solve. Exact robustification details vary by solver.

## 5. Schur complement: eliminate point updates

In this section, **$U$ and $V$ include LM damping**. The joint system is

$$
\begin{bmatrix}U&W\\W^\top&V\end{bmatrix}
\begin{bmatrix}\Delta\mathbf c\\\Delta\mathbf X\end{bmatrix}
=\begin{bmatrix}\mathbf b_c\\\mathbf b_p\end{bmatrix}.
$$

Write the two equations:

$$
U\Delta\mathbf c+W\Delta\mathbf X=\mathbf b_c,
$$

$$
W^\top\Delta\mathbf c+V\Delta\mathbf X=\mathbf b_p.
$$

Assuming $V$ is invertible, the second equation gives

$$
\Delta\mathbf X=V^{-1}(\mathbf b_p-W^\top\Delta\mathbf c).
$$

Substitute into the first:

$$
U\Delta\mathbf c+WV^{-1}(\mathbf b_p-W^\top\Delta\mathbf c)=\mathbf b_c.
$$

Collect camera terms:

$$
\boxed{
S\Delta\mathbf c=\mathbf h,
\quad S=U-WV^{-1}W^\top,
\quad\mathbf h=\mathbf b_c-WV^{-1}\mathbf b_p.
}
$$

Solve this $6m\times6m$ camera system, then recover points:

$$
\boxed{\Delta\mathbf X=V^{-1}(\mathbf b_p-W^\top\Delta\mathbf c).}
$$

This gives the same joint linear-system solution in exact arithmetic. Points are temporarily eliminated, not fixed or discarded. They are updated and relinearized in the next nonlinear iteration.

OKVIS2-X performs this elimination in two places: implicitly, by letting Ceres' `DENSE_SCHUR` solver do it in the realtime window, and explicitly by hand, to compress old keyframe observations into 6-DoF relative-pose factors ([§10.7](#107-marginalisation-schur-into-a-6-dof-factor-not-a-prior)).

### Why the point elimination is cheap

$V$ contains independent $3\times3$ blocks. Factor each $V_j$ and use small solves instead of inverting a large matrix.

For a point $j$ observed by a camera set $\mathcal C_j$:

$$
\Delta\mathbf X_j=V_j^{-1}\left(
\mathbf b_{p,j}-\sum_{i\in\mathcal C_j}W_{ij}^\top\Delta\mathbf c_i
\right).
$$

### Constructing the reduced system by point

Initialize $S=U$ and $\mathbf h=\mathbf b_c$. For each point $j$, update every observing camera $i$ and every pair of observing cameras $i,k$:

$$
\mathbf h_i\mathrel{-}=W_{ij}V_j^{-1}\mathbf b_{p,j},
$$

$$
S_{ik}\mathrel{-}=W_{ij}V_j^{-1}W_{kj}^\top.
$$

Implement $V_j^{-1}$ applications with solves. Cameras sharing a point become connected in $S$, even though $U$ originally had no off-diagonal camera blocks. A long feature track can create many camera–camera blocks; the reduced matrix can be substantially less sparse.

An iterative solver can also apply the reduced operator without explicitly forming $S$:

$$
S\mathbf x=U\mathbf x-W\left[V^{-1}(W^\top\mathbf x)\right].
$$

## 6. QR: solve without forming normal equations

QR operates on the Jacobian itself:

$$
\min_{\Delta\boldsymbol\theta}\|J\Delta\boldsymbol\theta+\mathbf r\|^2.
$$

For a tall, full-column-rank Jacobian, a full QR factorization gives

$$
J=Q\begin{bmatrix}R\\0\end{bmatrix},
\qquad Q^\top Q=I,
\qquad Q^\top\mathbf r=\begin{bmatrix}\mathbf d\\\mathbf e\end{bmatrix}.
$$

$R$ is square and upper triangular. Orthogonal transformations preserve Euclidean length:

$$
\|J\Delta\boldsymbol\theta+\mathbf r\|^2
=\|R\Delta\boldsymbol\theta+\mathbf d\|^2+\|\mathbf e\|^2.
$$

The second term is constant with respect to the update, so solve

$$
\boxed{R\Delta\boldsymbol\theta=-\mathbf d}
$$

by back-substitution. Householder or Givens transformations can apply $Q^\top$ without explicitly storing a dense $Q$.

For full column rank,

$$
\kappa_2(J^\top J)=\kappa_2(J)^2.
$$

Forming normal equations can therefore amplify conditioning problems and lose numerical accuracy. QR avoids this squaring in the matrix being factorized, although it does not remove the underlying geometric ambiguity or poor conditioning.

### LM damping as extra rows

The damped problem is

$$
\min_{\Delta\boldsymbol\theta}
\left\|\begin{bmatrix}J\\\sqrt\lambda I\end{bmatrix}
\Delta\boldsymbol\theta+
\begin{bmatrix}\mathbf r\\0\end{bmatrix}\right\|^2.
$$

Apply QR to this augmented Jacobian. Its normal equations are precisely

$$
(J^\top J+\lambda I)\Delta\boldsymbol\theta=-J^\top\mathbf r.
$$

For damping $\lambda D^\top D$, append $\sqrt\lambda D$ instead.

## 7. QR landmark elimination: the square-root counterpart of Schur

Stack the $k_j$ observations of point $j$:

$$
\mathbf r_j+A_j\Delta\mathbf c+B_j\Delta\mathbf X_j,
\qquad B_j\in\mathbb R^{2k_j\times3}.
$$

Assume $B_j$ has column rank three. Apply an orthogonal transformation that triangularizes its point columns:

$$
Q_j^\top\begin{bmatrix}B_j&A_j&\mathbf r_j\end{bmatrix}
=
\begin{bmatrix}
R_j&A_{j,1}&\mathbf r_{j,1}\\
0&A_{j,2}&\mathbf r_{j,2}
\end{bmatrix}.
$$

$R_j$ is $3\times3$. The transformed cost is

$$
\|R_j\Delta\mathbf X_j+A_{j,1}\Delta\mathbf c+\mathbf r_{j,1}\|^2
+
\|A_{j,2}\Delta\mathbf c+\mathbf r_{j,2}\|^2.
$$

For any camera update, choose a point update making the first term zero. Therefore only the bottom rows constrain the camera update after eliminating this point:

$$
\min_{\Delta\mathbf c}\sum_j
\|A_{j,2}\Delta\mathbf c+\mathbf r_{j,2}\|^2.
$$

After solving the camera problem, recover each point via

$$
R_j\Delta\mathbf X_j=-\mathbf r_{j,1}-A_{j,1}\Delta\mathbf c.
$$

### Exact connection to the Schur complement

Let $P_j^\perp$ project onto the orthogonal complement of the columns of $B_j$:

$$
P_j^\perp=I-B_j(B_j^\top B_j)^{-1}B_j^\top.
$$

The reduced cost for this point is

$$
\|P_j^\perp(A_j\Delta\mathbf c+\mathbf r_j)\|^2.
$$

Its normal matrix is

$$
A_j^\top P_j^\perp A_j
=A_j^\top A_j-A_j^\top B_j(B_j^\top B_j)^{-1}B_j^\top A_j.
$$

This is exactly that point's contribution to the Schur-reduced camera matrix. QR represents the reduced problem at the Jacobian level; Schur elimination represents it at the normal-equation level.

For damped elimination, append each point's damping rows before eliminating it:

$$
\widetilde B_j=\begin{bmatrix}B_j\\\sqrt\lambda I_3\end{bmatrix},
\quad
\widetilde A_j=\begin{bmatrix}A_j\\0\end{bmatrix},
\quad
\widetilde{\mathbf r}_j=\begin{bmatrix}\mathbf r_j\\0\end{bmatrix}.
$$

Then append camera damping rows once to the reduced camera problem. Do not repeat camera damping for every point. This is equivalent to Schur elimination of the fully damped joint system.

## 8. Solver choices and practical limits

| Technique | Operates on | Role and tradeoff |
|---|---|---|
| Cholesky | Positive-definite normal matrix | Efficient triangular factorization; inherits normal-equation conditioning |
| Schur elimination | Partitioned normal system | Removes landmarks; still needs a reduced-system solver |
| QR | Jacobian or augmented Jacobian | Avoids forming normal equations; storage and work depend on sparsity and ordering |
| SVD | Jacobian | Reveals singular directions; usually expensive for large BA |
| Iterative methods | Matrix/operator products | Avoid some explicit factorizations; effectiveness depends on conditioning and preconditioning |

These names describe different parts of a solve. For example, a solver can use Schur elimination followed by Cholesky, or use QR to eliminate landmarks and solve the reduced least-squares problem.

Practical qualifications:

- **Gauge freedom:** Generic calibrated monocular SfM has seven global similarity gauge freedoms. Fixing one pose removes the global frame freedom but not scale; a separate scale constraint is needed. Damping regularizes an update but does not establish physical scale.
- **Weak landmarks:** A point seen once, or with negligible parallax, does not have well-constrained depth. Undamped $V_j$ or $B_j$ may be singular or nearly singular. Use appropriate geometry checks, rank-aware handling, or regularization.
- **Additional SLAM factors:** IMU, odometry, and relative-pose factors can create direct off-diagonal state blocks in $U$. Independent landmark blocks can still permit Schur elimination. Priors involving multiple landmarks can break the block-diagonal structure of $V$.
- **Intrinsics:** Shared optimized intrinsics create extra couplings and must be included in the parameter ordering. The simple camera-diagonal $U$ above assumes fixed intrinsics and independent pose variables.
- **Temporary elimination versus marginalization:** Schur elimination during a BA solve is followed by landmark recovery and relinearization. Permanently removing old states and retaining a fixed linearized prior is a separate sliding-window marginalization operation. OKVIS2-X takes a third option: it Schur-eliminates landmarks into *reversible* binary pose-graph factors and freezes old states instead of marginalizing them, so the compression can be undone for loop closure and final BA ([§10.7](#107-marginalisation-schur-into-a-6-dof-factor-not-a-prior)).

## 9. One complete iteration

1. Evaluate reprojection residuals at the current poses and points.
2. Compute local Jacobian blocks; apply measurement whitening and any robust weights.
3. Choose a linear solution route:
   - Accumulate $U,V,W,\mathbf b$, add damping, Schur-eliminate points, and solve for cameras; or
   - Transform Jacobian blocks with QR, eliminate points, and solve the reduced least-squares problem with damping included.
4. Recover point updates by back-substitution.
5. Form a trial state using manifold pose updates and additive point updates.
6. Evaluate the nonlinear cost, accept or reject the trial, and adjust LM damping.
7. Relinearize after accepted updates and continue as needed.

The central structure is always the same: every pixel observation links one camera to one point. Schur and QR exploit that structure at different algebraic levels.

## 10. How OKVIS2-X implements this

Sections 1–9 are the generic derivation. This section maps each piece onto the code in this repository and flags where OKVIS deliberately departs from the textbook version.

### 10.1 Where the code lives

| Concern | Code |
|---|---|
| Graph container, Ceres problem, solver options | [ViGraph.cpp](okvis_ceres/src/ViGraph.cpp), [ViGraph.hpp](okvis_ceres/include/okvis/ViGraph.hpp) |
| Sliding-window surgery (freeze, eliminate, convert) | [ViGraphEstimator.cpp](okvis_ceres/src/ViGraphEstimator.cpp) |
| Two-graph architecture, strategy, loop closure, final BA | [ViSlamBackend.cpp](okvis_ceres/src/ViSlamBackend.cpp) |
| Manifolds ($\boxplus$, $\boxminus$) | [PoseLocalParameterization.cpp](okvis_ceres/src/PoseLocalParameterization.cpp), [HomogeneousPointLocalParameterization.cpp](okvis_ceres/src/HomogeneousPointLocalParameterization.cpp) |
| Landmark elimination into pose-graph factors | [TwoPoseGraphError.cpp](okvis_ceres/src/TwoPoseGraphError.cpp) |
| Rank-aware pseudo-inverse / matrix square roots | [PseudoInverse.hpp](okvis_ceres/include/okvis/PseudoInverse.hpp) |
| Thread that drives all of it | [ThreadedSlam.cpp](okvis_multisensor_processing/src/ThreadedSlam.cpp) |

### 10.2 The variables

Ceres never sees a monolithic $\boldsymbol\theta$. Each estimated quantity is a `ParameterBlockSized<ambient, tangent, ...>` owned by `ViGraph`:

| Variable | Ambient | Tangent | Manifold |
|---|---|---|---|
| Robot pose $T_{WS}$ | 7 | 6 | `PoseManifold` |
| Speed and IMU biases $[\mathbf v_W,\mathbf b_g,\mathbf b_a]$ | 9 | 9 | none (Euclidean) |
| Landmark $\underline{\mathbf h}_W$ | 4 | 3 | `HomogeneousPointManifold` |
| Camera extrinsics $T_{SC_i}$ | 7 | 6 | `PoseManifold` |
| World/GPS alignment $T_{GW}$ | 7 | 6 or 4 | `PoseManifold` / `PoseManifold4d` |

Two departures from §1 and §3 matter:

**Poses live on $\mathbb R^3\times SO(3)$, not $SE(3)$.** The update is $\mathbf r\leftarrow\mathbf r+\delta\mathbf r$ and $\mathbf q\leftarrow\delta\mathbf q(\delta\boldsymbol\alpha)\otimes\mathbf q$ ([PoseLocalParameterization.cpp:29](okvis_ceres/src/PoseLocalParameterization.cpp#L29)), so there is no $SE(3)$ left Jacobian coupling $\delta\boldsymbol\alpha$ into the translation increment. Tangent ordering is $[\delta\mathbf r,\delta\boldsymbol\alpha]$ with $\delta\boldsymbol\alpha$ a *world-frame* rotation error; the rotation still pivots about the sensor origin, so the lever-arm conditioning problem of a left-$SE(3)$ perturbation does not arise.

**Landmarks are homogeneous 4-vectors with a 3-dimensional tangent space**, not the $\mathbf X_j\in\mathbb R^3$ of §1. So the $B_{ij}$ of §3 is $2\times4$ in ambient coordinates and $2\times3$ minimally. The payoff is exactly the "weak landmarks" caveat of §8: a keypoint with negligible parallax stays a well-defined bearing with $w\to0$ instead of producing a near-singular $V_j$ at infinite depth.

### 10.3 The factors

| Factor | Residual dim | Parameter blocks | Robust loss |
|---|---|---|---|
| `ReprojectionError` | 2 | pose(7), landmark(4), extrinsics(7) | `CauchyLoss(1.0)` |
| `ImuError` | 15 | pose$_i$(7), sb$_i$(9), pose$_j$(7), sb$_j$(9) | — |
| `TwoPoseGraphError` | 6 | pose$_i$(7), pose$_j$(7) | — (already applied) |
| `RelativePoseError` | 6 | pose$_i$(7), pose$_j$(7) | — |
| `PoseError` | 6 | pose(7) | — |
| `SpeedAndBiasError` | 9 | sb(9) | — |
| `DepthError` | 1 | pose(7), landmark(4), extrinsics(7) | `TukeyLoss(0.1)` |
| `SubmapIcpError` | 1 | pose$_i$(7), pose$_j$(7) | `TukeyLoss(2.0)` |
| `GpsErrorAsynchronous` | 3 | pose(7), sb(9), $T_{GW}$(7) | `CauchyLoss(3.0)` |

Loss functions are constructed once in the `ViGraph` constructor ([ViGraph.cpp:233](okvis_ceres/src/ViGraph.cpp#L233)) and shared by all residual blocks.

Consequently the clean $U$-block-diagonal picture of §3 does **not** hold here: `ImuError` couples consecutive $(\text{pose},\text{speed/bias})$ pairs into a block-tridiagonal band, `TwoPoseGraphError` and `SubmapIcpError` couple keyframe pairs, and GPS couples a pose to the shared $T_{GW}$. What *does* survive is the property that actually enables §5: only reprojection and depth residuals touch landmarks, and each touches exactly one, so $V$ stays block diagonal with $3\times3$ blocks and landmark elimination remains valid.

### 10.4 Jacobians: analytic, minimal, and lifted

There is no automatic differentiation anywhere in the backend. Every factor derives from `ErrorInterface` and implements

```cpp
bool EvaluateWithMinimalJacobians(double const* const* parameters, double* residuals,
                                  double** jacobians, double** jacobiansMinimal) const;
```

returning both the ambient Jacobian Ceres wants ($2\times7$ for a pose) and the minimal one ($2\times6$). The ambient form is produced from the minimal one by the manifold's lift Jacobian,

$$
J_{\text{ambient}} = J_{\text{minimal}}\, J_{\text{lift}},\qquad
J_{\text{lift}}=\frac{\partial\,(y\boxminus x)}{\partial y}\in\mathbb R^{6\times7},
$$

computed by `PoseManifold::minusJacobian` — see [ReprojectionError.hpp:149](okvis_ceres/include/okvis/ceres/implementation/ReprojectionError.hpp#L149) and roughly fifteen other call sites. Whitening (§4) is done inside each factor: it stores a `squareRootInformation_` matrix $L$ with $L^\top L=\Omega$ and premultiplies residual and Jacobians by it, so Ceres only ever sees whitened quantities. Robust weighting is left to Ceres, with one exception noted in §10.7.

### 10.5 The linear solve

The Ceres problem is built with non-owning semantics and `enable_fast_removal = true`, because the backend adds and removes residual blocks continuously ([ViGraph.cpp:239-249](okvis_ceres/src/ViGraph.cpp#L239-L249)):

```cpp
options_.linear_solver_type = ::ceres::SPARSE_NORMAL_CHOLESKY;
options_.trust_region_strategy_type = ::ceres::DOGLEG;
```

**Dogleg, not Levenberg–Marquardt.** The boxed damped system of §2 is not what OKVIS solves. Dogleg solves a (lightly regularised) Gauss–Newton system once per iteration and then picks the step *along the dogleg path* between the Cauchy point and the GN point, clipped to the trust region — so one factorisation per iteration instead of LM's repeated refactorisation at different $\lambda$. That is the right trade when factorisation dominates and the problem is well-scaled, which it is here after whitening.

**Two different linear solvers for two different sparsity patterns:**

| Graph | Solver | Why |
|---|---|---|
| `realtimeGraph_` | `DENSE_SCHUR` ([ViSlamBackend.cpp:877](okvis_ceres/src/ViSlamBackend.cpp#L877)) | This *is* §5. The window holds ~8 states, so the reduced camera system $S$ is a few hundred rows — dense factorisation of $S$ beats sparse bookkeeping. |
| `fullGraph_` | `SPARSE_NORMAL_CHOLESKY` | After marginalisation most keyframes carry no landmarks at all; they are joined by 6-dimensional pose-graph factors. There is little left to eliminate, so Schur reduction would buy nothing and sparse Cholesky on the whole system is better. |

No `linear_solver_ordering` is ever set. Ceres therefore derives the elimination group itself (maximum independent set over the parameter-block graph), which lands on the landmarks — the same $\Delta\mathbf X$ elimination as §5, chosen automatically.

**Bounded time, not bounded convergence.** `CeresIterationCallback` ([ViGraph.cpp:1892](okvis_ceres/src/ViGraph.cpp#L1892)) aborts the solve once a wall-clock budget is spent, subject to a minimum iteration count, by extrapolating the previous iteration's duration. The realtime solve is a truncated optimisation by design; the "iterate until convergence" of §9 applies only to the background and final solves.

### 10.6 Two graphs, two threads

`ViSlamBackend` maintains the same states in two Ceres problems:

- **`realtimeGraph_`** — the sliding window, solved on every frame inside the frame budget (`realtime_max_iterations`, `realtime_time_limit`).
- **`fullGraph_`** — the entire map, solved on a background thread when loop closures demand it (`optimiseFullGraph`, [ViSlamBackend.cpp:1971](okvis_ceres/src/ViSlamBackend.cpp#L1971)).

Results flow both ways: after each realtime solve, poses, speed/biases and landmarks are copied into `fullGraph_` (only when it is not mid-optimisation), and when the background solve finishes, `synchroniseRealtimeAndFullGraph` ([ViSlamBackend.cpp:1589](okvis_ceres/src/ViSlamBackend.cpp#L1589)) folds the loop-closed map back into the window. The `isLoopClosing_` / `isLoopClosureAvailable_` flags are what keep the two from being written concurrently.

### 10.7 Marginalisation: Schur into a 6-DoF factor, not a prior

This is the largest departure from the standard recipe, and it is the point of the architecture. OKVIS2 does **not** keep a growing marginalisation prior over the window boundary. Instead:

1. **Choose a victim.** `applyStrategy` ([ViSlamBackend.cpp:555](okvis_ceres/src/ViSlamBackend.cpp#L555)) drops non-keyframe IMU frames first (merging their IMU factors), then, while the window exceeds `num_keyframes`, picks the keyframe with the *fewest covisible observations* with the current frame and keyframe.

2. **Pick which pairs to keep.** `convertToPoseGraphMst` ([ViGraphEstimator.cpp:334](okvis_ceres/src/ViGraphEstimator.cpp#L334)) builds a maximum spanning tree over the covisibility graph and converts that keyframe's observations into binary `TwoPoseGraphError` links along the tree edges. An observation reachable through several links has its information progressively halved so it is not double-counted.

3. **Eliminate the landmarks by hand.** `TwoPoseStandardGraphError::compute` ([TwoPoseGraphError.cpp:162](okvis_ceres/src/TwoPoseGraphError.cpp#L162)) is §5 written out explicitly, in the reference frame $S_0$. For each landmark it accumulates the robustified blocks $H_{00}=A^\top A$, $H_{01}=W=A^\top B$, $H_{11}=V=B^\top B$, $\mathbf b_0$, $\mathbf b_1$ over that landmark's observations, then subtracts its Schur contribution:

$$
S \;=\; \sum_j\Big(H_{00}^{(j)} - W_j V_j^{+} W_j^\top\Big),
\qquad
\mathbf h \;=\; \sum_j\Big(\mathbf b_{0}^{(j)} - W_j V_j^{+} \mathbf b_{1}^{(j)}\Big).
$$

   It never forms $V_j^{-1}$: `PseudoInverse::symmSqrt` returns a symmetric square root $M_j$ of $V_j^{+}$ together with its numerical rank, and the update is written as $M(M)^\top$ with $M=W_jM_j$ — the square-root form of §7 applied per landmark. A landmark whose $V_j$ is rank deficient *and* which sits closer than about 3 m is skipped entirely rather than injecting a degenerate direction into the factor. That is §8's "weak landmarks" bullet, implemented.

   Note the robust loss is applied *inside* this accumulation, replicating Ceres' `Corrector` arithmetic on the residual and Jacobians (the IRLS-with-second-order-correction of §4) — because at this point Ceres is no longer in the loop.

4. **Store it in square-root form.** $S$ is generally rank deficient (it constrains at most the 6 relative DoF, sometimes fewer), so `compute` eigendecomposes $S=U D U^\top$, zeroes eigenvalues below tolerance, and stores $J_-=D^{1/2}U^\top$ and $\Delta\mathbf x_-=-UD^{-1/2}\,(UD^{-1/2})^\top\mathbf h$. The resulting 6-dimensional residual is

$$
\mathbf e \;=\; J_-\Big(\Delta\mathbf x_- + \big(T_{S_0S_i} \boxminus T_{S_0S_i}^{\text{lin}}\big)\Big),
$$

   evaluated from the *relative* transform of the two poses, so the factor is invariant to a common rigid motion of both keyframes and contributes no spurious gauge information.

5. **Freeze rather than marginalise.** Old poses and speed/biases are made constant with `SetParameterBlockConstant` ([ViGraphEstimator.cpp:216](okvis_ceres/src/ViGraphEstimator.cpp#L216)); the states remain in the map, they just stop moving in the realtime problem.

6. **The elimination is reversible.** `convertToObservations` / `expandKeyframe` turn a `TwoPoseGraphError` back into its original reprojection errors and landmarks, re-merging observations that had been duplicated across links. This is what makes loop closure and final BA possible, and it is precisely what a conventional marginalisation prior forecloses: a prior locks in its linearisation point forever, whereas these factors can be discarded and the true residuals relinearised. §8's last bullet — temporary elimination versus permanent marginalisation — is resolved here in favour of *recoverable* elimination.

### 10.8 Gauge

§8 counts seven gauge freedoms for calibrated monocular SfM. Visual-inertial estimation has four: global position and yaw. Scale and roll/pitch are observable through gravity and the accelerometer, which is why nothing in this codebase adds a scale constraint. The four are handled by:

- **During initialisation** — a temporary `PoseError` on the newest state with information diag $[10^8,10^8,10^8,0,0,0]$, pinning position only and leaving orientation free ([ViSlamBackend.cpp:822](okvis_ceres/src/ViSlamBackend.cpp#L822)). It is removed as soon as the solve returns.
- **Afterwards** — `freezePosesUntil` fixes everything older than the window, which anchors the frame implicitly.
- **For GNSS** — `T_GW` is estimated until observable, then frozen; the 4-DoF `PoseManifold4d` exists so that alignment can be restricted to translation plus yaw.

### 10.9 Final bundle adjustment

`doFinalBa` ([ViSlamBackend.cpp:2005](okvis_ceres/src/ViSlamBackend.cpp#L2005)) undoes the entire sliding-window compression: every keyframe carrying pose-graph links is expanded back into observations, unobserved landmarks are pruned, all poses and speed/biases are unfrozen from state 1, and `ImuError::redoPropagationAlways` is set so preintegration is redone from raw measurements at the new bias estimates. Then it solves twice — once with priors and fixed extrinsics, once with the speed/bias prior removed and extrinsics either free or softly constrained. This is the only place the full §1–§7 problem is actually assembled.

### 10.10 Config knobs

From [config/euroc/okvis2.yaml](config/euroc/okvis2.yaml):

| Key | Effect |
|---|---|
| `num_keyframes`, `num_imu_frames`, `num_loop_closure_frames` | Window size; drives when §10.7 fires |
| `realtime_max_iterations`, `realtime_min_iterations`, `realtime_time_limit`, `enforce_realtime` | Truncation of the realtime solve |
| `realtime_num_threads`, `full_graph_num_threads` | `Solver::Options::num_threads` for each graph |
| `full_graph_iterations` | Iteration cap for the background solve |
| `do_loop_closures`, `do_final_ba` | Whether §10.6 and §10.9 run at all |

### 10.11 Summary of departures

| §1–9 says | OKVIS2-X does |
|---|---|
| Landmarks $\mathbf X\in\mathbb R^3$ | Homogeneous $\underline{\mathbf h}\in\mathbb P^3$, 3-dimensional tangent space |
| Poses updated with a generic $\boxplus$ | $\mathbb R^3\times SO(3)$ with world-frame rotation error, explicitly not $SE(3)$ |
| LM damping $(J^\top J+\lambda I)$ | Dogleg trust region, one factorisation per iteration |
| Schur complement to reduce cost | Schur in the realtime window (`DENSE_SCHUR`); plain sparse Cholesky on the full graph, where few landmarks remain |
| $U$ block diagonal | Block-tridiagonal from IMU factors, plus pose-graph and GNSS couplings; only $V$ stays block diagonal |
| Iterate to convergence | Realtime solve truncated on a wall-clock budget |
| Marginalisation is permanent | Landmarks Schur-eliminated into reversible 6-DoF two-pose factors; old states frozen, not marginalised |
| Fix one pose to remove gauge freedom | Position-only prior during initialisation, then freeze the tail of the window; scale is observable from the IMU |
