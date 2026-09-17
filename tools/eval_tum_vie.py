#!/usr/bin/env python3
"""Evaluate OKVIS2 trajectories against TUM-VIE motion-capture ground truth.

Usage:
    python3 tools/eval_tum_vie.py <sequence-folder> <results-folder> [--out <folder>]

<sequence-folder> must contain mocap_data.txt (t[us] px py pz qx qy qz qw).
Every okvis2-*trajectory.csv in <results-folder> is evaluated.

Mocap only sees the rig inside the tracking volume, so ground truth is split into segments.
Two alignments are reported:
  * full   -- SE(3) alignment on all ground-truth-covered poses (standard ATE)
  * start  -- SE(3) alignment on the first segment only; error on the last segment then measures
              drift accumulated over the part of the sequence mocap could not see
The mocap body (marker) is not the IMU. Its position offset (lever arm) and orientation offset
relative to the IMU are estimated from the data and removed; ATE without that correction is
reported too.
"""

import argparse
import glob
import os

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

MAX_GT_GAP = 0.05  # [s] only interpolate ground truth across gaps shorter than this


def load_okvis(path):
    a = np.genfromtxt(path, delimiter=",", skip_header=1)[:, :8]
    return a[:, 0] * 1e-9, a[:, 1:4], Rotation.from_quat(a[:, 4:8])


def load_mocap(path):
    a = np.loadtxt(path)
    return a[:, 0] * 1e-6, a[:, 1:4], Rotation.from_quat(a[:, 4:8])


def associate(t_est, t_gt, p_gt, R_gt):
    """Interpolate ground truth at estimate timestamps inside covered intervals."""
    idx = np.searchsorted(t_gt, t_est)
    ok = (idx > 0) & (idx < len(t_gt))
    i1 = np.clip(idx, 1, len(t_gt) - 1)
    i0 = i1 - 1
    ok &= (t_gt[i1] - t_gt[i0]) < MAX_GT_GAP
    w = ((t_est - t_gt[i0]) / (t_gt[i1] - t_gt[i0]))[:, None]
    p = (1 - w) * p_gt[i0] + w * p_gt[i1]
    slerp = Slerp(t_gt, R_gt)
    R = slerp(np.clip(t_est, t_gt[0], t_gt[-1]))
    return ok, p, R


def umeyama_se3(src, dst):
    """R, t minimising sum |dst - (R src + t)|^2."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    return R, mu_d - R @ mu_s


def align_with_lever_arm(p_est, R_est, p_gt, iters=20):
    """Jointly estimate world alignment (R, t) and body lever arm b: p_gt ~ R (p_est + R_est b) + t."""
    b = np.zeros(3)
    for _ in range(iters):
        R, t = umeyama_se3(p_est + R_est.apply(b), p_gt)
        # linear least squares for b with R, t fixed: R R_est_i b = p_gt_i - R p_est_i - t
        A = np.einsum("ij,njk->nik", R, R_est.as_matrix()).reshape(-1, 3)
        y = (p_gt - p_est @ R.T - t).reshape(-1)
        b = np.linalg.lstsq(A, y, rcond=None)[0]
    R, t = umeyama_se3(p_est + R_est.apply(b), p_gt)
    return R, t, b


def rotation_errors_deg(R_align, R_est, R_gt):
    """Orientation error after removing the constant marker-to-IMU rotation offset."""
    R_world_est = Rotation.from_matrix(R_align) * R_est
    delta = R_gt.inv() * R_world_est          # = R_marker_imu, if estimate were perfect
    offset = delta.mean()                     # chordal L2 mean
    err = (delta * offset.inv()).magnitude()
    return np.degrees(err), offset


def stats(e):
    return dict(rmse=np.sqrt(np.mean(e ** 2)), mean=e.mean(), median=np.median(e), max=e.max())


def fmt(s, unit, scale=1.0):
    return "  ".join(f"{k} {v * scale:7.3f}" for k, v in s.items()) + f" {unit}"


def segments(t, gap=1.0):
    breaks = np.where(np.diff(t) > gap)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(t) - 1]
    return list(zip(starts, ends))


def evaluate(name, t_est, p_est, R_est, t_gt, p_gt_all, R_gt_all, out_dir, plot):
    ok, p_gt, R_gt = associate(t_est, t_gt, p_gt_all, R_gt_all)
    t, pe, Re, pg, Rg = t_est[ok], p_est[ok], R_est[ok], p_gt[ok], R_gt[ok]
    segs = segments(t)
    print(f"\n== {name}")
    print(f"   {ok.sum()} of {len(t_est)} poses covered by mocap, "
          f"{len(segs)} segment(s): "
          + ", ".join(f"[{t[a]:.1f}, {t[b]:.1f}] s" for a, b in segs))
    length = np.sum(np.linalg.norm(np.diff(p_est, axis=0), axis=1))
    print(f"   estimated path length {length:.2f} m")

    R0, t0 = umeyama_se3(pe, pg)
    e_raw = np.linalg.norm(pg - (pe @ R0.T + t0), axis=1)
    print(f"   ATE full, no lever arm   : {fmt(stats(e_raw), 'cm', 100)}")

    R1, t1, b = align_with_lever_arm(pe, Re, pg)
    e_full = np.linalg.norm(pg - ((pe + Re.apply(b)) @ R1.T + t1), axis=1)
    rot_err, offset = rotation_errors_deg(R1, Re, Rg)
    print(f"   ATE full                 : {fmt(stats(e_full), 'cm', 100)}")
    print(f"   rotation error full      : {fmt(stats(rot_err), 'deg')}")
    print(f"   estimated lever arm IMU->marker [m]: {np.round(b, 4)}  (|b| = {np.linalg.norm(b)*100:.1f} cm)")
    print(f"   estimated rotation offset: {np.degrees(offset.magnitude()):.2f} deg")

    e_drift = None
    if len(segs) >= 2:
        a0, b0 = segs[0]
        sl = slice(a0, b0 + 1)
        Rs, ts = umeyama_se3(pe[sl] + Re[sl].apply(b), pg[sl])
        aN, bN = segs[-1]
        sN = slice(aN, bN + 1)
        e_drift = np.linalg.norm(pg - ((pe + Re.apply(b)) @ Rs.T + ts), axis=1)
        print(f"   start-aligned, first seg : {fmt(stats(e_drift[sl]), 'cm', 100)}")
        print(f"   start-aligned, last seg  : {fmt(stats(e_drift[sN]), 'cm', 100)}")
        # drift on re-entering the mocap volume, before a loop closure can correct it
        reentry = (t >= t[aN]) & (t < t[aN] + 1.0)
        unseen = t_est <= t[aN]
        dist = np.sum(np.linalg.norm(np.diff(p_est[unseen], axis=0), axis=1))
        e_re = e_drift[reentry].mean()
        print(f"   drift on re-entry (1 s)  : {100 * e_re:.2f} cm after {dist:.1f} m"
              f"  -> {100 * e_re / dist:.3f} % of distance travelled")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        p_all = (p_est + R_est.apply(b)) @ R1.T + t1
        gt_ok = np.r_[True, np.diff(t_gt) < MAX_GT_GAP]
        pg_plot = p_gt_all.copy()
        pg_plot[~gt_ok] = np.nan
        ax[0].plot(pg_plot[:, 0], pg_plot[:, 1], "k-", lw=2, label="mocap")
        ax[0].plot(p_all[:, 0], p_all[:, 1], "-", color="tab:blue", lw=1, label="OKVIS2 (aligned)")
        ax[0].plot(p_all[0, 0], p_all[0, 1], "go", label="start")
        ax[0].plot(p_all[-1, 0], p_all[-1, 1], "rs", label="end")
        ax[0].set_aspect("equal")
        ax[0].set_xlabel("x [m]")
        ax[0].set_ylabel("y [m]")
        ax[0].set_title(f"{name}: top view")
        ax[0].legend()
        ax[1].plot(t, 100 * e_full, ".", ms=2, label="full alignment")
        if e_drift is not None:
            ax[1].plot(t, 100 * e_drift, ".", ms=2, label="start-segment alignment")
        ax[1].set_xlabel("time [s]")
        ax[1].set_ylabel("position error [cm]")
        ax[1].set_title("error where mocap is available")
        ax[1].legend()
        fig.tight_layout()
        png = os.path.join(out_dir, name + ".png")
        fig.savefig(png, dpi=110)
        plt.close(fig)
        print(f"   plot: {png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence")
    ap.add_argument("results")
    ap.add_argument("--out", default=None, help="plot folder (default: <results>/eval)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    t_gt, p_gt, R_gt = load_mocap(os.path.join(args.sequence, "mocap_data.txt"))
    out_dir = args.out or os.path.join(args.results, "eval")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.results, "okvis2-*trajectory.csv")))
    if not files:
        raise SystemExit(f"no okvis2-*trajectory.csv in {args.results}")
    for f in files:
        name = os.path.basename(f)[:-4]
        evaluate(name, *load_okvis(f), t_gt, p_gt, R_gt, out_dir, not args.no_plot)


if __name__ == "__main__":
    main()
