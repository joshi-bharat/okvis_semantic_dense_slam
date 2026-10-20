# Running and evaluating BRISK vs. SuperPoint

Commands for `okvis_app_synchronous` on EuRoC and TUM-VIE, then evaluation and plots with
[evo](https://github.com/MichaelGrupp/evo). Output folders are created automatically.

## Common variables

```bash
REPO=/home/bharatjoshi/ros_ws/okvis_ws/src/okvis_semantic_dense_slam
APP=/home/bharatjoshi/ros_ws/okvis_ws/build/okvis/okvis_app_synchronous
MODE=vio   # file prefix: "slam" if do_loop_closures: true in the config, "vio" if false
```

Trajectory files written by the app, in TUM format (`timestamp[s] tx ty tz qx qy qz qw`;
`<mode>` is `slam` or `vio`, from `do_loop_closures`):

- `okvis2-<mode>_trajectory_tum.txt` — live estimate
- `okvis2-<mode>-final_trajectory_tum.txt` — after final optimisation (use this for results)

Rerunning into the same folder leaves files from earlier runs with the other mode in place, so
check the timestamps or use a fresh folder per run.

## EuRoC

Configs: `config/euroc/okvis2.yaml` (BRISK) and `config/euroc/okvis2-superpoint.yaml` (SuperPoint).
SuperPoint disables loop closure; set `do_loop_closures: false` in `okvis2.yaml` for a fair
comparison.

### Unpack a sequence (once)

```bash
cd /home/bharatjoshi/Data/vi_slam/euroc/machine_hall/MH_01_easy
unzip MH_01_easy.zip   # creates mav0/
```

### Run

```bash
NAME=V2_03
SEQ=/home/bharatjoshi/Data/vi_slam/euroc/vicon_room2/V2_03_difficult/V2_03_difficult/mav0
OUT=/home/bharatjoshi/Data/vi_slam/euroc/results

$APP $REPO/config/euroc/okvis2.yaml            $SEQ $OUT/${NAME}_brisk
$APP $REPO/config/euroc/okvis2-superpoint.yaml $SEQ $OUT/${NAME}_superpoint
```

### Error (ATE)

`evo_ape euroc` reads the EuRoC ground-truth CSV as reference and the estimate as TUM, so the
app's output is used as is. EuRoC ground truth is the IMU pose, the same frame OKVIS writes, so
no lever arm is needed.

```bash
GT=$SEQ/state_groundtruth_estimate0/data.csv
for r in brisk superpoint; do
  echo "== $r"
  evo_ape euroc $GT $OUT/${NAME}_$r/okvis2-${MODE}-final_trajectory_tum.txt -a -va
done
```

Add `--plot --plot_mode xy` to see the trajectory coloured by error.

### Plot

`evo_traj` needs every file in the same format, so convert the ground truth to TUM once:

```bash
cd $OUT && evo_traj euroc $GT --save_as_tum && mv data.tum ${NAME}_gt.tum
```

```bash
# one trajectory vs. ground truth, top view
evo_traj tum $OUT/${NAME}_brisk/okvis2-${MODE}-final_trajectory_tum.txt \
  --ref $OUT/${NAME}_gt.tum -a --plot --plot_mode xy

# BRISK and SuperPoint together, saved to a file
evo_traj tum $OUT/${NAME}_brisk/okvis2-${MODE}-final_trajectory_tum.txt \
             $OUT/${NAME}_superpoint/okvis2-${MODE}-final_trajectory_tum.txt \
  --ref $OUT/${NAME}_gt.tum -a --plot --plot_mode xy --save_plot $OUT/${NAME}_trajectories.pdf
```

`--plot_mode`: `xy` top view, `xz` / `yz` side views, omit for 3D.

## TUM-VIE

Configs: `config/tum_vie/okvis2.yaml` (BRISK) and `config/tum_vie/okvis2-superpoint.yaml`
(SuperPoint). The dataset needs the `-tumvie` flag.

### Run

```bash
NAME=loop-floor3
SEQ=/home/bharatjoshi/Data/vi_slam/tum_vie/loop-floor3-vi_gt_data
OUT=/home/bharatjoshi/Data/vi_slam/tum_vie/results

$APP $REPO/config/tum_vie/okvis2.yaml            $SEQ $OUT/${NAME}_brisk      -tumvie
$APP $REPO/config/tum_vie/okvis2-superpoint.yaml $SEQ $OUT/${NAME}_superpoint -tumvie
```

### Error and plots

Mocap only covers part of the sequence and tracks a marker, not the IMU, so use the repo script
rather than evo: it estimates and removes the IMU-to-marker offset and reports drift between
mocap segments.

```bash
for r in brisk superpoint; do
  python3 $REPO/tools/eval_tum_vie.py $SEQ $OUT/${NAME}_$r
done
```

Plots are written to `$OUT/${NAME}_<run>/eval/`, aligned on the first mocap segment so the
trajectories start together. Use `--no-plot` to skip them.

## Timing

The app prints a timing table at the end. SuperPoint-specific rows:

- `1.1 SuperPoint inference`
- `1.2 SuperPoint NMS and descriptors`
- `2.00 SuperPoint descriptor GEMM`
