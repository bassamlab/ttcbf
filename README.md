# TTCBF: A Sampled-Time Truncated Taylor Control Barrier Function for High-Order Safety Constraints

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![arXiv: 2601.15196](https://img.shields.io/badge/arXiv-2601.15196-b31b1b.svg)](https://arxiv.org/abs/2601.15196)

This repository contains the reproducible simulation code for the manuscript **“TTCBF: A Sampled-Time Truncated Taylor Control Barrier Function for High-Order Safety Constraints.”** It compares the proposed truncated Taylor control barrier function (TTCBF) and adaptive TTCBF (aTTCBF) with ten baseline methods in a nonlinear static-obstacle avoidance scenario.

## Experiment overview

The simulated vehicle has state `[p_x, p_y, theta, v]` and control `[u_1, u_2]`, where `u_1` is yaw rate and `u_2` is acceleration. The default experiment uses:

- sampling period: `0.05 s`;
- simulation horizon: `14 s`;
- initial state: `(0, -1.5, 0, 1)`;
- goal position: `(10, 0)`;
- obstacle center and radius: `(5, 0)` and `1.0 m`;
- vehicle radius: `0.2 m`;
- yaw-rate bounds: `[-2, 2] rad/s`;
- acceleration bounds: `[-1, 1] m/s^2`.

All method implementations and parameter values used for the manuscript are self-contained in [`main.py`](main.py).

## Installation

The reference environment uses Python 3.11. Create an isolated Conda environment and install the pinned dependencies:

```bash
git clone https://github.com/bassamlab/ttcbf.git
cd ttcbf
conda create -n ttcbf python=3.11 -y
conda activate ttcbf
python -m pip install -r requirements.txt
```

MP4 export is optional and requires [FFmpeg](https://ffmpeg.org/) on the system path:

```bash
# macOS with Homebrew
brew install ffmpeg

# Ubuntu or Debian
sudo apt-get install ffmpeg

# Windows PowerShell with WinGet
winget install --id Gyan.FFmpeg -e
```

Windows users can alternatively download a compiled build from the [FFmpeg Windows download page](https://ffmpeg.org/download.html#build-windows) and add its `bin` directory to `PATH`. Open a new terminal and run `ffmpeg -version` to verify the installation.

The manuscript results were generated on an Apple M2 Pro with 16 GB of RAM. The simulation outputs are deterministic within normal solver tolerances, but reported wall-clock runtimes depend on the hardware and current system load.

## Reproducing the results

### Default comparison

Run all 12 methods with the manuscript's default control bounds:

```bash
python main.py
```

Results are written to `eval_results_accel_min-1/`. Under these bounds, TTCBF, aTTCBF, all recursive-chain baselines, and ET-aTLC reach the goal. ZOH-TLC, rTLC, and ET-TLC terminate because their QPs become infeasible.

### Enlarged braking bound

Reproduce the comparison with the lower acceleration bound set to `-1.5 m/s^2`:

```bash
python main.py --accel-min -1.5
```

The default output directory is `eval_results_accel_min-1.5/`.

### Run selected methods without cached baselines

Use `--methods` with `--no-reuse-cache` to simulate only selected methods:

```bash
python main.py \
  --methods ttcbf,attcbf \
  --no-reuse-cache \
  --out-dir eval_results_ttcbf_only
```

Without `--no-reuse-cache`, compatible cached baselines in the output directory are included in the comparison. Requested methods are always recomputed; missing or stale baseline caches are simulated automatically.

### Export a trajectory video

Add `--save-video` to a single-scenario run:

```bash
python main.py --save-video
```

After simulation and cache loading are complete, the script exports `video_xy_trajectories.mp4`. The video supports fully cached, fully fresh, and mixed result sets, and uses the same trajectory styles and inset view as `fig_xy_trajectories.pdf`. Its legend is placed above the plotting frame. Each colored triangle shows vehicle heading, while the short line extending ahead of it has length proportional to speed normalized by the configured speed bounds. Video export is intentionally unavailable for grid sweeps.

### Override scenario or method parameters

Use repeatable `--set NAME=VALUE` arguments for fields defined by `Scenario`:

```bash
python main.py \
  --methods avcbf \
  --set avcbf_k1=0.5 \
  --set avcbf_k2=0.5
```

The dedicated `--accel-min` option is equivalent to setting `accel_min` and takes precedence when both forms are supplied. Run `python main.py --help` for the complete command-line interface.

### Tight-control adaptive sweep

Reproduce the manuscript's 400 control-bound combinations for the six adaptive methods:

```bash
python main.py --grid-sweep --no-plot-figure
```

This evaluates 20 braking bounds and 20 negative yaw-rate bounds, for 2,400 rollouts in total. Omit `--no-plot-figure` to additionally save a trajectory figure for every control-bound combination. The sweep can take substantially longer than the default comparison.

## Outputs and caching

A standard run produces:

- `summary.csv`: rollout outcomes, safety, control, and timing metrics;
- `simulation_logs.npz`: state and control trajectories;
- `method_cache/*.npz`: reusable per-method simulation results;
- `fig_all_methods_legend.pdf`, `fig_composite_results.pdf`, `fig_cbf_h.pdf`,
  `fig_speed.pdf`, `fig_acceleration_steering_rate.pdf`,
  `fig_taylor_residuals.pdf`, and `fig_xy_trajectories.pdf`: manuscript-ready
  figures.

When `--save-video` is supplied, the run also produces `video_xy_trajectories.mp4`. The video is regenerated from the collected trajectories and is not part of the method cache.

Grid sweeps produce `grid_sweep_rollouts.csv` and `grid_sweep_method_summary.csv`. Generated result directories are intentionally ignored by Git. Delete an output directory or pass `--no-reuse-cache` when a completely fresh run is required. Cache compatibility accounts for shared scenario settings and method-specific parameters.

## Number of tuning parameters

The manuscript counts user-chosen safety-constraint parameters, auxiliary
dynamics parameters, and associated QP weights; goal-oriented CLF parameters
are excluded.

| Method | Count | Main source of tuning parameters |
| --- | ---: | --- |
| TTCBF (our) | **1** | One class-K coefficient |
| aTTCBF (our) | **1** | One adaptive-gain penalty weight |
| DT-HOCBF | 2 | Two discrete-time class-K coefficients |
| aDT-HOCBF | 11 | Adaptive discrete-time gains and auxiliary dynamics |
| CT-HOCBF | 2 | Two class-K coefficients |
| PACBF | 9 | Base gains, penalty dynamics, targets, and QP weights |
| RACBF | 11 | Base gains, relaxation dynamics, targets, and QP weights |
| AVCBF | 9 | Base gains, auxiliary-variable dynamics, and QP weights |
| ZOH-TLC | 1 | Taylor time step |
| rTLC | 1 | Taylor time step |
| ET-TLC | 3 | Time step and event-triggering state bounds |
| ET-aTLC | 4 | Time-scale bounds, look-ahead horizon, and state bounds |

TTCBF and aTTCBF retain one tuning parameter independently of the safety constraint's relative degree; recursive-chain constructions require additional class-K functions as the relative degree increases.

## Reproducibility note

The experiments use a rate-bounded estimate of the Taylor residual. As stated in the manuscript, this estimate is rigorous for smooth inputs but is a practical approximation under zero-order hold. The script therefore also exports an a posteriori comparison between the estimated lower bound and the realized closed-loop residual along the simulated TTCBF trajectories.

## Citation

If this repository supports your work, please cite:

```bibtex
@article{xu2026ttcbf,
  title   = {{TTCBF}: A Sampled-Time Truncated Taylor Control Barrier Function for High-Order Safety Constraints},
  author  = {Xu, Jianye and Alrifaee, Bassam},
  journal = {arXiv preprint arXiv:2601.15196},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
