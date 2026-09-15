# Upgraded Version of PPGA

## Notable Changes

- Moved Python source under a single folder, `ppga/`
- Switched to `requirements.txt` instead of `environment.yml`
- Upgraded pyribs to 0.7.1
- Upgraded to jax 0.4.28 and Brax 0.9.4, particularly by adopting code in the
  DCG-MAP-Elites repo:
  https://github.com/adaptive-intelligent-robotics/DCG-MAP-Elites
  and the qd_position repo:
  https://github.com/SumeetBatra/qd_position/tree/main
  - jax is only used in Brax in this repo, so upgrading Brax allowed upgrading
    jax.
- Upgraded to torch 2.3.1 (torch >= 2.0 and jax >= 0.4.26 work well together due
  to CUDA versions).
- Upgraded to evotorch 0.5.1
- Removed attrdict since it is outdated and replaced it with Box
  https://github.com/cdgriffith/Box

## Installation

```bash
conda create --prefix ./env python=3.10
conda activate ./env
# There are options for installing JAX for CUDA and for CPU in requirements.txt;
# it is set to CPU by default.
pip install -r requirements.txt

```

### Isaac Lab backend

Isaac Lab uses a separate environment because its Python, PyTorch, and CUDA
requirements conflict with the Brax/JAX environment above. This integration
targets Isaac Lab 2.3.2, Isaac Sim 5.1, and Python 3.11.

Isaac Sim must run on native Windows or native Linux with a supported NVIDIA
GPU and driver; its Python package does not support WSL2. On Windows, enable
long paths or choose a short Conda environment path if package installation
hits the Windows path-length limit.

From the repository root, create the environment and install its dependencies:

```bash
conda create -n ppga-isaaclab python=3.11 pip
conda activate ppga-isaaclab

# flatdict 4.0.1 is incompatible with the newest isolated setuptools build.
python -m pip install "pip<26" "setuptools<81"
python -m pip install --no-build-isolation flatdict==4.0.1
python -m pip install "isaaclab[isaacsim,all]==2.3.2" --extra-index-url https://pypi.nvidia.com

# Use this CUDA 12.8 build for Blackwell GPUs (such as the RTX 50 series).
# Other GPUs may use the PyTorch build selected by Isaac Lab instead.
python -m pip install --force-reinstall torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-isaaclab.txt
```

Accept the Isaac Sim EULA in each shell that launches training:

```bash
# Linux (bash)
export OMNI_KIT_ACCEPT_EULA=YES
```

```powershell
# Windows (PowerShell)
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
```

Run a small PPO smoke test before a full QD experiment:

```bash
python -m ppga.RL.train_ppo --env_name=humanoid --env_type=isaac --env_batch_size=64 --rollout_length=32 --total_timesteps=65536 --num_minibatches=4 --update_epochs=5 --num_dims=2 --value_bootstrap=True
```

Run the full local PPGA preset from a Bash-compatible shell, or submit its
Slurm preset on a cluster:

```bash
bash runners/local/train_ppga_humanoid.sh
sbatch runners/slurm/train_ppga_humanoid_slurm.sh
```

The presets use thousands of parallel environments and are intended for GPUs
with ample VRAM. For local validation, copy the module command from the runner
and reduce `--env_batch_size`, `--popsize`, and `--total_iterations`. Cluster
users must also adapt the `#SBATCH` settings, modules, environment name, and any
container path in the Slurm script to their site.

### MJLab manipulation backend

MJLab is optional and uses a separate environment because its MuJoCo Warp and
PyTorch dependencies evolve independently of Isaac Lab. The pinned environment
is intended for Linux x86-64 with an NVIDIA GPU; WSL2 can be used for local
MJLab runs when GPU passthrough is configured. The first integrated task is the
state-based YAM cube-lift task.

From the repository root:

```bash
conda create -n ppga-mjlab python=3.11 pip
conda activate ppga-mjlab
python -m pip install -r requirements-mjlab.txt
```

Run a small PPO smoke test:

```bash
python -m ppga.RL.train_ppo --env_name=lift_cube --env_type=mjlab --env_batch_size=256 --rollout_length=24 --total_timesteps=262144 --num_minibatches=4 --update_epochs=5 --learning_rate=0.0001 --entropy_coef=0.005 --target_kl=0.01 --adaptive_kl=True --norm_adv_per_minibatch=False --mixed_precision=False --normalize_obs=True --action_transform=none --action_std_parameterization=direct --initial_action_std=0.5 --actor_hidden_dims 512 256 128 --num_dims=2 --value_bootstrap=True --mjlab_fixed_goal=False --mjlab_disable_curriculum=True --mjlab_command_resampling_time=40 --mjlab_terminate_on_success=True --mjlab_success_bonus=50 --mjlab_success_max_object_speed=0.15 --mjlab_descriptor_mode=grip_orientation_elbow_extension
```

To verify the installed simulator and task independently of PPGA's PPO, run
MJLab's native RSL-RL baseline with its original dynamic commands and
curriculum:

```bash
MAX_ITERATIONS=1000 ENV_BATCH_SIZE=768 bash runners/ppo/train_mjlab_reference_lift_cube.sh
```

Two additional controls keep commands fixed within each episode. The
episodic control randomizes the cube and goal on reset; the fixed-goal control
always uses `[0.4, 0.0, 0.3]` and is intentionally retained as a harder
diagnostic:

```bash
MAX_ITERATIONS=700 ENV_BATCH_SIZE=768 bash runners/ppo/train_mjlab_reference_episodic_lift_cube.sh
MAX_ITERATIONS=1000 ENV_BATCH_SIZE=768 bash runners/ppo/train_mjlab_reference_stationary_lift_cube.sh
```

The custom episodic PPO runner is retained as a diagnostic baseline. Require
`evaluation.json` to report nonzero `episode_success_rate` and meaningful
`max_object_height` before using one of its checkpoints to seed QD. In current
validation, the native RSL-RL trainer below was the more reliable bootstrap
path. The custom runner writes `final_model.pt`, `evaluation.json`, and `cfg.json` under
`experiments/ppo_mjlab_episodic_lift_cube/<seed>/`. It uses MJLab's unbounded
action convention, direct standard-deviation parameterization, observation
normalization, full-precision inference, shuffled minibatches, rollout-wide
advantage normalization, and adaptive KL scheduling.

```bash
bash runners/ppo/train_ppo_mjlab_lift_cube.sh
sbatch runners/slurm/train_ppo_mjlab_lift_cube_slurm.sh
```

Resumable checkpoints are written every 100 PPO updates and the newest three
are retained. Resume an interrupted run by repeating the same configuration
with `--resume_checkpoint=<checkpoint path>`; `TOTAL_TIMESTEPS` remains the
overall target rather than an additional step count. Console diagnostics show
the individual reward terms, maximum cube height, minimum goal error, success,
policy log-standard-deviation, entropy, and KL.

The native trainer is the preferred bootstrap path. Convert a successful
RSL-RL checkpoint; the converter copies its MLP, per-joint standard deviations,
and observation-normalization statistics:

See [MJLab PPO comparison](docs/mjlab_ppo_comparison.md) for concise evidence
and the remaining difference between PPGA PPO and RSL-RL PPO.

```bash
python -m ppga.RL.convert_mjlab_rsl_checkpoint /path/to/model_999.pt /path/to/ppga_actor_model_999.pt
INITIAL_ACTOR_CHECKPOINT=/path/to/ppga_actor_model_999.pt bash runners/local/train_ppga_mjlab_lift_cube.sh
sbatch --export=ALL,INITIAL_ACTOR_CHECKPOINT=/path/to/ppga_actor_model_999.pt runners/slurm/train_ppga_mjlab_lift_cube_slurm.sh
```

The PPGA runner also works without `INITIAL_ACTOR_CHECKPOINT` for a random
initial mean. Both runners accept environment overrides such as `SEED=43`,
`ENV_BATCH_SIZE=384`, `TOTAL_TIMESTEPS=1000000`, `POPSIZE=32`, `SIGMA0=0.05`,
and `TOTAL_ITERATIONS=10`. For a seed sweep, submit the same runner with
different `SEED` values. Adapt the Slurm resource, module, and Conda settings
to the target cluster.

The recommended MJLab preset randomizes the cube and goal once per episode,
disables the reward curriculum, and makes the command-resampling interval
longer than the episode. This is a stationary task distribution without
mid-episode cube teleports. A single fixed 30 cm target was empirically prone
to a reach-only local optimum. A lift now succeeds only when the cube is within
the task's 5 cm goal tolerance and moving no faster than 0.15 m/s. Success
terminates the episode and adds a one-time reward of 50, eliminating the old
incentive to accumulate reward by holding the cube motionless at the goal.

The default `grip_orientation_elbow_extension` descriptors in `[0, 1]`
describe successful grasp posture without rewarding slow movement. Grip
orientation is the wrist approach-axis tilt: zero points down and one points
up. Elbow extension is zero when folded and one near a straight arm. Both use
the closest pre-transport pose to the cube. The 75%
success gate prevents failed reaches from filling the archive.

`grip_orientation_arm_length` remains available, but calibration showed that
base-to-grasp-site distance mostly follows cube position and occupied one bin.
The pretrained seed policies also cluster tightly in elbow extension; unlike
arm length, however, elbow angle is directly controllable without changing the
cube goal. A new PPGA run is needed to determine whether it expands this axis.

Legacy `approach_transport` descriptors target visibly different manipulation
paths. The first is the end effector's signed deviation
from its episode-initial straight path to the cube, weighted toward samples
near the cube. The second retains the cube's largest signed perpendicular
deviation from its direct start-to-goal transport path after the cube has moved
3 cm. This peak statistic does not disappear when the cube later returns to the
direct path. Approach deviations of 5 cm and transport deviations of 15 cm map
to their respective descriptor endpoints. Terminal archive coordinates use
these complete episode statistics rather than averages of intermediate
estimates. `motion_effort`, `height_approach`, and `progress` also remain
available for reproducing older archives.

Task metrics are logged during PPO and stored in elite metadata. PPGA's
`summary.csv` includes archive mean/max success rate, maximum object height,
and mean trajectory length. The default PPGA preset uses a 12-by-12 grid and
admits only policies with at least 75% episode success, in addition to
rejecting objectives below zero. Dense task and descriptor
rewards remain available for PPO/DQD gradients, so this gate changes archive
eligibility rather than making gradient learning sparse. Archive-only
checkpoints are saved every 10 iterations and heatmaps every 10 iterations to
avoid multi-gigabyte scheduler checkpoints. `SIGTERM` finishes the active QD
iteration and saves a final archive and summary row, which also makes
Slurm time-limit termination recoverable. Set `--save_scheduler=True` only when
full optimizer/emitter restart state is worth the storage cost.

Branch-policy evaluation uses common reset scenarios by default: every
contiguous policy block receives the same ordered cube and goal samples for a
given evaluation. The scenario seed advances between evaluations so search
does not optimize a single fixed batch. This removes reset-placement luck from
comparisons inside an XNES population while retaining MJLab's observation and
startup domain randomization.

The MJLab runner initializes the XNES gradient-coefficient center at zero,
uses `SIGMA0=0.05`, and conservatively moves the learned mean for one PPO
iteration at `LEARNING_RATE=0.0001`. This makes the search local around the
successful bootstrap policy instead of immediately erasing its manipulation
skill. The legacy random-center behavior remains the default for other runners
and can be selected explicitly with `--xnes_center_init=random`. Override the
MJLab defaults with `CALC_GRADIENT_ITERS`, `MOVE_MEAN_ITERS`, and
`LEARNING_RATE` when conducting ablations.

For a higher-throughput local or cluster run, first smoke-test
`ENV_BATCH_SIZE=1152`; if memory and simulation stability remain healthy, use
`ENV_BATCH_SIZE=1536`. With the default population of 32, these settings give
36 or 48 independent evaluation episodes per policy and make archive admission
more reliable. Do not reuse a seed directory from an older descriptor mode.

Reevaluate every policy in a saved archive under the same reset-only episodic
task distribution with:

```bash
python -m ppga.algorithm.evaluate_mjlab_archive --archive experiments/ppga_mjlab_episodic_lift_cube/42/checkpoints/cp_00000025/archive_df_00000025.pkl --config experiments/ppga_mjlab_episodic_lift_cube/42/cfg.json --output experiments/ppga_mjlab_episodic_lift_cube/42/archive_reevaluation.csv
```

The output retains stored objective/descriptors alongside fresh objective,
descriptors, success rate, maximum object height, and minimum goal-position
error, plus mean trajectory length. The adapter disables MJLab auto-reset,
records the true terminal state, and partially resets only completed
environments.

Render representative successful archive policies to MP4 after a checkpoint:

```bash
ARCHIVE=experiments/ppga_mjlab_episodic_lift_cube/42/checkpoints/cp_00000025/archive_df_00000025.pkl \
CONFIG=experiments/ppga_mjlab_episodic_lift_cube/42/cfg.json \
OUTPUT_DIR=experiments/ppga_mjlab_episodic_lift_cube/42/videos/cp_00000025 \
bash runners/local/render_mjlab_archive.sh
```

The renderer records the highest-objective successful elite first, then uses
farthest-point selection in descriptor space to choose four behaviorally
different successful elites. It saves deterministic, full-episode videos and
a `manifest.csv` containing stored and fresh rollout metrics. Override
`NUM_POLICIES`, `EPISODES_PER_POLICY`, `MIN_SUCCESS_RATE`, `VIDEO_WIDTH`,
`VIDEO_HEIGHT`, or `FRAME_STRIDE` through environment variables. Rendering is
offscreen and does not require a visible viewer; on a headless Linux node, set
`MUJOCO_GL=egl` if the node's graphics stack requires an explicit backend.

### Transition and reset semantics

Isaac and MJLab now follow the same QD transition contract. `info` contains
`measures`, dt-scaled `measure_rewards`, `final_observation`,
`final_observation_mask`, and `final_measures`. PPO uses the pre-reset final
observation for time-limit bootstrapping; it raises an error instead of using a
post-reset state when terminal data is unavailable.

Each DQD gradient, branch-evaluation, and mean-movement phase still begins with
a reset because those phases change the policy-to-environment assignment.
`PPO.train(..., reset_env=False)` is available only for consecutive calls with
the identical policy grouping; assigning `ppo.agents` invalidates continuity
automatically.

The corrected terminal descriptors change archive semantics. Do not resume an
Isaac archive produced before this change in a new run; reevaluate old policies
under the new adapter if a comparison is required.

## Running PPO for Diverse Generators Work

1. Train the agent with `bash runners/ppo/train_ppo_halfcheetah.sh`
1. Collect agent trajectory data with `bash runners/ppo/collect_ppo_halfcheetah.sh`

Outputs should now be in the `checkpoints/` folder as
`checkpoints/halfcheetah_brax_model_0_checkpoint` (the model checkpoint)
`output_file=checkpoints/halfcheetah_trajectories.h5` (the dataset of
trajectories)

---

# Proximal Policy Gradient Arborescence

The official repo of PPGA! Implemented in PyTorch and run with
[Brax](https://github.com/google/brax), a GPU-Accelerated high-throughput
simulator for rigid bodies. This project also contains a modified version of
[pyribs](https://github.com/icaros-usc/pyribs), a QD library, and implements a
modified multi-objective, vectorized version of Proximal Policy Optimization
(PPO) based off of [cleanrl](https://github.com/vwxyzjn/cleanrl).

## Requirements

We use Anaconda to manage dependencies.

```bash
conda env create -f environment.yml
conda activate ppga
```

Then install this project's custom version of pyribs.

```bash
cd pyribs && pip install -e. && cd ..
```

### CUDA

This project has been tested on Ubuntu 20.04 with an NVIDIA RTX 3090 GPU. In
order to enable GPU-Acceleration, your machine must support CUDA 11.X with
minimum driver version 450.80.02 (Linux x86_64). See
[here](https://docs.nvidia.com/deploy/cuda-compatibility/) for more details on
cuda compatibility.

The environment.yml file intentionally contains no CUDA dependencies since this
is a machine dependent property, and so jax-cuda and related CUDA packages must
be installed by the user. We recommend installing one of the following
jaxlib-cuda packages:

```bash
# for CUDA 11 and cuDNN 8.2 or newer
wget https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.25+cuda11.cudnn82-cp39-cp39-manylinux2014_x86_64.whl
pip install jaxlib-0.3.25+cuda11.cudnn82-cp39-cp39-manylinux2014_x86_64.whl

# OR

# for CUDA 11 and cuDNN 8.0.5 or newer
wget https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.25+cuda11.cudnn805-cp39-cp39-manylinux2014_x86_64.whl
pip install jaxlib-0.3.25+cuda11.cudnn805-cp39-cp39-manylinux2014_x86_64.whl
```

If you run into issues getting cuda-accelerated jax to work, please see the
[jax github](https://github.com/google/jax) for more details.

We recommend using conda to install cuDNN and cudatoolkit

```bash
conda install -c anaconda cudnn
conda install -c anaconda cudatoolkit
```

### Common gotchas

Most issues arise from having the wrong version of Jax, Flax, Brax etc.
installed. If you followed the steps above and are still running into issues,
please make sure the following packages are of the right version:

```bash
jax==0.3.25
jaxlib==0.3.25+cuda11.cudnn82 # or whatever your cuDNN version is
jaxopt==0.5.5
flax==0.6.1
brax==0.1.0
chex==0.1.5
gym==0.23.1
```

### Preflight Checklist

Depending on your machine specs, you may encounter out of memory errors due to
how Jax VRAM preallocation works. If this is you, you will need to disable
memory preallocation.

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

With CUDA enabled, you will also need to add the cublas library to your
LD_LIBRARY_PATH like so:

```bash
export LD_LIBRARY_PATH=<PATH_TO_ANACONDA>/envs/ppga/lib/python3.9/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH
```

For example, if you use miniconda, this would be
`/home/{username}/miniconda3/...`

## Running Experiments

Run all commands from the repository root. The backend-specific installation,
smoke-test, local PPGA, and Slurm commands are documented above under
**Isaac Lab backend** and **MJLab manipulation backend**. Brax paper presets
remain under `runners/local/` and `runners/slurm/`.

For the complete PPO and QD command-line options:

```bash
python -m ppga.RL.train_ppo --help
python -m ppga.algorithm.train_ppga --help
python -m ppga.algorithm.train_ppga_isaac --help
```

For example, a Brax ant preset can be run locally or through Slurm with:

```bash
bash runners/local/train_ppga_ant.sh
sbatch runners/slurm/train_ppga_ant_slurm.sh
```

## Evaluating an Archive

See the jupyter notebook `algorithm/enjoy_ppga.ipynb` for instructions and
examples on how to visualize results!

## Pretrained Archives

Trained archives reported in the paper and scheduler checkpoints are hosted on
Google Drive and can be downloaded from
[here](https://drive.google.com/drive/folders/1dPV5mJNaalqHdMH87KNvGAqnuHi7ozdw?usp=sharing).

## Results

| **Environment** | **QD-Score**       | **Coverage** | **Best Reward** | **Experiment Command**                      |
| --------------- | ------------------ | ------------ | --------------- | ------------------------------------------- |
| Humanoid        | $7.01 \times 10^6$ | 70.0%        | 9755            | `./runners/local/train_ppga_humanoid.sh`    |
| Walker2D        | $5.82 \times 10^6$ | 67.8%        | 4796            | `./runners/local/train_ppga_walker2d.sh`    |
| HalfCheetah     | $2.94 \times 10^7$ | 98.4%        | 9335            | `./runners/local/train_ppga_halfcheetah.sh` |
| Ant             | $2.26 \times 10^7$ | 53.1%        | 7854            | `./runners/local/train_ppga_ant.sh`         |
