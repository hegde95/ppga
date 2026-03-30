"""
eval_policies.py

Pairwise policy evaluation script.  Loads elite checkpoints from a directory,
samples random pairs, rolls out each policy in a Brax vec-env, then scores
each trajectory on a set of measurable functions.  Results are reported
per-function (no cross-function collapse).

Usage:
    python eval_policies.py \
        --archive_path data/archive_df.pkl \
        --env_name humanoid \
        --env_batch_size 32 \
        --rollout_length 1000 \
        --num_evals 10 \
        --seed 0

Notes on env interface (confirmed via inspect_env.py):
    - env.step() returns a 4-tuple: (obs, reward, terminated, info)
      The info dict is the 4th element; there is no separate truncated tensor.
    - pipeline_state is NOT exposed in info (first_pipeline_state is None).
      All measurable functions therefore use pre-computed scalar fields from info.

Available info fields (confirmed):
    x_velocity, y_velocity          — root body velocities
    x_position, y_position          — root body positions
    distance_from_origin            — scalar distance
    forward_reward, reward_linvel   — velocity-based reward components
    reward_alive                    — alive bonus (5.0 per step)
    reward_quadctrl                 — control cost = -0.1 * ||action||^2
    steps                           — timestep counter
    truncation                      — 1.0 if episode truncated this step
    measures                        — QD behavior measures [B, 2]
    first_obs, first_pipeline_state — values at episode start (state is None)

Dependencies:
    ppga (Actor, make_vec_env_brax)
    convert_state_dict.py  → EliteCheckpointDataset, get_elite_dataloader
"""

import argparse
import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import torch
from box import Box

from ppga.envs.brax_custom.brax_env import make_vec_env_brax
from ppga.models.actor_critic import Actor

from convert_state_dict import EliteCheckpointDataset, get_elite_dataloader


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

OBS_SHAPE    = (244,)
ACTION_SHAPE = (17,)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trajectory:
    """Stores everything collected during one policy rollout.

    Each list field accumulates one tensor per timestep.
    After collection, stack along dim=0 to get shape [T, B, ...].
    """
    obs:         List[torch.Tensor] = field(default_factory=list)  # [T, B, obs_dim]
    actions:     List[torch.Tensor] = field(default_factory=list)  # [T, B, act_dim]
    rewards:     List[torch.Tensor] = field(default_factory=list)  # [T, B]
    terminated:  List[torch.Tensor] = field(default_factory=list)  # [T, B]
    truncated:   List[torch.Tensor] = field(default_factory=list)  # [T, B]

    # Pre-computed scalars from the info dict — confirmed available.
    x_velocity:   List[torch.Tensor] = field(default_factory=list)  # [T, B]
    y_velocity:   List[torch.Tensor] = field(default_factory=list)  # [T, B]
    x_position:   List[torch.Tensor] = field(default_factory=list)  # [T, B]
    ctrl_cost:    List[torch.Tensor] = field(default_factory=list)  # [T, B]  reward_quadctrl
    steps:        List[torch.Tensor] = field(default_factory=list)  # [T, B]
    truncation:   List[torch.Tensor] = field(default_factory=list)  # [T, B]
    qd_measures:  List[torch.Tensor] = field(default_factory=list)  # [T, B, 2]

    # Raw brax State objects for video rendering (env 0 only).
    # Only populated when capture_video=True is passed to rollout().
    brax_states: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Video helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_brax_sys(env):
    """Walk the wrapper chain to find the brax System object needed by html.render.

    TorchWrapper -> GymWrapper (brax) -> VmapWrapper/training -> actual env with .sys
    """
    cur = env
    visited = set()
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        if hasattr(cur, "sys"):
            return cur.sys
        for attr in ("env", "_env", "unwrapped"):
            if hasattr(cur, attr):
                cur = getattr(cur, attr)
                break
        else:
            break
    raise AttributeError(
        "Could not find .sys on any wrapper. "
        "Inspect with: cur = env; while hasattr(cur, 'env'): print(type(cur)); cur = cur.env"
    )


def save_trajectory_video(env, brax_states: list, out_path: str, subsample: int = 1):
    """Render a list of brax State objects to a self-contained HTML file.

    The HTML file can be opened in any browser and plays back the trajectory
    as an interactive 3D animation via Three.js (no server required).

    Args:
        env          : TorchWrapper vec-env (used to walk to brax .sys)
        brax_states  : list of brax State objects, already sliced to env index 0
        out_path     : .html path to write
        subsample    : keep every Nth state (default 1 = all states already subsampled
                       at capture time via video_subsample in rollout)
    """
    try:
        import jax  # noqa: F401
        from brax.io import html as brax_html
    except ImportError as e:
        print(f"  [video] Skipping — missing dependency: {e}")
        return

    try:
        sys = _get_brax_sys(env)
    except AttributeError as e:
        print(f"  [video] Skipping — {e}")
        return

    # Extract pipeline_state from each State if needed
    pipeline_states = [getattr(s, "pipeline_state", s) for s in brax_states[::subsample]]

    html_str = brax_html.render(sys, pipeline_states, height=480, colab=False)

    import os
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html_str)
    print(f"  [video] Saved {out_path}  ({len(pipeline_states)} frames)")


@torch.no_grad()
def rollout(
    actor: Actor,
    env,
    rollout_length: int,
    device: torch.device,
    capture_video: bool = False,
    video_subsample: int = 2,
) -> Trajectory:
    """
    Roll out a single policy for rollout_length steps.

    The env returns a 4-tuple from step():
        obs, reward, terminated, info = env.step(action)

    'terminated' here is a scalar done flag (not a bool terminated/truncated
    split).  The info dict carries 'truncation' as a separate field.

    Args:
        actor          : loaded Actor with obs_normalizer already set
        env            : TorchWrapper vec-env (env_batch_size envs in parallel)
        rollout_length : number of env steps to collect
        device         : torch device

    Returns:
        Trajectory with all rollout_length steps populated
    """
    actor.eval()
    actor.to(device)

    traj = Trajectory()

    reset_out = env.reset()
    # env.reset() returns either a raw obs tensor or a (obs, info) tuple
    # depending on the gym wrapper version. Normalise to always get the tensor.
    obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
    obs = obs.to(device)

    episode_done_0 = False  # stop capturing after env 0 finishes its first episode
    for _ in range(rollout_length):
        # Normalize observations if the actor has a normalizer
        norm_obs = actor.obs_normalizer(obs) if actor.obs_normalizer is not None else obs

        # Deterministic action (mean of the Gaussian policy, matching ppo.evaluate)
        action = actor.actor_mean(norm_obs)  # deterministic: use distribution mean, skip sampling
        action = action.to(device)

        # 4-tuple return (confirmed by inspect_env.py)
        next_obs, reward, terminated, info = env.step(action)

        traj.obs.append(obs.cpu())
        traj.actions.append(action.cpu())
        traj.rewards.append(reward.cpu())
        traj.terminated.append(terminated.cpu())
        traj.truncated.append(info['truncation'].cpu())

        traj.x_velocity.append(info['x_velocity'].cpu())
        traj.y_velocity.append(info['y_velocity'].cpu())
        traj.x_position.append(info['x_position'].cpu())
        traj.ctrl_cost.append(info['reward_quadctrl'].cpu())
        traj.steps.append(info['steps'].cpu())
        traj.truncation.append(info['truncation'].cpu())
        traj.qd_measures.append(info['measures'].cpu())

        # Capture brax state for video (env 0 only, up to its first episode end)
        done_0 = (terminated[0].item() > 0) or (info["truncation"][0].item() > 0)
        if done_0:
            episode_done_0 = True
        if capture_video and not episode_done_0 and len(traj.obs) % video_subsample == 0:
            try:
                import jax
                raw_state = env.env.env.env._state
                state_0 = jax.tree_util.tree_map(lambda x: x[0], raw_state)
                traj.brax_states.append(state_0)
            except Exception:
                pass  # silently skip if state not accessible this step

        obs = next_obs.to(device)

    return traj


# ─────────────────────────────────────────────────────────────────────────────
# Measurable functions
# Each function: Trajectory → Tensor[B]
# Higher is always better (binary comparison checks which score is larger).
# ─────────────────────────────────────────────────────────────────────────────

def avg_forward_speed(traj: Trajectory) -> torch.Tensor:
    """
    Mean forward (x-axis) velocity of the root body across all timesteps.
    Directly available as info['x_velocity'].

    Returns: Tensor[B]
    """
    vel = torch.stack(traj.x_velocity, dim=0)   # [T, B]
    return vel.mean(dim=0)                        # [B]


def energy_efficiency(traj: Trajectory) -> torch.Tensor:
    """
    Negative mean control cost across the trajectory.

    info['reward_quadctrl'] = -0.1 * ||action||^2 per step, so it is already
    negative — we negate it again so higher = less energy used = better.

    Returns: Tensor[B]
    """
    ctrl = torch.stack(traj.ctrl_cost, dim=0)   # [T, B]  each value ≤ 0
    return -ctrl.mean(dim=0)                     # [B]  negate → higher is better


def survival_length(traj: Trajectory) -> torch.Tensor:
    """
    Number of steps before the first episode termination (done = terminated
    OR truncated).  If the episode never ends, returns the full rollout length.

    Returns: Tensor[B]
    """
    terminated = torch.stack(traj.terminated,  dim=0)   # [T, B]
    truncated  = torch.stack(traj.truncation,  dim=0)   # [T, B]
    done = ((terminated + truncated) > 0).float()        # [T, B]

    T = done.shape[0]
    first_done = torch.full((done.shape[1],), float(T))
    for t in range(T):
        newly_done = (done[t] > 0) & (first_done == float(T))
        first_done[newly_done] = float(t + 1)
    return first_done   # [B]


def lateral_stability(traj: Trajectory) -> torch.Tensor:
    """
    Negative absolute mean lateral (y-axis) velocity.

    A policy that stays on a straight path has near-zero y_velocity throughout.
    Lower |y_velocity| → more stable → higher score after negation.

    Note: this replaces avg_foot_height, which requires pipeline_state
    (unavailable — confirmed None by inspect_env.py).

    Returns: Tensor[B]
    """
    y_vel = torch.stack(traj.y_velocity, dim=0)   # [T, B]
    return -y_vel.abs().mean(dim=0)               # [B]


def com_height_stability(traj: Trajectory) -> torch.Tensor:
    """
    Negative standard deviation of the cumulative x_position over time.

    Serves as a proxy for locomotion consistency: a policy that moves at a
    steady pace accumulates x_position smoothly (low variance in the
    incremental steps).  We compute step-wise displacements and take their
    std, so the metric is independent of episode speed.

    Note: direct CoM height requires pipeline_state (unavailable).

    Returns: Tensor[B]
    """
    x_pos = torch.stack(traj.x_position, dim=0)    # [T, B]
    # Step-to-step displacement along x
    dx = x_pos[1:] - x_pos[:-1]                    # [T-1, B]
    if dx.shape[0] == 0:
        return torch.zeros(x_pos.shape[1])
    return -dx.std(dim=0)                           # [B]  negate → lower variance is better





def foot_contact_rate(traj: Trajectory) -> torch.Tensor:
    """
    Mean fraction of timesteps where at least one foot is in contact with
    the ground.  Uses the QD boolean measures (confirmed {0,1}-valued).

    Higher values indicate a more grounded, stable gait; lower values
    suggest a more dynamic, aerial gait (e.g. running vs walking).

    Returns: Tensor[B]
    """
    # qd_measures: [T, B, 2]  where dim 2 = (left_foot, right_foot) contact
    m = torch.stack(traj.qd_measures, dim=0).float()   # [T, B, 2]
    any_contact = (m.sum(dim=-1) > 0).float()           # [T, B]  1 if either foot down
    return any_contact.mean(dim=0)                       # [B]


def gait_symmetry(traj: Trajectory) -> torch.Tensor:
    """
    Negative absolute difference in contact rate between left and right foot.

    A perfectly symmetric gait has equal contact fractions for both feet.
    Lower asymmetry → higher score after negation.

    Returns: Tensor[B]
    """
    m = torch.stack(traj.qd_measures, dim=0).float()   # [T, B, 2]
    left_rate  = m[:, :, 0].mean(dim=0)                 # [B]
    right_rate = m[:, :, 1].mean(dim=0)                 # [B]
    return -(left_rate - right_rate).abs()               # [B]  negate → higher is more symmetric


def action_smoothness(traj: Trajectory) -> torch.Tensor:
    """
    Negative mean squared change in actions between consecutive timesteps.

    Measures how jerky the control signal is.  A smooth policy produces
    similar actions on adjacent steps; a jerky one oscillates rapidly.
    Lower jerk → higher score after negation.

    Returns: Tensor[B]
    """
    acts = torch.stack(traj.actions, dim=0)   # [T, B, A]
    if acts.shape[0] < 2:
        return torch.zeros(acts.shape[1])
    delta = acts[1:] - acts[:-1]              # [T-1, B, A]
    jerk  = (delta ** 2).sum(dim=-1)          # [T-1, B]  per-step squared change
    return -jerk.mean(dim=0)                  # [B]  negate → lower jerk is better


def distance_traveled(traj: Trajectory) -> torch.Tensor:
    """
    Net forward displacement: x_position[-1] - x_position[0].

    Unlike avg_forward_speed (which averages velocities), this directly
    measures how far the policy actually moved the agent, accounting for
    episodes that reset mid-rollout.

    Returns: Tensor[B]
    """
    x = torch.stack(traj.x_position, dim=0)   # [T, B]
    return x[-1] - x[0]                        # [B]


def speed_efficiency(traj: Trajectory) -> torch.Tensor:
    """
    Forward speed per unit of control cost: mean(x_velocity) / mean(|ctrl_cost|).

    Captures the trade-off between locomotion speed and energy expenditure.
    A policy that moves fast but wastes energy scores lower than one that
    achieves similar speed more economically.

    Numerically stable: if ctrl_cost is near zero (free locomotion),
    returns mean speed directly.

    Returns: Tensor[B]
    """
    vel  = torch.stack(traj.x_velocity, dim=0).mean(dim=0)   # [B]
    cost = torch.stack(traj.ctrl_cost,  dim=0).abs().mean(dim=0)  # [B]  ≥ 0
    return vel / (cost + 1e-8)                                # [B]


def arm_stability(traj: Trajectory) -> torch.Tensor:
    """
    Negative mean squared deviation of arm joint actions from their
    time-averaged value.

    Measures how much the arm joints oscillate during locomotion.  A stable
    policy holds its arms in a consistent pose; an unstable one swings them
    erratically.  Lower variance → higher score after negation.

    Arm joint indices for the standard Brax humanoid (17 actuators):
        11: right_shoulder1   12: right_shoulder2   13: right_elbow
        14: left_shoulder1    15: left_shoulder2    16: left_elbow
    Adjust ARM_JOINT_INDICES below if your env uses a different ordering.

    Returns: Tensor[B]
    """
    ARM_JOINT_INDICES = [11, 12, 13, 14, 15, 16]

    acts = torch.stack(traj.actions, dim=0)              # [T, B, A]
    arm  = acts[:, :, ARM_JOINT_INDICES]                 # [T, B, 6]
    mean = arm.mean(dim=0, keepdim=True)                 # [1, B, 6]
    var  = ((arm - mean) ** 2).mean(dim=(0, 2))          # [B]
    return -var                                          # [B]  negate → lower variance is better

# Registry: name → function.  Add / remove freely.
MEASURABLES: Dict[str, Callable[[Trajectory], torch.Tensor]] = {
    'avg_forward_speed':    avg_forward_speed,
    'energy_efficiency':    energy_efficiency,
    'survival_length':      survival_length,
    'lateral_stability':    lateral_stability,
    'com_height_stability': com_height_stability,
    'foot_contact_rate':    foot_contact_rate,
    'gait_symmetry':        gait_symmetry,
    'action_smoothness':    action_smoothness,
    'distance_traveled':    distance_traveled,
    'speed_efficiency':     speed_efficiency,
    'arm_stability':        arm_stability,
}


# ─────────────────────────────────────────────────────────────────────────────
# Policy loader
# ─────────────────────────────────────────────────────────────────────────────

def load_actor(checkpoint: dict, device: torch.device) -> Actor:
    """Reconstruct an Actor from a checkpoint dict (as saved by convert_state_dict.py)."""
    actor = Actor(OBS_SHAPE, ACTION_SHAPE,
                  normalize_obs=True, normalize_returns=False)
    actor.load_state_dict(checkpoint['model_state_dict'], strict=False)
    actor.actor_logstd.data = torch.zeros_like(actor.actor_logstd.data)
    if checkpoint.get('obs_normalizer_state') is not None:
        actor.obs_normalizer.load_state_dict(checkpoint['obs_normalizer_state'])
    return actor.to(device)
# ─────────────────────────────────────────────────────────────────────────────
# HDF5 serialisation
# ─────────────────────────────────────────────────────────────────────────────

def save_trajectory_h5(traj: Trajectory, scores: dict, path: str):
    """Save one trajectory's raw fields and pre-computed scores to HDF5.

    File layout:
        /fields/obs            float32 [T, B, obs_dim]
        /fields/actions        float32 [T, B, act_dim]
        /fields/rewards        float32 [T, B]
        /fields/terminated     float32 [T, B]
        /fields/x_velocity     float32 [T, B]
        /fields/y_velocity     float32 [T, B]
        /fields/x_position     float32 [T, B]
        /fields/ctrl_cost      float32 [T, B]
        /fields/qd_measures    float32 [T, B, 2]
        /scores/<fn_name>      float32 [B]   — one value per parallel env
    """
    import h5py, numpy as np, os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, 'w') as f:
        g = f.create_group('fields')
        def _stack(lst): return torch.stack(lst, dim=0).numpy()
        g.create_dataset('obs',        data=_stack(traj.obs),        compression='gzip')
        g.create_dataset('actions',    data=_stack(traj.actions),    compression='gzip')
        g.create_dataset('rewards',    data=_stack(traj.rewards),    compression='gzip')
        g.create_dataset('terminated', data=_stack(traj.terminated), compression='gzip')
        g.create_dataset('x_velocity', data=_stack(traj.x_velocity), compression='gzip')
        g.create_dataset('y_velocity', data=_stack(traj.y_velocity), compression='gzip')
        g.create_dataset('x_position', data=_stack(traj.x_position), compression='gzip')
        g.create_dataset('ctrl_cost',  data=_stack(traj.ctrl_cost),  compression='gzip')
        g.create_dataset('qd_measures',data=_stack(traj.qd_measures),compression='gzip')
        sg = f.create_group('scores')
        for fn_name, score_tensor in scores.items():
            sg.create_dataset(fn_name, data=score_tensor.numpy())


def load_trajectory_from_h5(path: str) -> Tuple[Trajectory, dict]:
    """Load raw fields and pre-computed scores from an HDF5 file.

    Returns:
        traj   : Trajectory with list fields populated (each element is [B, ...]
                 rather than [T, B, ...] — the time dimension is split back out)
        scores : dict[fn_name → Tensor[B]]
    """
    import h5py
    traj = Trajectory()
    with h5py.File(path, 'r') as f:
        def _split(key):
            arr = torch.from_numpy(f['fields'][key][:])   # [T, B, ...]
            return [arr[t] for t in range(arr.shape[0])]  # list of [B, ...]

        traj.obs         = _split('obs')
        traj.actions     = _split('actions')
        traj.rewards     = _split('rewards')
        traj.terminated  = _split('terminated')
        traj.x_velocity  = _split('x_velocity')
        traj.y_velocity  = _split('y_velocity')
        traj.x_position  = _split('x_position')
        traj.ctrl_cost   = _split('ctrl_cost')
        traj.qd_measures = _split('qd_measures')
        # truncation mirrors terminated for scoring purposes
        traj.truncated   = traj.terminated
        traj.truncation  = traj.terminated

        scores = {k: torch.from_numpy(f['scores'][k][:])
                  for k in f['scores']}
    return traj, scores


def load_scores_from_h5(path: str) -> dict:
    """Load only the pre-computed scores from an HDF5 file (cheap)."""
    import h5py
    with h5py.File(path, 'r') as f:
        return {k: torch.from_numpy(f['scores'][k][:]) for k in f['scores']}


# ─────────────────────────────────────────────────────────────────────────────
# Eval result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Comparison results for one (policy_i, policy_j, traj_k) triple."""
    policy_a_index: int
    policy_b_index: int
    traj_index:     int          # which trajectory index k this corresponds to
    scores: Dict[str, int]       = field(default_factory=dict)   # +1 / -1 / 0
    raw_a:  Dict[str, torch.Tensor] = field(default_factory=dict)
    raw_b:  Dict[str, torch.Tensor] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: roll out K trajectories per policy, save to HDF5
# ─────────────────────────────────────────────────────────────────────────────

def rollout_phase(cfg, env, all_ckpts: list, traj_dir: str, device: torch.device):
    """
    For each of N policies, roll out K trajectories and save each to:
        <traj_dir>/elite{idx:04d}_traj{k:02d}.h5

    Skips trajectories whose HDF5 already exists (safe to resume).
    Videos are saved here if --save_video is set.
    """
    import os
    fn_names = list(MEASURABLES.keys())
    N, K = len(all_ckpts), cfg.num_trajectories
    os.makedirs(traj_dir, exist_ok=True)

    print(f'\n{"═"*60}')
    print(f'Phase 1: rolling out {K} trajectories × {N} policies')
    print(f'{"═"*60}')

    for i, ckpt in enumerate(all_ckpts):
        elite_idx = int(ckpt['elite_index'])
        obj       = float(ckpt['objective'])

        # Check which trajectories still need to be rolled out
        missing = [k for k in range(K)
                   if not os.path.exists(_traj_path(traj_dir, elite_idx, k))]
        if not missing:
            print(f'  [{i+1}/{N}] elite #{elite_idx} — all {K} trajectories already exist, skipping')
            continue

        print(f'\n  [{i+1}/{N}] elite #{elite_idx}  (obj={obj:.2f}) — rolling out {len(missing)} trajectories')
        actor = load_actor(ckpt, device)

        for k in missing:
            capture = cfg.save_video and k == 0   # only capture video for traj 0
            traj = rollout(actor, env, cfg.rollout_length, device,
                           capture_video=capture,
                           video_subsample=cfg.video_subsample)

            # Save video before clearing brax_states
            if capture and traj.brax_states:
                vid_path = os.path.join(cfg.video_dir, f'elite{elite_idx:04d}.html')
                save_trajectory_video(env, traj.brax_states, vid_path)
            traj.brax_states.clear()

            # Compute scores and save HDF5
            scores = {fn: MEASURABLES[fn](traj) for fn in fn_names}   # each [B]
            h5_path = _traj_path(traj_dir, elite_idx, k)
            save_trajectory_h5(traj, scores, h5_path)
            print(f'    traj {k}: saved → {h5_path}')
            del traj

        actor.cpu(); del actor; torch.cuda.empty_cache()


def _traj_path(traj_dir: str, elite_idx: int, k: int) -> str:
    return f'{traj_dir}/elite{elite_idx:04d}_traj{k:02d}.h5'


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: compare K matched pairs for every (i, j) policy pair
# ─────────────────────────────────────────────────────────────────────────────

def compare_phase(cfg, all_ckpts: list, traj_dir: str) -> Tuple[torch.Tensor, List[EvalResult]]:
    """
    Load pre-computed scores from HDF5 files and build the binary matrix.

    For each of the N*(N-1)/2 pairs and each of the K matched trajectory
    indices, compare policy i's traj_k against policy j's traj_k.

    Binary matrix shape: [num_fns, num_pairs, K]
        +1 = policy i wins on traj k
        -1 = policy j wins on traj k
         0 = tie
    """
    fn_names  = list(MEASURABLES.keys())
    num_fns   = len(fn_names)
    N, K      = len(all_ckpts), cfg.num_trajectories
    all_pairs = list(itertools.combinations(range(N), 2))
    num_pairs = len(all_pairs)

    print(f'\n{"═"*60}')
    print(f'Phase 2: comparing {num_pairs} pairs × {K} trajectories')
    print(f'{"═"*60}')

    # Load all scores upfront (scores only, not raw fields — cheap)
    print('  Loading scores from HDF5...')
    # all_policy_scores[i][k] = dict[fn_name → Tensor[B]]
    all_policy_scores = []
    for i, ckpt in enumerate(all_ckpts):
        elite_idx = int(ckpt['elite_index'])
        traj_scores = []
        for k in range(K):
            path = _traj_path(traj_dir, elite_idx, k)
            traj_scores.append(load_scores_from_h5(path))
        all_policy_scores.append(traj_scores)

    binary_matrix = torch.zeros(num_fns, num_pairs, K, dtype=torch.int8)
    all_results: List[EvalResult] = []

    for pair_idx, (i, j) in enumerate(all_pairs):
        idx_i = int(all_ckpts[i]['elite_index'])
        idx_j = int(all_ckpts[j]['elite_index'])

        print(f'\n  Pair {pair_idx+1}/{num_pairs}: elite #{idx_i} vs #{idx_j}')
        print(f'     {"Function":<25}  ' + '  '.join(f'traj{k:02d}' for k in range(K)))
        print('     ' + '-' * (27 + 8 * K))

        for k in range(K):
            scores_i = torch.stack([all_policy_scores[i][k][fn] for fn in fn_names]).mean(dim=1)  # [num_fns]
            scores_j = torch.stack([all_policy_scores[j][k][fn] for fn in fn_names]).mean(dim=1)  # [num_fns]

            binary = torch.sign(scores_i - scores_j).to(torch.int8)   # [num_fns]
            binary_matrix[:, pair_idx, k] = binary

            result = EvalResult(policy_a_index=idx_i, policy_b_index=idx_j, traj_index=k)
            for fn_idx, fn_name in enumerate(fn_names):
                b = int(binary[fn_idx].item())
                result.scores[fn_name] = b
                result.raw_a[fn_name]  = all_policy_scores[i][k][fn_name]
                result.raw_b[fn_name]  = all_policy_scores[j][k][fn_name]
            all_results.append(result)

        # Print per-function summary across K trajectories
        for fn_idx, fn_name in enumerate(fn_names):
            row = '  '.join(
                f'{binary_matrix[fn_idx, pair_idx, k].item():>+6}'
                for k in range(K)
            )
            print(f'     {fn_name:<25}  {row}')

    return binary_matrix, all_results


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(binary_matrix: torch.Tensor, results: List[EvalResult]):
    """
    Aggregate win/loss/tie counts across all pairs and all K trajectories.

    binary_matrix shape: [num_fns, num_pairs, K]
    """
    fn_names  = list(MEASURABLES.keys())
    num_pairs = binary_matrix.shape[1]
    K         = binary_matrix.shape[2]

    # Aggregate over both pairs and trajectories
    wins_a = (binary_matrix == +1).sum(dim=(1, 2))   # [num_fns]
    wins_b = (binary_matrix == -1).sum(dim=(1, 2))   # [num_fns]
    ties   = (binary_matrix ==  0).sum(dim=(1, 2))   # [num_fns]
    total  = num_pairs * K

    print('\n' + '=' * 66)
    print(f'Summary  —  {num_pairs} pairs × {K} trajectories = {total} comparisons')
    print('=' * 66)
    print(f'  {"Function":<25}  {"i wins":>8}  {"j wins":>8}  {"Ties":>6}  {"i win%":>8}')
    print('  ' + '-' * 62)
    for fi, fn in enumerate(fn_names):
        wa, wb, t = wins_a[fi].item(), wins_b[fi].item(), ties[fi].item()
        pct = 100.0 * wa / total if total > 0 else 0.0
        print(f'  {fn:<25}  {wa:>8}  {wb:>8}  {t:>6}  {pct:>7.1f}%')
    print()
    print(f'binary_matrix shape: {tuple(binary_matrix.shape)}  '
          f'(num_functions, num_pairs, K)')


# ─────────────────────────────────────────────────────────────────────────────
# Arg parsing + entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Exhaustive pairwise policy evaluator')
    parser.add_argument('--archive_path',     type=str, required=True,
                        help='Path to pickled pyribs archive DataFrame (.pkl)')
    parser.add_argument('--output_dir',       type=str, required=True,
                        help='Experiment output directory.  Layout:\n'
                             '  <output_dir>/trajectories/  — HDF5 files\n'
                             '  <output_dir>/videos/        — HTML videos\n'
                             '  <output_dir>/binary_matrix.pt')
    parser.add_argument('--env_name',         type=str, default='humanoid')
    parser.add_argument('--env_batch_size',   type=int, default=32,
                        help='Parallel envs per rollout (the B dimension)')
    parser.add_argument('--rollout_length',   type=int, default=1000,
                        help='Steps per rollout')
    parser.add_argument('--num_trajectories', type=int, default=3,
                        help='Number of trajectories K to roll out per policy')
    parser.add_argument('--seed',             type=int, default=0)
    parser.add_argument('--clip_obs_rew',     action='store_true', default=False)
    parser.add_argument('--save_video',       action='store_true', default=False,
                        help='Render trajectory 0 of each policy as an HTML video')
    parser.add_argument('--video_subsample',  type=int, default=2,
                        help='Capture every Nth frame during rollout')
    return Box(vars(parser.parse_args()))


if __name__ == '__main__':
    import os
    cfg    = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(cfg.output_dir, exist_ok=True)
    traj_dir      = os.path.join(cfg.output_dir, 'trajectories')
    cfg.video_dir = os.path.join(cfg.output_dir, 'videos')
    if cfg.save_video:
        os.makedirs(cfg.video_dir, exist_ok=True)

    env_cfg = Box({
        'env_name':       cfg.env_name,
        'env_batch_size': cfg.env_batch_size,
        'seed':           cfg.seed,
        'clip_obs_rew':   cfg.clip_obs_rew,
    })
    env       = make_vec_env_brax(env_cfg)
    dataset   = EliteCheckpointDataset(cfg.archive_path)
    all_ckpts = [dataset[i] for i in range(len(dataset))]

    print(f'[eval] env={cfg.env_name}, B={cfg.env_batch_size}, '
          f'T={cfg.rollout_length}, K={cfg.num_trajectories}, device={device}')
    print(f'[eval] {len(all_ckpts)} checkpoints, output → {cfg.output_dir}')

    # ── Phase 1: rollout ─────────────────────────────────────────────────────
    rollout_phase(cfg, env, all_ckpts, traj_dir, device)

    # ── Phase 2: compare ─────────────────────────────────────────────────────
    binary_matrix, results = compare_phase(cfg, all_ckpts, traj_dir)
    print_summary(binary_matrix, results)

    matrix_path = os.path.join(cfg.output_dir, 'binary_matrix.pt')
    torch.save({
        'binary_matrix': binary_matrix,                                          # [num_fns, num_pairs, K]
        'fn_names':      list(MEASURABLES.keys()),
        'pair_indices':  [(r.policy_a_index, r.policy_b_index) for r in results
                         if r.traj_index == 0],                                  # one entry per pair
        'num_trajectories': cfg.num_trajectories,
    }, matrix_path)
    print(f'\n[saved] binary_matrix → {matrix_path}')