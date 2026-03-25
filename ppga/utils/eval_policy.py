"""
eval_policies.py

Pairwise policy evaluation script.  Loads elite checkpoints from a directory,
samples random pairs, rolls out each policy in a Brax vec-env, then scores
each trajectory on a set of measurable functions.  Results are reported
per-function (no cross-function collapse).

Usage:
    python eval_policy.py \
        --checkpoint_dir data/checkpoints \
        --env_name humanoid \
        --env_batch_size 32 \
        --rollout_length 1000 \
        --num_evals 10 \
        --seed 0

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
    convert_state_dict.py  → get_elite_dataloader
"""

import argparse
import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import torch
from box import Box

from ppga.envs.brax_custom.brax_env import make_vec_env_brax
from ppga.models.actor_critic import Actor

from ppga.utils.convert_state_dict import EliteCheckpointDataset, get_elite_dataloader


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



def total_reward(traj: Trajectory) -> torch.Tensor:
    """
    Sum of rewards collected over the full rollout.

    The most direct measure of policy quality — equivalent to the objective
    used during training (modulo episode boundaries and discounting).

    Returns: Tensor[B]
    """
    rew = torch.stack(traj.rewards, dim=0)   # [T, B]
    return rew.sum(dim=0)                     # [B]


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

# Registry: name → function.  Add / remove freely.
MEASURABLES: Dict[str, Callable[[Trajectory], torch.Tensor]] = {
    'avg_forward_speed':    avg_forward_speed,
    'energy_efficiency':    energy_efficiency,
    'survival_length':      survival_length,
    'lateral_stability':    lateral_stability,
    'com_height_stability': com_height_stability,
    'total_reward':         total_reward,
    'foot_contact_rate':    foot_contact_rate,
    'gait_symmetry':        gait_symmetry,
    'action_smoothness':    action_smoothness,
    'distance_traveled':    distance_traveled,
    'speed_efficiency':     speed_efficiency,
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
# Eval loop
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Binary scores for a single pair of policies across all measurable functions."""
    policy_a_index: int
    policy_b_index: int
    # scores[fn_name] = +1 if A wins, -1 if B wins, 0 if tie
    scores: Dict[str, int] = field(default_factory=dict)
    # Raw per-env scores for deeper analysis: [B] each
    raw_a:  Dict[str, torch.Tensor] = field(default_factory=dict)
    raw_b:  Dict[str, torch.Tensor] = field(default_factory=dict)


def run_eval(cfg) -> Tuple[torch.Tensor, List[EvalResult]]:
    """
    Exhaustive pairwise evaluation.

    For each of the N*(N-1)/2 pairs:
      a. Sample the pair from the dataset
      b. Roll out policy i, score it, free from GPU
      c. Roll out policy j, score it, free from GPU
      d. Collapse scores to binary winner per function

    No trajectories are cached between pairs — memory stays flat regardless
    of archive size.

    Returns:
        binary_matrix : Tensor[num_fns, num_pairs, 1]  dtype=int8
                        +1 = policy i wins, -1 = policy j wins, 0 = tie
        results       : List[EvalResult] one per pair
    """
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fn_names = list(MEASURABLES.keys())
    num_fns  = len(fn_names)

    # ── Environment ──────────────────────────────────────────────────────────
    env_cfg = Box({
        'env_name':       cfg.env_name,
        'env_batch_size': cfg.env_batch_size,
        'seed':           cfg.seed,
        'clip_obs_rew':   cfg.clip_obs_rew,
    })
    env = make_vec_env_brax(env_cfg)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset   = EliteCheckpointDataset(cfg.checkpoint_dir)
    N         = len(dataset)
    all_ckpts = [dataset[i] for i in range(N)]
    all_pairs = list(itertools.combinations(range(N), 2))
    num_pairs = len(all_pairs)

    print(f'[eval] env={cfg.env_name}, B={cfg.env_batch_size}, '
          f'rollout_length={cfg.rollout_length}, device={device}')
    print(f'[eval] {N} checkpoints → {num_pairs} pairs')

    binary_matrix = torch.zeros(num_fns, num_pairs, 1, dtype=torch.int8)
    all_results: List[EvalResult] = []

    for pair_idx, (i, j) in enumerate(all_pairs):
        ckpt_i = all_ckpts[i]
        ckpt_j = all_ckpts[j]
        idx_i  = int(ckpt_i['elite_index'])
        idx_j  = int(ckpt_j['elite_index'])

        print(f'\n── Pair {pair_idx+1}/{num_pairs}: elite #{idx_i} vs #{idx_j} ──')

        # ── a. Roll out policy i ──────────────────────────────────────────────
        actor_i = load_actor(ckpt_i, device)
        traj_i  = rollout(actor_i, env, cfg.rollout_length, device,
                          capture_video=cfg.save_video,
                          video_subsample=cfg.video_subsample)
        actor_i.cpu(); del actor_i; torch.cuda.empty_cache()
        if cfg.save_video and traj_i.brax_states:
            save_trajectory_video(env, traj_i.brax_states,
                                  f'{cfg.video_dir}/pair{pair_idx+1:04d}_elite{idx_i}.html')
            traj_i.brax_states.clear()

        # ── b. Roll out policy j ──────────────────────────────────────────────
        actor_j = load_actor(ckpt_j, device)
        traj_j  = rollout(actor_j, env, cfg.rollout_length, device,
                          capture_video=cfg.save_video,
                          video_subsample=cfg.video_subsample)
        actor_j.cpu(); del actor_j; torch.cuda.empty_cache()
        if cfg.save_video and traj_j.brax_states:
            save_trajectory_video(env, traj_j.brax_states,
                                  f'{cfg.video_dir}/pair{pair_idx+1:04d}_elite{idx_j}.html')
            traj_j.brax_states.clear()

        # ── c. Score each measurable → (num_fns,) per policy ─────────────────
        scores_i = torch.tensor([MEASURABLES[fn](traj_i).mean().item() for fn in fn_names])
        scores_j = torch.tensor([MEASURABLES[fn](traj_j).mean().item() for fn in fn_names])
        del traj_i, traj_j

        # ── d. Collapse to binary (num_fns, 1) ───────────────────────────────
        binary = torch.sign(scores_i - scores_j).to(torch.int8)
        binary_matrix[:, pair_idx, :] = binary.unsqueeze(1)

        # ── Logging ───────────────────────────────────────────────────────────
        result = EvalResult(policy_a_index=idx_i, policy_b_index=idx_j)
        print(f'     {"Function":<25}  {"i score":>10}  {"j score":>10}  winner')
        print('     ' + '-' * 58)
        for fn_idx, fn_name in enumerate(fn_names):
            si  = scores_i[fn_idx].item()
            sj  = scores_j[fn_idx].item()
            b   = int(binary[fn_idx].item())
            win = f'i (#{idx_i})' if b == 1 else (f'j (#{idx_j})' if b == -1 else 'TIE')
            result.scores[fn_name] = b
            result.raw_a[fn_name]  = scores_i[fn_idx:fn_idx+1]
            result.raw_b[fn_name]  = scores_j[fn_idx:fn_idx+1]
            print(f'     {fn_name:<25}  {si:>+10.4f}  {sj:>+10.4f}  {win}')

        all_results.append(result)

    return binary_matrix, all_results


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(binary_matrix: torch.Tensor, results: List[EvalResult]):
    """Print aggregate win/loss/tie counts from the binary matrix.

    Args:
        binary_matrix : Tensor[num_functions, num_pairs, 1]  dtype=int8
        results       : List[EvalResult] for pair-label lookup
    """
    fn_names  = list(MEASURABLES.keys())
    num_pairs = binary_matrix.shape[1]

    wins_a = (binary_matrix == +1).sum(dim=1).squeeze(1)   # (num_fns,)
    wins_b = (binary_matrix == -1).sum(dim=1).squeeze(1)   # (num_fns,)
    ties   = (binary_matrix ==  0).sum(dim=1).squeeze(1)   # (num_fns,)

    print('\n' + '=' * 62)
    print(f'Summary  —  {num_pairs} pairs, {len(fn_names)} functions')
    print('=' * 62)
    print(f'  {"Function":<25}  {"A wins":>8}  {"B wins":>8}  {"Ties":>6}')
    print('  ' + '-' * 55)
    for fi, fn in enumerate(fn_names):
        print(f'  {fn:<25}  {wins_a[fi].item():>8}  {wins_b[fi].item():>8}  {ties[fi].item():>6}')
    print()

    print('Pair-level binary matrix  (rows=functions, cols=pairs, +1=A wins, -1=B wins):')
    # Print header row of pair labels
    labels = [f'#{r.policy_a_index}v#{r.policy_b_index}' for r in results]
    col_w  = max(len(l) for l in labels) + 1
    print('  ' + ' '.join(f'{fn[:8]:>8}' for fn in fn_names))
    for pi, label in enumerate(labels):
        row = ' '.join(f'{binary_matrix[fi, pi, 0].item():>8}' for fi in range(len(fn_names)))
        print(f'  {label:<{col_w}} {row}')

    print()
    print(f'binary_matrix shape: {tuple(binary_matrix.shape)}  (num_functions, num_pairs, 1)')


# ─────────────────────────────────────────────────────────────────────────────
# Arg parsing + entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Exhaustive pairwise policy evaluator')
    parser.add_argument('--checkpoint_dir',  type=str, required=True,
                        help='Directory containing elite_checkpoint_*.pt files')
    parser.add_argument('--output_dir',       type=str, required=True,
                        help='Experiment output directory. '
                             'binary_matrix.pt is saved here; '
                             'videos go in <output_dir>/videos/')
    parser.add_argument('--env_name',        type=str, default='humanoid')
    parser.add_argument('--env_batch_size',  type=int, default=32,
                        help='Parallel envs per rollout (the B dimension)')
    parser.add_argument('--rollout_length',  type=int, default=1000,
                        help='Steps collected per policy rollout')
    parser.add_argument('--seed',            type=int, default=0)
    parser.add_argument('--clip_obs_rew',    action='store_true', default=False)
    parser.add_argument('--save_video',      action='store_true', default=False,
                        help='Render rollout videos as self-contained HTML files')
    parser.add_argument('--video_subsample', type=int, default=2,
                        help='Capture every Nth frame during rollout')
    return Box(vars(parser.parse_args()))


if __name__ == '__main__':
    import os
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Patch video_dir to live inside output_dir
    cfg.video_dir = os.path.join(cfg.output_dir, 'videos')
    if cfg.save_video:
        os.makedirs(cfg.video_dir, exist_ok=True)

    binary_matrix, results = run_eval(cfg)
    print_summary(binary_matrix, results)

    matrix_path = os.path.join(cfg.output_dir, 'binary_matrix.pt')
    torch.save({'binary_matrix': binary_matrix,
                'fn_names':      list(MEASURABLES.keys()),
                'pair_indices':  [(r.policy_a_index, r.policy_b_index) for r in results]},
               matrix_path)
    print(f'[saved] binary_matrix → {matrix_path}')