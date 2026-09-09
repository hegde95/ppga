import argparse
import json
import os
import sys
import time
from distutils.util import strtobool
from pathlib import Path

import torch
from box import Box

from ppga.RL.ppo import PPO
from ppga.envs.factory import make_vec_env
from ppga.envs.qd_env import policy_observation_space
from ppga.utils.utilities import config_wandb, log, save_cfg


def _json_default(value):
    """Serialize tensor/NumPy diagnostics without weakening checkpoint data."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def _actor_state_dict(alg):
    actor_state = {
        key: value.detach().cpu()
        for key, value in alg.agents[0].state_dict().items()
    }
    # Indexing a VectorizedActor removes the one-policy dimension from
    # actor_logstd, while a standalone Actor expects (1, action_dim).
    if ('actor_logstd' in actor_state
            and actor_state['actor_logstd'].ndim == 1):
        actor_state['actor_logstd'] = actor_state['actor_logstd'].unsqueeze(0)
    return actor_state


def _checkpoint_payload(alg, cfg, update, global_step):
    payload = {
        'format_version': 1,
        'update': int(update),
        'global_step': int(global_step),
        'action_transform': cfg.action_transform,
        'actor_state_dict': _actor_state_dict(alg),
        'vec_inference_state_dict': alg.vec_inference.state_dict(),
        'optimizer_state_dict': alg.vec_optimizer.state_dict(),
        'qd_critic_state_dict': alg.qd_critic.state_dict(),
        'qd_critic_optimizer_state_dict': alg.qd_critic_optim.state_dict(),
        'mean_critic_state_dict': alg.mean_critic.state_dict(),
        'mean_critic_optimizer_state_dict': alg.mean_critic_optim.state_dict(),
    }
    if cfg.normalize_obs:
        payload['obs_normalizer_state_dicts'] = [
            normalizer.state_dict()
            for normalizer in alg.vec_inference.obs_normalizers
        ]
    if cfg.normalize_returns:
        payload['return_normalizer_state_dicts'] = [
            normalizer.state_dict()
            for normalizer in alg.vec_inference.rew_normalizers
        ]
    return payload


def _save_periodic_checkpoint(alg, cfg, outdir, update, global_step):
    checkpoint_dir = outdir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f'checkpoint_{global_step:012d}.pt'
    temporary_path = path.with_suffix('.tmp')
    torch.save(_checkpoint_payload(alg, cfg, update, global_step),
               temporary_path)
    temporary_path.replace(path)
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_*.pt'))
    for old_checkpoint in checkpoints[:-cfg.max_checkpoints]:
        old_checkpoint.unlink()
    log.info(f'Saved PPO checkpoint: {path}')


def _load_training_checkpoint(alg, cfg, checkpoint_path):
    checkpoint = torch.load(checkpoint_path,
                            map_location=alg.device,
                            weights_only=False)
    if 'vec_inference_state_dict' not in checkpoint:
        raise ValueError(
            'Resume requires a periodic PPO checkpoint with full optimizer state')
    saved_transform = checkpoint.get('action_transform')
    if saved_transform is not None and saved_transform != cfg.action_transform:
        raise ValueError(
            f'Checkpoint action transform is {saved_transform!r}, but the '
            f'current run uses {cfg.action_transform!r}')
    alg.vec_inference.load_state_dict(checkpoint['vec_inference_state_dict'])
    alg.vec_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    alg.qd_critic.load_state_dict(checkpoint['qd_critic_state_dict'])
    alg.qd_critic_optim.load_state_dict(
        checkpoint['qd_critic_optimizer_state_dict'])
    alg.mean_critic.load_state_dict(checkpoint['mean_critic_state_dict'])
    alg.mean_critic_optim.load_state_dict(
        checkpoint['mean_critic_optimizer_state_dict'])
    for normalizer, state in zip(
            getattr(alg.vec_inference, 'obs_normalizers', []),
            checkpoint.get('obs_normalizer_state_dicts', [])):
        normalizer.load_state_dict(state)
    for normalizer, state in zip(
            getattr(alg.vec_inference, 'rew_normalizers', []),
            checkpoint.get('return_normalizer_state_dicts', [])):
        normalizer.load_state_dict(state)
    return int(checkpoint['update']), int(checkpoint['global_step'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name',
                        type=str,
                        help="Choose from [QDAntBulletEnv-v0,"
                        "QDHalfCheetahBulletEnv-v0]")
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument(
        "--torch_deterministic",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--use_wandb",
                        default=False,
                        type=lambda x: bool(strtobool(x)),
                        help='Use weights and biases to track the exp')
    parser.add_argument('--wandb_run_name', type=str, default='ppo_ant')
    parser.add_argument('--wandb_group', type=str)
    parser.add_argument('--wandb_project', type=str, default='PPGA')
    parser.add_argument('--report_interval',
                        type=int,
                        default=5,
                        help='Log objective results every N updates')

    # algorithm args
    parser.add_argument('--total_timesteps', type=int, default=1000000)
    parser.add_argument('--env_type',
                        type=str,
                        choices=['brax', 'isaac', 'mjlab'],
                        default='brax',
                        help='Whether to use cpu-envs or gpu-envs for rollouts')
    # args for brax
    parser.add_argument('--env_batch_size',
                        default=1,
                        type=int,
                        help='Number of parallel environments to run')

    # args for cpu-envs
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of worker processes to spawn. '
        'Should always be <= number of logical cores on your machine')
    parser.add_argument(
        '--envs_per_worker',
        type=int,
        default=1,
        help='Num envs each worker process will step through sequentially')
    parser.add_argument(
        '--rollout_length',
        type=int,
        default=2048,
        help='the number of steps to run in each environment per policy rollout'
    )
    # ppo hyperparams
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument(
        '--anneal_lr',
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help='Toggle learning rate annealing for policy and value networks')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.99,
                        help='Discount factor for rewards')
    parser.add_argument('--gae_lambda',
                        type=float,
                        default=0.95,
                        help='Lambda discount used for general advantage est')
    parser.add_argument('--num_minibatches', type=int, default=32)
    parser.add_argument('--update_epochs',
                        type=int,
                        default=10,
                        help='The K epochs to update the policy')
    parser.add_argument("--norm_adv",
                        type=lambda x: bool(strtobool(x)),
                        default=True,
                        nargs="?",
                        const=True,
                        help="Toggles advantages normalization")
    parser.add_argument(
        "--norm_adv_per_minibatch",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Normalize advantages per minibatch instead of once per rollout")
    parser.add_argument(
        "--mixed_precision",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Use autocast in the vectorized actor forward pass")
    parser.add_argument("--clip_coef",
                        type=float,
                        default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip_value_coef",
                        type=float,
                        default=0.2,
                        help="value clipping coefficient")
    parser.add_argument(
        "--clip_vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help=
        "Toggles whether or not to use a clipped loss for the value function, as per the paper."
    )
    parser.add_argument("--entropy_coef",
                        type=float,
                        default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument("--vf_coef",
                        type=float,
                        default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max_grad_norm",
                        type=float,
                        default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl",
                        type=float,
                        default=None,
                        help="the target KL divergence threshold")
    parser.add_argument(
        "--adaptive_kl",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Adapt the optimizer learning rate around target_kl, matching "
        "the RSL-RL PPO schedule used by MJLab")
    parser.add_argument(
        '--normalize_obs',
        type=lambda x: bool(strtobool(x)),
        default=False,
        help=
        'Normalize observations across a batch using running mean and stddev')
    parser.add_argument(
        '--normalize_returns',
        type=lambda x: bool(strtobool(x)),
        default=False,
        help='Normalize rewards across a batch using running mean and stddev')
    parser.add_argument('--value_bootstrap',
                        type=lambda x: bool(strtobool(x)),
                        default=None,
                        help='Bootstrap artificial time limits (default: on for Isaac/MJLab)')

    parser.add_argument('--weight_decay',
                        type=float,
                        default=None,
                        help='Apply L2 weight regularization to the NNs')

    parser.add_argument('--clip_obs_rew',
                        type=lambda x: bool(strtobool(x)),
                        default=False,
                        help='Clip obs and rewards b/w -10 and 10')
    parser.add_argument('--action_transform',
                        choices=['none', 'clip', 'tanh'],
                        default=None,
                        help='Bound policy actions; defaults to tanh for Isaac and none for Brax')
    parser.add_argument(
        '--action_std_parameterization',
        choices=['log', 'direct'],
        default='log',
        help='Learn log standard deviation, or MJLab/RSL-RL-style direct std')
    parser.add_argument('--initial_action_std', type=float, default=1.0,
                        help='Initial standard deviation for Gaussian actions')
    parser.add_argument('--actor_hidden_dims', type=int, nargs=3,
                        default=(400, 200, 100),
                        metavar=('H1', 'H2', 'H3'),
                        help='Actor hidden-layer widths')
    parser.add_argument('--eval_deterministic',
                        type=lambda x: bool(strtobool(x)),
                        default=True,
                        help='Use policy means rather than samples during evaluation')
    parser.add_argument('--eval_max_steps', type=int, default=0)
    parser.add_argument('--measure_reward_scale',
                        type=float,
                        default=None,
                        help='Isaac DQD descriptor reward scale; defaults to env.step_dt')
    parser.add_argument('--episode_length_s', type=float, default=None,
                        help='Override the simulator episode horizon in seconds')
    parser.add_argument('--mjlab_descriptor_mode',
                        choices=['approach_transport', 'motion_effort',
                                 'height_approach', 'progress'],
                        default='approach_transport',
                        help='MJLab QD descriptor pair')
    parser.add_argument('--mjlab_motion_speed_reference', type=float,
                        default=None,
                        help='Arm-speed normalization; defaults to the task velocity threshold')
    parser.add_argument('--mjlab_fixed_goal',
                        type=lambda x: bool(strtobool(x)),
                        default=False,
                        help='Use MJLab fixed-goal mode')
    parser.add_argument('--mjlab_disable_curriculum',
                        type=lambda x: bool(strtobool(x)),
                        default=False,
                        help='Disable MJLab reward curriculum for a stationary objective')
    parser.add_argument('--mjlab_command_resampling_time', type=float,
                        default=None,
                        help='MJLab command period in seconds; must exceed the episode horizon')
    parser.add_argument('--mjlab_terminate_on_success',
                        type=lambda x: bool(strtobool(x)),
                        default=True,
                        help='Terminate MJLab lift episodes on controlled success')
    parser.add_argument('--mjlab_success_bonus', type=float, default=50.0,
                        help='One-time, dt-neutral reward added on MJLab success')
    parser.add_argument('--mjlab_success_max_object_speed', type=float,
                        default=0.15,
                        help='Maximum cube speed in m/s for controlled success')
    parser.add_argument('--mjlab_transport_start_distance', type=float,
                        default=0.03,
                        help='Cube displacement in meters that begins transport')
    parser.add_argument('--mjlab_transport_deviation_reference', type=float,
                        default=0.15,
                        help='Signed transport deviation mapped to descriptor endpoints')
    parser.add_argument('--expdir', type=str, default=None,
                        help='Optional directory for PPO config, evaluation, and checkpoint')
    parser.add_argument('--checkpoint_interval_updates', type=int, default=100,
                        help='Save a resumable PPO checkpoint every N updates; 0 disables')
    parser.add_argument('--max_checkpoints', type=int, default=3,
                        help='Maximum number of periodic PPO checkpoints to retain')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Resume PPO model, optimizer, and counters from a periodic checkpoint')
    parser.add_argument('--initial_actor_checkpoint', type=str, default=None,
                        help='Initialize only the actor (and its normalizer) from a checkpoint')

    # vestigial QD params
    parser.add_argument('--num_dims', type=int)

    args = parser.parse_args()
    cfg = Box(vars(args))
    return cfg


if __name__ == '__main__':
    cfg = parse_args()

    if cfg.seed is None:
        cfg.seed = int(time.time()) + int(os.getpid())

    if cfg.checkpoint_interval_updates < 0:
        raise ValueError('checkpoint_interval_updates cannot be negative')
    if cfg.max_checkpoints < 1:
        raise ValueError('max_checkpoints must be at least 1')
    if cfg.adaptive_kl and cfg.target_kl is None:
        raise ValueError('adaptive_kl requires target_kl')
    if cfg.initial_action_std <= 0:
        raise ValueError('initial_action_std must be positive')
    if cfg.mjlab_success_bonus < 0:
        raise ValueError('mjlab_success_bonus cannot be negative')
    if cfg.mjlab_success_max_object_speed <= 0:
        raise ValueError('mjlab_success_max_object_speed must be positive')
    if cfg.mjlab_transport_start_distance <= 0:
        raise ValueError('mjlab_transport_start_distance must be positive')
    if cfg.mjlab_transport_deviation_reference <= 0:
        raise ValueError(
            'mjlab_transport_deviation_reference must be positive')
    if cfg.resume_checkpoint and cfg.initial_actor_checkpoint:
        raise ValueError('Use either resume_checkpoint or initial_actor_checkpoint')

    vec_env = make_vec_env(cfg)

    if cfg.action_transform is None:
        cfg.action_transform = (
            'tanh' if cfg.env_type == 'isaac' else 'none')
    if cfg.value_bootstrap is None:
        cfg.value_bootstrap = cfg.env_type in ('isaac', 'mjlab')

    cfg.batch_size = int(cfg.env_batch_size * cfg.rollout_length)
    cfg.num_envs = int(cfg.env_batch_size)
    cfg.num_emitters = 1
    cfg.envs_per_model = cfg.num_envs // cfg.num_emitters
    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)

    # Isaac exposes a Dict observation space; Brax exposes a flat Box.
    observation_space = policy_observation_space(vec_env.observation_space)
    cfg.obs_shape = observation_space.shape[1:]
    cfg.action_shape = vec_env.action_space.shape[1:]

    log.debug(
        f'Environment: {cfg.env_name}, obs_shape: {cfg.obs_shape}, action_shape: {cfg.action_shape}'
    )

    outdir = None
    if cfg.expdir:
        outdir = Path(cfg.expdir) / str(cfg.seed)
        outdir.mkdir(parents=True, exist_ok=True)
        cfg.outdir = str(outdir)
        save_cfg(str(outdir), cfg)

    if cfg.use_wandb:
        config_wandb(cfg=cfg,
                     batch_size=cfg.batch_size,
                     total_steps=cfg.total_timesteps,
                     run_name=cfg.wandb_run_name,
                     wandb_group=cfg.wandb_group,
                     wandb_project=cfg.wandb_project)

    alg = PPO(cfg)
    num_updates = cfg.total_timesteps // cfg.batch_size
    start_update = 0
    initial_global_step = 0
    if cfg.initial_actor_checkpoint:
        checkpoint = torch.load(cfg.initial_actor_checkpoint,
                                map_location=alg.device,
                                weights_only=False)
        actor_state = dict(checkpoint.get('actor_state_dict', checkpoint))
        actor = alg._agents[0]
        if ('actor_logstd' in actor_state
                and actor_state['actor_logstd'].shape
                != actor.actor_logstd.shape):
            actor_state['actor_logstd'] = actor_state[
                'actor_logstd'].reshape_as(actor.actor_logstd)
        actor.load_state_dict(actor_state)
        alg.agents = [actor]
        log.info(f'Initialized PPO actor from {cfg.initial_actor_checkpoint}')
    if cfg.resume_checkpoint:
        start_update, initial_global_step = _load_training_checkpoint(
            alg, cfg, cfg.resume_checkpoint)
        log.info(
            f'Resumed PPO from update {start_update}, step {initial_global_step}')

    checkpoint_callback = None
    if outdir is not None and cfg.checkpoint_interval_updates > 0:
        checkpoint_callback = lambda trainer, update, step: (
            _save_periodic_checkpoint(trainer, cfg, outdir, update, step))
    alg.train(
        vec_env,
        num_updates,
        rollout_length=cfg.rollout_length,
        start_update=start_update,
        initial_global_step=initial_global_step,
        checkpoint_callback=checkpoint_callback,
        checkpoint_interval=cfg.checkpoint_interval_updates)
    objectives, measures, metadata = alg.evaluate(
        alg.vec_inference,
        vec_env,
        verbose=True,
        deterministic=cfg.eval_deterministic)
    if outdir is not None:
        final_payload = _checkpoint_payload(
            alg, cfg, num_updates, num_updates * cfg.batch_size)
        final_payload.update({
            'objective': float(objectives[0]),
            'measures': measures[0].tolist(),
            'metadata': metadata[0],
        })
        torch.save(final_payload, outdir / 'final_model.pt')
        with open(outdir / 'evaluation.json', 'w') as f:
            json.dump({
                'objective': float(objectives[0]),
                'measures': measures[0].tolist(),
                'metadata': metadata[0],
            }, f, indent=2, default=_json_default)
    vec_env.close()
    sys.exit(0)
