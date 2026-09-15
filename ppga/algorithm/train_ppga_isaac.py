import argparse
import copy
import csv
import os
import pickle
import signal
import shutil
from pathlib import Path

import numpy as np
import torch
import wandb
from box import Box
from ribs.schedulers import Scheduler

from ppga.envs.factory import make_vec_env, reward_offset
from ppga.envs.qd_env import policy_observation_space
from ppga.models.actor_critic import Actor
from ppga.algorithm.mjlab_archive_utils import success_gated_objectives
from ppga.qd.archives import GridArchive
from ppga.qd.emitters import PPGAEmitter
from ppga.RL.ppo import PPO
from ppga.utils.archive_utils_isaac import (archive_df_to_archive,
                                      load_scheduler_from_checkpoint,
                                      save_heatmap)
from ppga.utils.utilities import (config_wandb, get_checkpoints, log, save_cfg,
                                  set_file_handler)


_STOP_SIGNAL = None


def _request_graceful_stop(signum, _frame):
    """Finish the active QD iteration so its archive can be checkpointed."""
    global _STOP_SIGNAL
    _STOP_SIGNAL = signum
    log.warning(
        f'Received signal {signum}; checkpointing after the current iteration')


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))


def parse_args():
    parser = argparse.ArgumentParser()
    # PPO params
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--env_type', choices=['isaac', 'mjlab'],
                        default='isaac',
                        help='GPU simulator backend')
    parser.add_argument('--seed', type=int, default=0)
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

    # args for isaac
    parser.add_argument('--env_batch_size',
                        default=1,
                        type=int,
                        help='Number of parallel environments to run')

    # ppo hyperparams
    parser.add_argument('--report_interval',
                        type=int,
                        default=5,
                        help='Log objective results every N updates')
    parser.add_argument(
        '--rollout_length',
        type=int,
        default=2048,
        help='the number of steps to run in each environment per policy rollout'
    )
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
    parser.add_argument(
        "--clip_vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help=
        "Toggles whether or not to use a clipped loss for the value function, as per the paper."
    )
    parser.add_argument("--clip_value_coef",
                        type=float,
                        default=0.2,
                        help="value clipping coefficient")
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
        help="Adapt optimizer learning rates around target_kl")
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
        help='Normalize returns across a batch using running mean and stddev')
    parser.add_argument('--value_bootstrap',
                        type=lambda x: bool(strtobool(x)),
                        default=None,
                        help='Bootstrap artificial time limits (default: enabled)')
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
                        help='Policy action bounding; defaults to tanh for Isaac and none for MJLab')
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
                        help='Use policy means rather than samples for archive evaluation')
    parser.add_argument('--eval_max_steps', type=int, default=0)
    parser.add_argument('--eval_common_random_numbers',
                        type=lambda x: bool(strtobool(x)),
                        default=False,
                        help='Evaluate candidate policies on identical reset scenarios')
    parser.add_argument('--eval_common_seed_offset', type=int, default=1000000,
                        help='Seed offset for common-random-number evaluations')
    parser.add_argument('--measure_reward_scale',
                        type=float,
                        default=None,
                        help='DQD descriptor reward scale; defaults to env.step_dt')
    parser.add_argument('--episode_length_s', type=float, default=None,
                        help='Override the simulator episode horizon in seconds')
    parser.add_argument('--mjlab_descriptor_mode',
                        choices=['grip_orientation_elbow_extension',
                                 'grip_orientation_arm_length',
                                 'approach_transport', 'motion_effort',
                                 'height_approach', 'progress'],
                        default='grip_orientation_elbow_extension',
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
    parser.add_argument('--mjlab_approach_deviation_reference', type=float,
                        default=0.05,
                        help='Approach-path deviation mapped to descriptor endpoints')
    parser.add_argument('--mjlab_transport_deviation_reference', type=float,
                        default=0.15,
                        help='Peak signed transport deviation mapped to descriptor endpoints')
    parser.add_argument('--mjlab_arm_length_min', type=float, default=0.20,
                        help='Arm length in meters mapped to descriptor zero')
    parser.add_argument('--mjlab_arm_length_max', type=float, default=0.50,
                        help='Arm length in meters mapped to descriptor one')
    parser.add_argument('--mjlab_grip_tilt_min_degrees', type=float,
                        default=15.0,
                        help='Grip tilt in degrees mapped to descriptor zero')
    parser.add_argument('--mjlab_grip_tilt_max_degrees', type=float,
                        default=45.0,
                        help='Grip tilt in degrees mapped to descriptor one')
    parser.add_argument('--mjlab_elbow_extension_min_degrees', type=float,
                        default=45.0,
                        help='Elbow angle in degrees mapped to descriptor zero')
    parser.add_argument('--mjlab_elbow_extension_max_degrees', type=float,
                        default=65.0,
                        help='Elbow angle in degrees mapped to descriptor one')

    # QD Params
    parser.add_argument("--num_emitters",
                        type=int,
                        default=1,
                        help="Number of parallel"
                        " CMA-ES instances exploring the archive")
    parser.add_argument('--grid_size',
                        type=int,
                        required=True,
                        help='Number of cells per archive dimension')
    parser.add_argument("--num_dims",
                        type=int,
                        required=True,
                        help="Dimensionality of measures")
    parser.add_argument(
        "--popsize",
        type=int,
        required=True,
        help=
        "Branching factor for each step of MEGA i.e. the number of branching solutions from the current solution point"
    )
    parser.add_argument(
        '--log_arch_freq',
        type=int,
        default=10,
        help='Frequency in num iterations at which we checkpoint the archive')
    parser.add_argument(
        '--save_scheduler',
        type=lambda x: bool(strtobool(x)),
        default=True,
        help=
        'Choose whether or not to save the scheduler during checkpointing. If the archive is too big,'
        'it may be impractical to save both the scheduler and the archive_df. However, you cannot later restart from '
        'a scheduler checkpoint and instead will have to restart from an archive_df checkpoint, which may impact the performance of the run.'
    )
    parser.add_argument(
        '--load_scheduler_from_cp',
        type=str,
        default=None,
        help='Load an existing QD scheduler from a checkpoint path')
    parser.add_argument(
        '--load_archive_from_cp',
        type=str,
        default=None,
        help=
        'Load an existing archive from a checkpoint path. This can be used as an alternative to loading the scheduler if save_scheduler'
        'was disabled and only the archive df checkpoint is available. However, this can affect the performance of the run. Cannot be used together with save_scheduler'
    )
    parser.add_argument('--initial_actor_checkpoint',
                        type=str,
                        default=None,
                        help='Initialize a new QD run from train_ppo final_model.pt')
    parser.add_argument(
        '--total_iterations',
        type=int,
        default=100,
        help='Number of iterations to run the entire dqd-rl loop')
    parser.add_argument(
        '--dqd_algorithm',
        type=str,
        choices=['cma_mega_adam', 'cma_maega'],
        help='Which DQD algorithm should be running in the outer loop')
    parser.add_argument('--expdir',
                        type=str,
                        help='Experiment results directory')
    parser.add_argument(
        '--save_heatmaps',
        type=lambda x: bool(strtobool(x)),
        default=True,
        help=
        'Save the archive heatmaps. Only applies to archives with <= 2 measures'
    )
    parser.add_argument('--heatmap_freq',
                        type=int,
                        default=10,
                        help='Save a heatmap every N iterations')
    parser.add_argument(
        '--use_surrogate_archive',
        type=lambda x: bool(strtobool(x)),
        default=False,
        help=
        "Use a surrogate archive at a higher resolution to get a better gradient signal for DQD"
    )
    parser.add_argument(
        '--sigma0',
        type=float,
        default=1.0,
        help=
        'Initial standard deviation parameter for the covariance matrix used in NES methods'
    )
    parser.add_argument(
        '--xnes_center_init',
        choices=['random', 'zero'],
        default='random',
        help=('Initialize the XNES gradient-coefficient mean randomly within '
              'its legacy bounds or explicitly at zero'))
    parser.add_argument('--restart_rule',
                        type=str,
                        choices=['basic', 'no_improvement'])
    parser.add_argument(
        '--calc_gradient_iters',
        type=int,
        help=
        'Number of iters to run PPO when estimating the objective-measure gradients (N1)'
    )
    parser.add_argument(
        '--move_mean_iters',
        type=int,
        help=
        'Number of iterations to run PPO when moving the mean solution point (N2)'
    )
    parser.add_argument('--archive_lr',
                        type=float,
                        help='Archive learning rate for MAEGA')
    parser.add_argument(
        '--threshold_min',
        type=float,
        default=0.0,
        help='Min objective threshold for adding new solutions to the archive')
    parser.add_argument(
        '--archive_min_success_rate',
        type=float,
        default=0.0,
        help='Reject policies below this episode success rate from the archive')
    parser.add_argument(
        '--take_archive_snapshots',
        type=lambda x: bool(strtobool(x)),
        default=False,
        help=
        'Log the objective scores in every cell in the archive every log_freq iterations. Useful for pretty visualizations'
    )
    parser.add_argument(
        '--adaptive_stddev',
        type=lambda x: bool(strtobool(x)),
        default=True,
        help=
        'If False, the log stddev parameter in the actor will be reset on each QD iteration. Can potentially help exploration but may lose performance'
    )

    args = parser.parse_args()
    return Box(vars(args))


def save_scheduler(scheduler, save_path):
    # cannot pickle generator objects so need to remove it temporarily
    gen = scheduler.emitters[0].opt.problem._generator
    scheduler.emitters[0].opt.problem._generator = None
    # save the scheduler for checkpointing
    with open(save_path, 'wb') as f:
        pickle.dump(scheduler, f, protocol=pickle.HIGHEST_PROTOCOL)
    scheduler.emitters[0].opt.problem._generator = gen


def create_scheduler(cfg: Box,
                     archive_learning_rate: float = None,
                     use_result_archive: bool = True,
                     initial_sol: np.ndarray = None):
    '''Creates a scheduler that uses the ppga emitter
        Args:
        cfg (Box): config file
        archive_learning_rate (float): Learning rate of archive.
        use_result_archive (bool): Whether to use a separate archive to store
            the results.
        initial_sol: initial solution (agent)
    Returns:
        ribs.schedulers.Scheduler: A ribs scheduler for running the algorithm.
    """
    '''
    num_emitters = 1
    obs_shape, action_shape = cfg.obs_shape, cfg.action_shape
    action_dim, obs_dim = np.prod(action_shape), np.prod(obs_shape)
    log.debug(f'Environment {cfg.env_name}, {action_dim=}, {obs_dim=}')
    batch_size = cfg.popsize
    # empirically calculated for brax envs to make the qd-score strictly positive
    cur_reward_offset = reward_offset.get(cfg.env_name, 0.0)

    if initial_sol is None:
        initial_agent = Actor(obs_shape, action_shape, cfg.normalize_obs,
                              cfg.normalize_returns, cfg.action_transform,
                              cfg.action_std_parameterization,
                              cfg.initial_action_std,
                              hidden_dims=cfg.actor_hidden_dims)
        initial_sol = initial_agent.serialize()
    solution_dim = len(initial_sol)
    mode = 'batch'
    # threshold for adding solutions to the archive
    threshold_min = -np.inf

    bounds = [(0.0, 1.0)] * cfg.num_dims
    archive_dims = [cfg.grid_size] * cfg.num_dims

    if cfg.dqd_algorithm == 'cma_maega':
        threshold_min = cfg.threshold_min

    if archive_learning_rate is None:
        if cfg.dqd_algorithm == 'cma_maega':
            archive_learning_rate = cfg.archive_lr
        else:
            archive_learning_rate = 1.0

    archive, result_archive = None, None
    if cfg.load_archive_from_cp is not None and cfg.load_scheduler_from_cp is None:
        log.info('Loading an existing archive dataframe...')
        with open(cfg.load_archive_from_cp, 'rb') as f:
            archive_df = pickle.load(f)
        archive = archive_df_to_archive(
            archive_df,
            solution_dim=solution_dim,
            dims=archive_dims,
            ranges=bounds,
            learning_rate=archive_learning_rate,
            threshold_min=threshold_min,
            seed=cfg.seed,
            reward_offset=cur_reward_offset,
            extra_fields={
                "metadata": ((), object),
            },
        )

        if use_result_archive:
            result_archive = archive_df_to_archive(
                archive_df,
                solution_dim=solution_dim,
                dims=archive_dims,
                ranges=bounds,
                learning_rate=1.0,
                threshold_min=threshold_min,
                seed=cfg.seed,
                reward_offset=cur_reward_offset,
                extra_fields={
                    "metadata": ((), object),
                },
            )
    else:
        archive = GridArchive(
            solution_dim=solution_dim,
            dims=archive_dims,
            ranges=bounds,
            learning_rate=archive_learning_rate,
            threshold_min=threshold_min,
            seed=cfg.seed,
            reward_offset=cur_reward_offset,
            extra_fields={
                "metadata": ((), object),
            },
        )

        if use_result_archive:
            result_archive = GridArchive(
                solution_dim=solution_dim,
                dims=archive_dims,
                ranges=bounds,
                learning_rate=1.0,
                threshold_min=threshold_min,
                seed=cfg.seed,
                reward_offset=cur_reward_offset,
                extra_fields={
                    "metadata": ((), object),
                },
            )

    ppo = PPO(cfg)

    # Create emitters. Each emitter needs a different seed, so that they do not
    # all do the same thing.
    emitter_seeds = [None] * num_emitters if cfg.seed is None else np.arange(
        cfg.seed, cfg.seed + num_emitters)

    if cfg.dqd_algorithm == 'cma_mega_adam':
        # Note that only one emitter is used for cma_mega_adam. This is to be
        # consistent with Fontaine 2021 <https://arxiv.org/abs/2106.03894>.
        emitters = [
            PPGAEmitter(
                ppo,
                archive,
                x0=initial_sol,
                sigma0=cfg.sigma0,
                batch_size=batch_size,
                seed=emitter_seeds[0],
                use_wandb=cfg.use_wandb,
                normalize_obs=cfg.normalize_obs,
                normalize_returns=cfg.normalize_returns,
                xnes_center_init=cfg.xnes_center_init,
            )
        ]
    else:
        # cma_maega
        emitters = [
            PPGAEmitter(
                ppo,
                archive,
                x0=initial_sol,
                sigma0=cfg.sigma0,
                batch_size=batch_size,
                ranker='imp',
                restart_rule=cfg.restart_rule,
                bounds=None,
                seed=emitter_seeds[0],
                use_wandb=cfg.use_wandb,
                normalize_obs=cfg.normalize_obs,
                normalize_returns=cfg.normalize_returns,
                xnes_center_init=cfg.xnes_center_init,
            )
        ]

    log.debug(
        f"Created Scheduler for {cfg.dqd_algorithm} with an archive learning rate of {archive_learning_rate}, "
        f"and add mode {mode}, using solution dim {solution_dim} and archive "
        f"dims {archive_dims}. Min threshold is {threshold_min}. Restart rule is {cfg.restart_rule}"
    )

    return Scheduler(
        archive,
        emitters,
        result_archive=result_archive,
        add_mode=mode,
    )


def train_ppga(cfg: Box, vec_env):
    # setup logging
    exp_dir = Path(cfg.outdir)
    logdir = exp_dir.joinpath(Path('logs'))
    if not logdir.is_dir():
        logdir.mkdir()
    set_file_handler(logdir)
    save_cfg(str(exp_dir), cfg)

    # checkpointing
    cp_dir = exp_dir.joinpath(Path('checkpoints'))
    if not cp_dir.is_dir():
        cp_dir.mkdir()

    # (optional) save 2d archive heatmaps
    heatmap_dir = exp_dir.joinpath(Path('heatmaps'))
    if cfg.save_heatmaps and not heatmap_dir.is_dir():
        heatmap_dir.mkdir()

    # path to summary file
    summary_filename = os.path.join(str(exp_dir), 'summary.csv')
    if os.path.exists(summary_filename):
        os.remove(summary_filename)
    with open(summary_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Iteration', 'QD-Score', 'Coverage', 'Maximum', 'Average',
            'Mean Success Rate', 'Max Success Rate', 'Max Object Height',
            'Mean Trajectory Length'
        ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_result_archive = cfg.dqd_algorithm == 'cma_maega'

    if cfg.load_scheduler_from_cp:
        log.info("Loading an existing scheduler!")
        scheduler = load_scheduler_from_checkpoint(cfg.load_scheduler_from_cp,
                                                   cfg.seed, device)
    else:
        initial_sol = None
        initial_obs_normalizer = None
        initial_return_normalizer = None
        if cfg.initial_actor_checkpoint:
            checkpoint = torch.load(cfg.initial_actor_checkpoint,
                                    map_location=device,
                                    weights_only=False)
            actor_state = checkpoint.get('actor_state_dict', checkpoint)
            initial_actor = Actor(
                cfg.obs_shape, cfg.action_shape, cfg.normalize_obs,
                cfg.normalize_returns, cfg.action_transform,
                cfg.action_std_parameterization,
                hidden_dims=cfg.actor_hidden_dims).to(device)
            actor_state = dict(actor_state)
            if ('actor_logstd' in actor_state and actor_state['actor_logstd'].shape
                    != initial_actor.actor_logstd.shape):
                actor_state['actor_logstd'] = actor_state[
                    'actor_logstd'].reshape_as(initial_actor.actor_logstd)
            initial_actor.load_state_dict(actor_state)
            initial_sol = initial_actor.serialize()
            if cfg.normalize_obs:
                initial_obs_normalizer = copy.deepcopy(
                    initial_actor.obs_normalizer)
            if cfg.normalize_returns:
                initial_return_normalizer = copy.deepcopy(
                    initial_actor.return_normalizer)
            log.info(
                f'Initialized QD mean from {cfg.initial_actor_checkpoint}')
        scheduler = create_scheduler(
            cfg,
            use_result_archive=use_result_archive,
            initial_sol=initial_sol)
        if initial_obs_normalizer is not None:
            scheduler.emitters[
                0].mean_agent_obs_normalizer = initial_obs_normalizer
        if initial_return_normalizer is not None:
            scheduler.emitters[
                0].mean_agent_return_normalizer = initial_return_normalizer

    # (optional) take 3d archive snapshots and use to construct a gif
    archive_snapshot_filename = os.path.join(str(logdir),
                                             'archive_snapshots.csv')
    if cfg.take_archive_snapshots:
        if os.path.exists(archive_snapshot_filename):
            os.remove(archive_snapshot_filename)
        num_cells = np.prod(scheduler.archive.dims)
        with open(archive_snapshot_filename, 'w') as archive_snapshot_file:
            row = ['Iteration'] + [f'cell_{i}' for i in range(num_cells)]
            writer = csv.writer(archive_snapshot_file)
            writer.writerow(row)

    result_archive = scheduler.result_archive
    best = 0.0

    obs_shape = cfg.obs_shape
    action_shape = cfg.action_shape

    ppo = scheduler.emitters[0].ppo

    # save the initial heatmap
    if cfg.save_heatmaps and cfg.num_dims <= 2:
        save_heatmap(result_archive,
                     os.path.join(str(heatmap_dir), f'heatmap_{0:05d}.png'))

    log_freq = 1
    log_arch_freq = cfg.log_arch_freq

    starting_iter = scheduler.emitters[
        0].itrs  # if loading a checkpoint, this will be > 0
    itrs = cfg.total_iterations
    # main loop
    for itr in range(starting_iter, itrs):
        # Current solution point. Returns a single solution per emitter.
        solution_batch = scheduler.ask_dqd()
        mean_agent = Actor(obs_shape, action_shape, cfg.normalize_obs,
                           cfg.normalize_returns,
                           cfg.action_transform,
                           cfg.action_std_parameterization,
                           hidden_dims=cfg.actor_hidden_dims).deserialize(
                               solution_batch.flatten()).to(device)
        if not cfg.adaptive_stddev:
            initial_std = (
                cfg.initial_action_std
                if cfg.action_std_parameterization == 'direct' else
                np.log(cfg.initial_action_std))
            mean_agent.actor_logstd = torch.nn.Parameter(
                torch.full((1, np.prod(cfg.action_shape)), initial_std,
                           device=device))

        if cfg.normalize_obs:
            if scheduler.emitters[0].mean_agent_obs_normalizer is not None:
                mean_agent.obs_normalizer = scheduler.emitters[
                    0].mean_agent_obs_normalizer
        if cfg.normalize_returns:
            if scheduler.emitters[0].mean_agent_return_normalizer is not None:
                mean_agent.return_normalizer = scheduler.emitters[
                    0].mean_agent_return_normalizer

        ppo.agents = [mean_agent]
        # calculate gradients of f and m
        objs, measures, jacobian, metadata = ppo.train(
            vec_env=vec_env,
            num_updates=cfg.calc_gradient_iters,
            rollout_length=cfg.rollout_length,
            calculate_dqd_gradients=True,
            negative_measure_gradients=False)

        emitter_loc = tuple(measures[0, :2])
        best = max(best, max(objs))
        archive_objs = success_gated_objectives(
            objs, metadata, cfg.archive_min_success_rate, cfg.threshold_min)
        scheduler.tell_dqd(
            archive_objs, measures, jacobian, metadata=metadata)

        branched_sols = scheduler.ask()
        branched_agents = [
            Actor(obs_shape, action_shape, cfg.normalize_obs,
                  cfg.normalize_returns,
                  cfg.action_transform,
                  cfg.action_std_parameterization,
                  hidden_dims=cfg.actor_hidden_dims).deserialize(sol).to(device)
            for sol in branched_sols
        ]
        # Do not overwrite actor_logstd here: it is part of each serialized
        # solution. Overwriting it makes the evaluated policy differ from the
        # solution given to scheduler.tell().
        ppo.agents = branched_agents

        eval_obs_normalizer = mean_agent.obs_normalizer if cfg.normalize_obs else None
        eval_rew_normalizer = mean_agent.return_normalizer if cfg.normalize_returns else None
        objs, measures, metadata = ppo.evaluate(
            ppo.vec_inference,
            vec_env,
            verbose=True,
            obs_normalizer=eval_obs_normalizer,
            return_normalizer=eval_rew_normalizer,
            deterministic=cfg.eval_deterministic)

        if cfg.weight_decay:
            reg_loss = cfg.weight_decay * np.array([
                np.linalg.norm(sol) for sol in branched_sols
            ]).reshape(objs.shape)
            objs -= reg_loss

        best = max(best, max(objs))
        archive_objs = success_gated_objectives(
            objs, metadata, cfg.archive_min_success_rate, cfg.threshold_min)
        scheduler.tell(archive_objs, measures, metadata=metadata)
        if scheduler.emitters[0].last_stop_status:
            log.debug('Emitter restarted. Changing the mean agent...')
            mean_agent = Actor(
                obs_shape, action_shape, cfg.normalize_obs,
                cfg.normalize_returns,
                cfg.action_transform,
                cfg.action_std_parameterization,
                hidden_dims=cfg.actor_hidden_dims).deserialize(
                    scheduler.emitters[0].theta).to(device)
            if cfg.normalize_obs:
                mean_agent.obs_normalizer = scheduler.emitters[
                    0].mean_agent_obs_normalizer
            if cfg.normalize_returns:
                mean_agent.return_normalizer = scheduler.emitters[
                    0].mean_agent_return_normalizer

        mean_grad_coeffs = np.expand_dims(
            scheduler.emitters[0].opt.mu, axis=0).astype(np.float32)
        log.info(f'New mean coefficients: {mean_grad_coeffs}')

        ppo.grad_coeffs = mean_grad_coeffs
        ppo.agents = [mean_agent]
        log.info('Moving the mean solution point...')
        ppo.train(vec_env=vec_env,
                  num_updates=cfg.move_mean_iters,
                  rollout_length=cfg.rollout_length,
                  calculate_dqd_gradients=False,
                  move_mean_agent=True)

        trained_mean_agent = ppo.agents[0]
        scheduler.emitters[0].update_theta(trained_mean_agent.serialize())
        if cfg.normalize_obs:
            scheduler.emitters[
                0].mean_agent_obs_normalizer = trained_mean_agent.obs_normalizer
        if cfg.normalize_returns:
            scheduler.emitters[
                0].mean_agent_return_normalizer = trained_mean_agent.return_normalizer

        completed_itr = itr + 1
        log.debug(
            f'{completed_itr=}, {itrs=}, '
            f'Progress: {(100.0 * (completed_itr / itrs)):.2f}%')

        if (cfg.save_heatmaps and cfg.num_dims <= 2
                and (completed_itr % cfg.heatmap_freq == 0
                     or completed_itr == itrs)):
            save_heatmap(result_archive,
                         os.path.join(str(heatmap_dir), f'heatmap_{completed_itr:05d}.png'),
                         emitter_loc=emitter_loc,
                         forces=None)

        final_itr = completed_itr == itrs or _STOP_SIGNAL is not None
        if completed_itr % log_arch_freq == 0 or final_itr:
            final_cp_dir = os.path.join(cp_dir, f'cp_{completed_itr:08d}')
            os.makedirs(final_cp_dir, exist_ok=True)
            result_archive.data(return_type='pandas').to_pickle(
                os.path.join(final_cp_dir, f'archive_df_{completed_itr:08d}.pkl'))
            if cfg.save_scheduler:
                save_scheduler(
                    scheduler,
                    os.path.join(final_cp_dir, f'scheduler_{completed_itr:08d}.pkl'))

        while len(get_checkpoints(str(cp_dir))) > 2:
            oldest_checkpoint = get_checkpoints(str(cp_dir))[0]
            if os.path.exists(oldest_checkpoint):
                log.info(f'Removing checkpoint {oldest_checkpoint}')
                shutil.rmtree(oldest_checkpoint)

        if completed_itr % log_freq == 0 or final_itr:
            elite_metadata = []
            for elite in result_archive:
                metadata = (elite.get('metadata')
                            if isinstance(elite, dict)
                            else getattr(elite, 'metadata', None))
                if isinstance(metadata, dict):
                    elite_metadata.append(metadata)
            success_rates = [
                data['episode_success_rate'] for data in elite_metadata
                if 'episode_success_rate' in data
            ]
            object_heights = [
                data['max_object_height'] for data in elite_metadata
                if 'max_object_height' in data
            ]
            trajectory_lengths = [
                data['traj_length'] for data in elite_metadata
                if 'traj_length' in data
            ]
            with open(summary_filename, 'a') as summary_file:
                csv.writer(summary_file).writerow([
                    completed_itr, result_archive.stats.qd_score,
                    result_archive.stats.coverage, result_archive.stats.obj_max,
                    result_archive.stats.obj_mean,
                    np.mean(success_rates) if success_rates else np.nan,
                    np.max(success_rates) if success_rates else np.nan,
                    np.max(object_heights) if object_heights else np.nan,
                    (np.mean(trajectory_lengths)
                     if trajectory_lengths else np.nan),
                ])

        if (completed_itr % log_freq == 0 or final_itr) and cfg.take_archive_snapshots:
            with open(archive_snapshot_filename, 'a') as archive_snapshot_file:
                num_cells = np.prod(scheduler.result_archive.dims)
                elite_scores = [0 for _ in range(num_cells)]
                for elite in scheduler.result_archive:
                    elite_scores[elite.index] = elite.objective
                csv.writer(archive_snapshot_file).writerow([completed_itr] + elite_scores)

        if cfg.use_wandb:
            with torch.no_grad():
                normA = torch.linalg.norm(
                    scheduler.emitters[0].opt.A).cpu().numpy().item()
            qd_metrics = {
                'QD/QD Score': scheduler.result_archive.offset_qd_score,
                'QD/average performance': result_archive.stats.obj_mean,
                'QD/coverage (%)': result_archive.stats.coverage * 100.0,
                'QD/best score': result_archive.stats.obj_max,
                'QD/iteration': completed_itr,
                'QD/restarts': scheduler.emitters[0].restarts,
                'QD/mean_coeff_obj': mean_grad_coeffs[0][0],
                'XNES/norm_A': normA,
            }
            for i in range(1, cfg.num_dims + 1):
                qd_metrics[f'QD/mean_coeff_measure{i}'] = mean_grad_coeffs[0][i]
            if success_rates:
                qd_metrics['Task/archive_mean_success_rate'] = np.mean(
                    success_rates)
                qd_metrics['Task/archive_max_success_rate'] = np.max(
                    success_rates)
            if object_heights:
                qd_metrics['Task/archive_max_object_height'] = np.max(
                    object_heights)
            wandb.log(qd_metrics)

        if _STOP_SIGNAL is not None:
            log.info(
                f'Stopping cleanly after iteration {completed_itr}; '
                'the final archive checkpoint and summary row were saved')
            break


def main():
    signal.signal(signal.SIGTERM, _request_graceful_stop)
    cfg = parse_args()
    if cfg.total_iterations < 1:
        raise ValueError('total_iterations must be at least 1')
    if cfg.save_heatmaps and cfg.heatmap_freq < 1:
        raise ValueError('heatmap_freq must be at least 1 when saving heatmaps')
    if cfg.adaptive_kl and cfg.target_kl is None:
        raise ValueError('adaptive_kl requires target_kl')
    if cfg.initial_action_std <= 0:
        raise ValueError('initial_action_std must be positive')
    if not 0.0 <= cfg.archive_min_success_rate <= 1.0:
        raise ValueError('archive_min_success_rate must be in [0, 1]')
    if (cfg.mjlab_motion_speed_reference is not None
            and cfg.mjlab_motion_speed_reference <= 0):
        raise ValueError('mjlab_motion_speed_reference must be positive')
    if cfg.mjlab_success_bonus < 0:
        raise ValueError('mjlab_success_bonus cannot be negative')
    if cfg.mjlab_success_max_object_speed <= 0:
        raise ValueError('mjlab_success_max_object_speed must be positive')
    if cfg.mjlab_transport_start_distance <= 0:
        raise ValueError('mjlab_transport_start_distance must be positive')
    if cfg.mjlab_approach_deviation_reference <= 0:
        raise ValueError(
            'mjlab_approach_deviation_reference must be positive')
    if cfg.mjlab_transport_deviation_reference <= 0:
        raise ValueError(
            'mjlab_transport_deviation_reference must be positive')
    if cfg.mjlab_arm_length_min < 0:
        raise ValueError('mjlab_arm_length_min cannot be negative')
    if cfg.mjlab_arm_length_max <= cfg.mjlab_arm_length_min:
        raise ValueError(
            'mjlab_arm_length_max must exceed mjlab_arm_length_min')
    if cfg.mjlab_grip_tilt_max_degrees <= cfg.mjlab_grip_tilt_min_degrees:
        raise ValueError(
            'mjlab_grip_tilt_max_degrees must exceed its minimum')
    if (cfg.mjlab_elbow_extension_max_degrees
            <= cfg.mjlab_elbow_extension_min_degrees):
        raise ValueError(
            'mjlab_elbow_extension_max_degrees must exceed its minimum')
    if cfg.eval_common_seed_offset < 0:
        raise ValueError('eval_common_seed_offset cannot be negative')
    cfg.num_emitters = 1
    vec_env = make_vec_env(cfg)
    if cfg.action_transform is None:
        cfg.action_transform = 'tanh' if cfg.env_type == 'isaac' else 'none'
    if cfg.value_bootstrap is None:
        cfg.value_bootstrap = True
    cfg.batch_size = int(cfg.env_batch_size * cfg.rollout_length)
    cfg.num_envs = int(cfg.env_batch_size)

    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)

    if cfg.env_batch_size % (cfg.num_dims + 1) != 0:
        raise ValueError('env_batch_size must be divisible by num_dims + 1 for DQD gradients')
    if cfg.env_batch_size % cfg.popsize != 0:
        raise ValueError('env_batch_size must be divisible by popsize for branch evaluation')

    # [1:] ignores the batch dimension.
    cfg.obs_shape = policy_observation_space(
        vec_env.observation_space).shape[1:]
    cfg.action_shape = vec_env.action_space.shape[1:]

    if cfg.use_wandb:
        config_wandb(batch_size=cfg.batch_size,
                     total_iters=cfg.total_iterations,
                     run_name=cfg.wandb_run_name,
                     wandb_project=cfg.wandb_project,
                     wandb_group=cfg.wandb_group,
                     cfg=cfg)
    outdir = os.path.join(cfg.expdir, str(cfg.seed))
    cfg.outdir = outdir
    # assert not os.path.exists(outdir) or cfg.load_scheduler_from_cp is not None or cfg.load_archive_from_cp is not None, \
        # f"Warning: experiment dir {outdir} exists. Danger of overwriting previous run"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not cfg.save_scheduler:
        log.warning(
            'Warning. You have set save scheduler to false. Only the archive dataframe will be saved in each '
            'checkpoint. If you plan to restart this experiment from a checkpoint or wish to have the added '
            'safety of recovering from a potential crash, it is recommended that you enable save_scheduler.'
        )
    train_ppga(cfg, vec_env)


if __name__ == '__main__':
    main()
