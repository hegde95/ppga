"""Collects a dataset of trajectories from rolling out a PPO agent."""
import argparse
import os
import sys
import time
from distutils.util import strtobool

import h5py
import numpy as np
import torch
from box import Box

from ppga.models.actor_critic import Actor
from ppga.RL.ppo import PPO
from ppga.utils.utilities import config_wandb, log


def get_reset_data():
    """Adapted from D4RL
    https://github.com/Farama-Foundation/D4RL/blob/71a9549f2091accff93eeff68f1f3ab2c0e0a288/scripts/generation/mujoco/collect_data.py
    """
    data = dict(observations=[],
                next_observations=[],
                actions=[],
                rewards=[],
                terminals=[],
                timeouts=[],
                logprobs=[],
                qpos=[],
                qvel=[])
    return data


def rollout(actor: Actor, env, max_path: int, num_data: int,
            normalize_obs: bool):
    """Adapted from D4RL
    https://github.com/Farama-Foundation/D4RL/blob/71a9549f2091accff93eeff68f1f3ab2c0e0a288/scripts/generation/mujoco/collect_data.py
    """
    data = get_reset_data()
    traj_data = get_reset_data()

    _returns = 0
    t = 0
    done = False
    s = env.reset()

    if normalize_obs:
        obs_mean = actor.obs_normalizer.obs_rms.mean
        obs_var = actor.obs_normalizer.obs_rms.var

    while len(data['rewards']) < num_data:

        # The dataset should only see the regular observations (i.e., `s`) while
        # the actor should see normalized observations (i.e., `obs`)
        if normalize_obs:
            obs = (s - obs_mean) / torch.sqrt(obs_var + 1e-8)
        else:
            obs = s

        # Samples the action.
        #  a, logprob, _ = actor.get_action_2(s)

        # Just use the mean action.
        a = actor(obs)

        #  torch_s = ptu.from_numpy(np.expand_dims(s, axis=0))
        #  distr = policy.forward(torch_s)
        #  a = distr.sample()
        #  logprob = distr.log_prob(a)
        #  a = ptu.get_numpy(a).squeeze()

        #mujoco only
        #  qpos, qvel = env.sim.data.qpos.ravel().copy(), env.sim.data.qvel.ravel(
        #  ).copy()

        ns, rew, done, infos = env.step(a)
        #  try:
        #      ns, rew, done, infos = env.step(a)
        #  except:
        #      print('lost connection')
        #      env.close()
        #      env = gym.make(env_name)
        #      s = env.reset()
        #      traj_data = get_reset_data()
        #      t = 0
        #      _returns = 0
        #      continue

        _returns += rew

        t += 1
        timeout = False
        terminal = False
        if t == max_path:
            timeout = True
        elif done:
            terminal = True

        traj_data['observations'].append(s.cpu().detach().numpy())
        traj_data['actions'].append(a.cpu().detach().numpy())
        traj_data['next_observations'].append(ns.cpu().detach().numpy())
        traj_data['rewards'].append(rew.cpu().detach().numpy())
        traj_data['terminals'].append(terminal)
        traj_data['timeouts'].append(timeout)
        #  traj_data['logprobs'].append(logprob)
        #  traj_data['qpos'].append(qpos)
        #  traj_data['qvel'].append(qvel)

        s = ns
        if terminal or timeout:
            for k in data:
                data[k].extend(traj_data[k])

            print('Finished trajectory. Len=%d, Returns=%f. Progress:%d/%d' %
                  (t, _returns, len(data['rewards']), num_data))

            traj_data = get_reset_data()
            s = env.reset()
            t = 0
            _returns = 0

    new_data = dict(
        observations=np.array(data['observations']).astype(np.float32),
        actions=np.array(data['actions']).astype(np.float32),
        next_observations=np.array(data['next_observations']).astype(
            np.float32),
        rewards=np.array(data['rewards']).astype(np.float32),
        terminals=np.array(data['terminals']).astype(bool),
        timeouts=np.array(data['timeouts']).astype(bool))

    #  new_data['infos/action_log_probs'] = np.array(data['logprobs']).astype(
    #      np.float32)
    #  new_data['infos/qpos'] = np.array(data['qpos']).astype(np.float32)
    #  new_data['infos/qvel'] = np.array(data['qvel']).astype(np.float32)

    for k in new_data:
        new_data[k] = new_data[k][:num_data]
    return new_data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint',
                        type=str,
                        help="Path to PPO checkpoint.")
    parser.add_argument('--output_file',
                        type=str,
                        help="Path to file to save data.")
    parser.add_argument('--max_path',
                        type=int,
                        default=1000,
                        help="Max episode length.")
    parser.add_argument('--num_data',
                        type=int,
                        help="Number of timesteps to generate.")

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
    parser.add_argument(
        '--env_type',
        type=str,
        choices=['brax', 'isaac'],
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
                        default=False,
                        help='Use bootstrap value estimates')

    parser.add_argument('--weight_decay',
                        type=float,
                        default=None,
                        help='Apply L2 weight regularization to the NNs')

    parser.add_argument('--clip_obs_rew',
                        type=lambda x: bool(strtobool(x)),
                        default=False,
                        help='Clip obs and rewards b/w -10 and 10')

    # vestigial QD params
    parser.add_argument('--num_dims', type=int)

    args = parser.parse_args()
    cfg = Box(vars(args))
    return cfg


def main():
    ## Configuration parsing ##

    cfg = parse_args()

    if cfg.seed is None:
        cfg.seed = int(time.time()) + int(os.getpid())

    if cfg.env_type == 'brax':
        from ppga.envs.brax_custom.brax_env import make_single_env_brax
        single_env = make_single_env_brax(cfg)
    else:
        raise NotImplementedError(
            f'{cfg.env_type} is undefined for "env_type"')

    cfg.batch_size = int(cfg.env_batch_size * cfg.rollout_length)
    cfg.num_envs = int(cfg.env_batch_size)
    cfg.num_emitters = 1
    cfg.envs_per_model = cfg.num_envs // cfg.num_emitters
    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
    cfg.obs_shape = single_env.observation_space.shape
    cfg.action_shape = single_env.action_space.shape

    log.debug(
        f'Environment: {cfg.env_name}, obs_shape: {cfg.obs_shape}, action_shape: {cfg.action_shape}'
    )

    if cfg.use_wandb:
        config_wandb(cfg=cfg,
                     batch_size=cfg.batch_size,
                     total_steps=cfg.total_timesteps,
                     run_name=cfg.wandb_run_name,
                     wandb_group=cfg.wandb_group,
                     wandb_project=cfg.wandb_project)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Could also be a Vectorized Actor if we want multiple experience
    # simultaneously.
    model = Actor(cfg.obs_shape, cfg.action_shape, cfg.normalize_obs,
                  cfg.normalize_returns).to(device)

    # See utilities.save_checkpoint for the structure; it's a dict with the
    # model and optimizer parameters.
    checkpoint = torch.load(cfg.checkpoint, map_location=device)
    checkpoint["model_state_dict"]["actor_logstd"] = \
        checkpoint["model_state_dict"]["actor_logstd"][None]
    model.load_state_dict(checkpoint["model_state_dict"])

    data = rollout(
        model,
        single_env,
        max_path=cfg.max_path,  # Episode length.
        num_data=cfg.num_data,
        normalize_obs=cfg.normalize_obs,
    )

    hfile = h5py.File(cfg.output_file, 'w')
    for k in data:
        hfile.create_dataset(k, data=data[k], compression='gzip')

    hfile['metadata/algorithm'] = np.string_('PPO')
    #  hfile['metadata/iteration'] = np.array([get_pkl_itr(args.pklfile)],
    #                                         dtype=np.int32)[0]
    #  hfile['metadata/policy/nonlinearity'] = np.string_('relu')
    #  hfile['metadata/policy/output_distribution'] = np.string_(
    #      'tanh_gaussian')
    #  for k, v in get_policy_wts(policy).items():
    #      hfile['metadata/policy/' + k] = v
    hfile.close()


if __name__ == '__main__':
    main()
