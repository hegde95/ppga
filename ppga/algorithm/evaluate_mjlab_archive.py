"""Reevaluate an MJLab archive under its saved task configuration."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from box import Box

from ppga.RL.ppo import PPO
from ppga.envs.factory import make_vec_env
from ppga.envs.qd_env import policy_observation_space
from ppga.models.actor_critic import Actor
from ppga.models.vectorized import VectorizedActor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--env_batch_size', type=int, default=384)
    parser.add_argument('--policies_per_batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20260907)
    parser.add_argument('--descriptor_mode',
                        choices=['height_approach', 'progress'],
                        default=None,
                        help='Optional override; defaults to the saved config')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.env_batch_size % args.policies_per_batch != 0:
        raise ValueError(
            'env_batch_size must be divisible by policies_per_batch')

    cfg = Box(json.loads(args.config.read_text()))
    cfg.env_type = 'mjlab'
    cfg.env_batch_size = args.env_batch_size
    cfg.num_envs = args.env_batch_size
    cfg.seed = args.seed
    cfg.use_wandb = False
    cfg.mjlab_fixed_goal = getattr(cfg, 'mjlab_fixed_goal', False)
    cfg.mjlab_disable_curriculum = getattr(
        cfg, 'mjlab_disable_curriculum', True)
    cfg.mjlab_command_resampling_time = getattr(
        cfg, 'mjlab_command_resampling_time', 40.0)
    if args.descriptor_mode is not None:
        cfg.mjlab_descriptor_mode = args.descriptor_mode

    archive = pd.read_pickle(args.archive)
    if archive.empty:
        raise ValueError(f'Archive contains no elites: {args.archive}')
    solution_columns = [
        column for column in archive.columns if column.startswith('solution_')
    ]
    if not solution_columns:
        raise ValueError(
            f'Archive has no solution columns: {args.archive}')
    env = make_vec_env(cfg)
    cfg.obs_shape = policy_observation_space(
        env.observation_space).shape[1:]
    cfg.action_shape = env.action_space.shape[1:]
    cfg.batch_size = cfg.env_batch_size * cfg.rollout_length
    cfg.minibatch_size = cfg.batch_size // cfg.num_minibatches
    ppo = PPO(cfg)

    rows = []
    for start in range(0, len(archive), args.policies_per_batch):
        chunk = archive.iloc[start:start + args.policies_per_batch]
        agents = [
            Actor(cfg.obs_shape, cfg.action_shape, cfg.normalize_obs,
                  cfg.normalize_returns, cfg.action_transform,
                  getattr(cfg, 'action_std_parameterization',
                          'log'),
                  hidden_dims=getattr(cfg, 'actor_hidden_dims',
                                      (400, 200, 100))).deserialize(
                      row[solution_columns].to_numpy(dtype=np.float32))
            for _, row in chunk.iterrows()
        ]
        while len(agents) < args.policies_per_batch:
            agents.append(
                Actor(cfg.obs_shape, cfg.action_shape, cfg.normalize_obs,
                      cfg.normalize_returns,
                      cfg.action_transform,
                      getattr(cfg, 'action_std_parameterization',
                              'log'),
                      hidden_dims=getattr(cfg, 'actor_hidden_dims',
                                          (400, 200, 100))).deserialize(
                          chunk.iloc[-1][solution_columns].to_numpy(
                              dtype=np.float32)))
        vectorized = VectorizedActor(
            agents, Actor, cfg.obs_shape, cfg.action_shape,
            cfg.normalize_obs, cfg.normalize_returns,
            getattr(cfg, 'mixed_precision', True)).to(ppo.device)
        objectives, measures, metadata = ppo.evaluate(
            vectorized, env, verbose=True, deterministic=True)

        for offset, (_, original) in enumerate(chunk.iterrows()):
            data = metadata[offset]
            rows.append({
                'archive_row': start + offset,
                'stored_objective': float(original['objective']),
                'stored_measure_0': float(original['measures_0']),
                'stored_measure_1': float(original['measures_1']),
                'reevaluated_objective': float(objectives[offset]),
                'reevaluated_measure_0': float(measures[offset, 0]),
                'reevaluated_measure_1': float(measures[offset, 1]),
                'success_rate': data.get('episode_success_rate', np.nan),
                'max_object_height': data.get('max_object_height', np.nan),
                'min_position_error': data.get('min_position_error', np.nan),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    env.close()


if __name__ == '__main__':
    main()
