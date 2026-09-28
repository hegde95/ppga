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
from ppga.algorithm.mjlab_archive_utils import (
    archive_solution_columns, restore_archive_actor)
from ppga.models.actor_critic import Actor
from ppga.models.vectorized import VectorizedActor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--episodes-output', type=Path,
        help='Optional per-episode CSV for descriptor calibration audits')
    parser.add_argument('--env_batch_size', type=int, default=384)
    parser.add_argument('--policies_per_batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20260907)
    parser.add_argument('--descriptor_mode',
                        choices=['grip_orientation_elbow_extension',
                                 'grip_orientation_arm_length',
                                 'approach_transport', 'motion_effort',
                                 'height_approach', 'progress',
                                 'contact_azimuth_height',
                                 'approach_orientation',
                                 'contact_transport',
                                 'approach_inclination',
                                 'approach_average_tilt'],
                        default=None,
                        help='Optional override; defaults to the saved config')
    parser.add_argument('--contact-azimuth-frame', choices=['object', 'task'],
                        default=None,
                        help='Optional contact-azimuth frame override')
    parser.add_argument('--contact-sample', choices=['sensor', 'site'],
                        default=None,
                        help='Optional first-contact position override')
    parser.add_argument('--contact-height-reference', type=float, default=None,
                        help='Optional contact-height scale override')
    parser.add_argument('--contact-height-min', type=float, default=None)
    parser.add_argument('--contact-height-max', type=float, default=None)
    parser.add_argument('--grip-tilt-min-degrees', type=float, default=None)
    parser.add_argument('--grip-tilt-max-degrees', type=float, default=None)
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
    cfg.eval_common_random_numbers = True
    cfg.eval_fixed_scenarios = True
    cfg.eval_common_seed_offset = getattr(
        cfg, 'eval_common_seed_offset', 1000000)
    cfg.mjlab_fixed_goal = getattr(cfg, 'mjlab_fixed_goal', False)
    cfg.mjlab_disable_curriculum = getattr(
        cfg, 'mjlab_disable_curriculum', True)
    cfg.mjlab_command_resampling_time = getattr(
        cfg, 'mjlab_command_resampling_time', 40.0)
    if args.descriptor_mode is not None:
        cfg.mjlab_descriptor_mode = args.descriptor_mode
    if args.contact_azimuth_frame is not None:
        cfg.mjlab_contact_azimuth_frame = args.contact_azimuth_frame
    if args.contact_sample is not None:
        cfg.mjlab_contact_sample = args.contact_sample
    if args.contact_height_reference is not None:
        cfg.mjlab_contact_height_reference = args.contact_height_reference
    if args.contact_height_min is not None:
        cfg.mjlab_contact_height_min = args.contact_height_min
    if args.contact_height_max is not None:
        cfg.mjlab_contact_height_max = args.contact_height_max
    if args.grip_tilt_min_degrees is not None:
        cfg.mjlab_grip_tilt_min_degrees = args.grip_tilt_min_degrees
    if args.grip_tilt_max_degrees is not None:
        cfg.mjlab_grip_tilt_max_degrees = args.grip_tilt_max_degrees

    archive = pd.read_pickle(args.archive)
    if archive.empty:
        raise ValueError(f'Archive contains no elites: {args.archive}')
    solution_columns = archive_solution_columns(archive)
    env = make_vec_env(cfg)
    cfg.obs_shape = policy_observation_space(
        env.observation_space).shape[1:]
    cfg.action_shape = env.action_space.shape[1:]
    cfg.batch_size = cfg.env_batch_size * cfg.rollout_length
    cfg.minibatch_size = cfg.batch_size // cfg.num_minibatches
    ppo = PPO(cfg)

    rows = []
    episode_rows = []
    for start in range(0, len(archive), args.policies_per_batch):
        chunk = archive.iloc[start:start + args.policies_per_batch]
        agents = [
            restore_archive_actor(row, cfg, solution_columns)
            for _, row in chunk.iterrows()
        ]
        while len(agents) < args.policies_per_batch:
            agents.append(
                restore_archive_actor(chunk.iloc[-1], cfg, solution_columns))
        vectorized = VectorizedActor(
            agents, Actor, cfg.obs_shape, cfg.action_shape,
            cfg.normalize_obs, cfg.normalize_returns,
            getattr(cfg, 'mixed_precision', True)).to(ppo.device)
        # Every chunk must see the same held-out scenarios, including chunks
        # evaluated in separate calls to the simulator.
        ppo._evaluation_reset_count = 0
        objectives, measures, metadata = ppo.evaluate(
            vectorized, env, verbose=True, deterministic=True,
            return_episode_data=args.episodes_output is not None)

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
                'trajectory_length': data.get('traj_length', np.nan),
                'measure_0_std': data.get('measure_0_std', np.nan),
                'measure_1_std': data.get('measure_1_std', np.nan),
            })
            if args.episodes_output is not None:
                episodes = data['episodes']
                for episode_index, episode_measures in enumerate(
                        episodes['measures']):
                    episode_rows.append({
                        'archive_row': start + offset,
                        'episode': episode_index,
                        'objective': episodes['objective'][episode_index],
                        'length': episodes['length'][episode_index],
                        'measure_0': episode_measures[0],
                        'measure_1': episode_measures[1],
                        'success': episodes.get(
                            'episode_success',
                            [np.nan] * len(episodes['measures']))[
                                episode_index],
                        'max_object_height': episodes.get(
                            'object_height',
                            [np.nan] * len(episodes['measures']))[
                                episode_index],
                        'min_position_error': episodes.get(
                            'position_error',
                            [np.nan] * len(episodes['measures']))[
                                episode_index],
                    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if args.episodes_output is not None:
        args.episodes_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.episodes_output, 'w', newline='') as output_file:
            writer = csv.DictWriter(
                output_file, fieldnames=episode_rows[0].keys())
            writer.writeheader()
            writer.writerows(episode_rows)
    env.close()


if __name__ == '__main__':
    main()
