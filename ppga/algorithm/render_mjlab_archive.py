"""Render representative successful MJLab archive elites to MP4 files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mediapy as media
import numpy as np
import pandas as pd
import torch
from box import Box

from ppga.algorithm.mjlab_archive_utils import (
    archive_solution_columns, metadata_success_rate,
    restore_archive_actor, select_representative_elites)
from ppga.envs.mjlab.mjlab_env import (
    ApproachTransportMeasures, _motion_effort_parameters, lift_cube_measures,
    make_base_env_mjlab)
from ppga.envs.qd_env import policy_observation, policy_observation_space


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--num_policies', type=int, default=5)
    parser.add_argument('--episodes_per_policy', type=int, default=1)
    parser.add_argument('--min_success_rate', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=20260907)
    parser.add_argument('--video_width', type=int, default=640)
    parser.add_argument('--video_height', type=int, default=480)
    parser.add_argument('--frame_stride', type=int, default=2)
    parser.add_argument(
        '--max_steps', type=int, default=None,
        help='Optional video truncation; defaults to one complete episode')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_policies < 1 or args.episodes_per_policy < 1:
        raise ValueError('num_policies and episodes_per_policy must be positive')
    if not 0.0 <= args.min_success_rate <= 1.0:
        raise ValueError('min_success_rate must be in [0, 1]')
    if args.frame_stride < 1:
        raise ValueError('frame_stride must be positive')

    cfg = Box(json.loads(args.config.read_text()))
    cfg.env_type = 'mjlab'
    cfg.env_batch_size = 1
    cfg.num_envs = 1
    cfg.use_wandb = False
    cfg.capture_video = True
    cfg.video_width = args.video_width
    cfg.video_height = args.video_height
    cfg.mjlab_fixed_goal = getattr(cfg, 'mjlab_fixed_goal', False)
    cfg.mjlab_disable_curriculum = getattr(
        cfg, 'mjlab_disable_curriculum', True)
    cfg.mjlab_command_resampling_time = getattr(
        cfg, 'mjlab_command_resampling_time', 40.0)

    archive = pd.read_pickle(args.archive)
    if archive.empty:
        raise ValueError(f'Archive contains no elites: {args.archive}')
    solution_columns = archive_solution_columns(archive)
    selected = select_representative_elites(
        archive, args.num_policies, args.min_success_rate)

    env = make_base_env_mjlab(cfg)
    cfg.obs_shape = policy_observation_space(
        env.observation_space).shape[1:]
    cfg.action_shape = env.action_space.shape[1:]
    device = torch.device(getattr(cfg, 'device', None) or (
        'cuda' if torch.cuda.is_available() else 'cpu'))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    descriptor_mode = getattr(
        cfg, 'mjlab_descriptor_mode', 'approach_transport')
    descriptor_kwargs = {}
    approach_transport_tracker = None
    if descriptor_mode == 'motion_effort':
        arm_ids, speed_reference, effort_limits = _motion_effort_parameters(
            env, getattr(cfg, 'mjlab_motion_speed_reference', None))
        descriptor_kwargs = {
            'arm_joint_ids': arm_ids,
            'speed_reference': speed_reference,
            'effort_limits': effort_limits,
        }
    elif descriptor_mode == 'approach_transport':
        approach_transport_tracker = ApproachTransportMeasures(
            env,
            transport_start_distance=getattr(
                cfg, 'mjlab_transport_start_distance', 0.03),
            transport_deviation_reference=getattr(
                cfg, 'mjlab_transport_deviation_reference', 0.15))

    rows = []
    try:
        for selection_position, (archive_index, row, reason) in enumerate(
                selected):
            actor = restore_archive_actor(row, cfg, solution_columns).to(device)
            actor.eval()
            for episode in range(args.episodes_per_policy):
                episode_seed = args.seed + episode
                observation, _ = env.reset(seed=episode_seed)
                if approach_transport_tracker is not None:
                    approach_transport_tracker.reset()
                observation = policy_observation(observation).to(device)
                frames = []
                initial_frame = env.render()
                if initial_frame is not None:
                    frames.append(initial_frame[0] if initial_frame.ndim == 4
                                  else initial_frame)
                reward_total = 0.0
                measure_total = torch.zeros(2, device=device)
                success = 0.0
                max_height = -float('inf')
                horizon = int(args.max_steps or env.max_episode_length)

                for step in range(horizon):
                    with torch.no_grad():
                        actor_obs = observation
                        if cfg.normalize_obs:
                            actor_obs = actor.obs_normalizer(
                                actor_obs, update=False)
                        action, _, _ = actor.get_action(
                            actor_obs, deterministic=True)
                    observation, reward, terminated, truncated, _ = env.step(
                        action.to(torch.float32))
                    observation = policy_observation(observation).to(device)
                    reward_total += float(reward[0].item())
                    if approach_transport_tracker is not None:
                        rollout_measures = approach_transport_tracker.update()[0]
                    else:
                        rollout_measures = lift_cube_measures(
                            env, descriptor_mode, **descriptor_kwargs)[0]
                        measure_total += rollout_measures
                    command = env.command_manager.get_term('lift_height')
                    if 'task_success' in env.termination_manager.active_terms:
                        episode_success = env.termination_manager.get_term(
                            'task_success').to(torch.float32)
                    else:
                        episode_success = command.metrics.get(
                            'episode_success')
                    if episode_success is not None:
                        success = max(success,
                                      float(episode_success[0].item()))
                    object_height = (
                        command.object.data.root_link_pos_w[0, 2]
                        - env.scene.env_origins[0, 2])
                    max_height = max(max_height, float(object_height.item()))
                    if step % args.frame_stride == 0:
                        frame = env.render()
                        if frame is not None:
                            frames.append(frame[0] if frame.ndim == 4 else frame)
                    if bool((terminated | truncated)[0].item()):
                        break

                steps = step + 1
                if approach_transport_tracker is not None:
                    final_rollout_measures = (
                        approach_transport_tracker.measures()[0])
                else:
                    final_rollout_measures = measure_total / steps
                filename = (
                    f'{selection_position:02d}_{reason}_archive_{archive_index}'
                    f'_episode_{episode:02d}.mp4')
                output_path = args.output_dir / filename
                fps = max(1, int(round(
                    env.metadata.get('render_fps', 30) / args.frame_stride)))
                if not frames:
                    raise RuntimeError(
                        'MJLab returned no RGB frames; verify offscreen rendering')
                media.write_video(str(output_path), frames, fps=fps)
                metadata = row.get('metadata')
                rows.append({
                    'video': filename,
                    'selection_reason': reason,
                    'archive_index': archive_index,
                    'episode_seed': episode_seed,
                    'stored_objective': float(row['objective']),
                    'stored_measure_0': float(row['measures_0']),
                    'stored_measure_1': float(row['measures_1']),
                    'stored_success_rate': metadata_success_rate(row),
                    'rollout_objective': reward_total,
                    'rollout_measure_0': float(
                        final_rollout_measures[0].item()),
                    'rollout_measure_1': float(
                        final_rollout_measures[1].item()),
                    'rollout_success': success,
                    'rollout_max_object_height': max_height,
                    'steps': steps,
                    'stored_max_object_height': (
                        metadata.get('max_object_height', np.nan)
                        if isinstance(metadata, dict) else np.nan),
                })
                print(f'Saved {output_path}')
    finally:
        env.close()

    with open(args.output_dir / 'manifest.csv', 'w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
