"""Run MJLab's native RSL-RL trainer with PPGA task overrides.

This is a control experiment: it changes the task configuration in exactly
the same way as the PPGA adapter while leaving MJLab's trainer untouched.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from types import SimpleNamespace

from mjlab.scripts.train import TrainConfig, launch_training

from ppga.envs.mjlab.mjlab_env import configure_lift_task


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='Mjlab-Lift-Cube-Yam')
    parser.add_argument('--num_envs', type=int, default=768)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_root', default='experiments/mjlab_reference_stationary')
    parser.add_argument('--run_name', default='stationary_reference')
    parser.add_argument('--command_resampling_time', type=float, default=40.0)
    parser.add_argument('--episode_length_s', type=float, default=None)
    parser.add_argument('--fixed_goal', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--disable_curriculum',
                        action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig.from_task(args.task)
    cfg.env.scene.num_envs = args.num_envs
    configure_lift_task(
        SimpleNamespace(
            episode_length_s=args.episode_length_s,
            mjlab_disable_curriculum=args.disable_curriculum,
            mjlab_fixed_goal=args.fixed_goal,
            mjlab_command_resampling_time=args.command_resampling_time,
        ), cfg.env)

    cfg.agent.seed = args.seed
    cfg.agent.max_iterations = args.max_iterations
    cfg.agent.logger = 'tensorboard'
    cfg.agent.upload_model = False
    cfg.agent.run_name = f'{args.run_name}_seed_{args.seed}'
    cfg = replace(cfg, log_root=args.log_root, gpu_ids=[0])
    launch_training(args.task, cfg)


if __name__ == '__main__':
    main()
