from types import SimpleNamespace

import numpy as np
import pytest
import pandas as pd
import torch

import ppga.envs.mjlab.mjlab_env as mjlab_env
from ppga.algorithm.mjlab_archive_utils import (
    select_representative_elites, success_gated_objectives)
from ppga.qd.emitters.opt import XNES


def test_xnes_can_initialize_gradient_coefficients_at_zero():
    optimizer = XNES(
        solution_dim=3,
        device='cpu',
        sigma0=0.05,
        batch_size=8,
        seed=42,
        initial_bounds=([0.0, -2.0, -2.0], [2.0, 2.0, 2.0]),
        center_init=np.zeros(3, dtype=np.float32))

    np.testing.assert_array_equal(optimizer.mu, np.zeros(3))


def test_motion_effort_descriptors_use_arm_speed_and_effort_limits():
    robot = SimpleNamespace(data=SimpleNamespace(
        joint_vel=torch.tensor([[0.5, 0.25], [1.0, 1.0]]),
        qfrc_actuator=torch.tensor([[5.0, 10.0], [20.0, 40.0]]),
    ))
    env = SimpleNamespace(scene={"robot": robot})

    measures = mjlab_env.lift_cube_measures(
        env,
        "motion_effort",
        arm_joint_ids=[0, 1],
        speed_reference=0.5,
        effort_limits=torch.tensor([10.0, 20.0]))

    torch.testing.assert_close(
        measures, torch.tensor([[0.75, 0.5], [1.0, 1.0]]))


def test_archive_success_gate_rejects_failed_policies_below_threshold():
    objectives = torch.tensor([12.0, 30.0, 42.0]).numpy()
    metadata = [
        {"episode_success_rate": 0.0},
        {"episode_success_rate": 0.5},
        {"episode_success_rate": 1.0},
    ]

    gated = success_gated_objectives(objectives, metadata, 0.5, 0.0)

    assert gated[0] < 0.0
    assert gated[1:].tolist() == [30.0, 42.0]


def test_video_selection_starts_best_then_spreads_across_descriptors():
    archive = pd.DataFrame({
        "objective": [10.0, 20.0, 100.0],
        "measures_0": [0.0, 0.6, 1.0],
        "measures_1": [0.0, 0.6, 1.0],
        "metadata": [
            {"episode_success_rate": 1.0},
            {"episode_success_rate": 1.0},
            {"episode_success_rate": 0.0},
        ],
    })

    selected = select_representative_elites(
        archive, count=2, min_success_rate=0.5)

    assert [index for index, _, _ in selected] == [1, 0]
    assert [reason for _, _, reason in selected] == [
        "best_objective", "diverse_01"]


def test_height_approach_descriptors_cover_height_and_approach_side(monkeypatch):
    terms = {
        "ee_to_cube": torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]),
        "cube_to_goal": torch.ones(3, 3),
    }
    monkeypatch.setattr(
        mjlab_env, "_raw_observation_term",
        lambda _env, name: terms[name])

    command = SimpleNamespace(object=SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=torch.tensor([
            [0.0, 0.0, 0.02],
            [0.0, 0.0, 0.21],
            [0.0, 0.0, 0.40],
        ]))))
    env = SimpleNamespace(
        command_manager=SimpleNamespace(
            get_term=lambda _name: command),
        scene=SimpleNamespace(env_origins=torch.zeros(3, 3)),
    )

    measures = mjlab_env.lift_cube_measures(env, "height_approach")

    torch.testing.assert_close(measures[:, 0], torch.tensor([0.0, 0.5, 1.0]))
    torch.testing.assert_close(measures[:, 1], torch.tensor([1.0, 0.5, 0.0]))


def test_progress_descriptors_remain_available(monkeypatch):
    terms = {
        "ee_to_cube": torch.tensor([[0.2, 0.0, 0.0]]),
        "cube_to_goal": torch.tensor([[0.3, 0.0, 0.0]]),
    }
    monkeypatch.setattr(
        mjlab_env, "_raw_observation_term",
        lambda _env, name: terms[name])

    measures = mjlab_env.lift_cube_measures(object(), "progress")

    torch.testing.assert_close(measures, torch.exp(-torch.ones(1, 2)))


def test_stationary_task_overrides_disable_curriculum_and_resampling():
    command = SimpleNamespace(
        difficulty="dynamic", resampling_time_range=(8.0, 12.0))
    env_cfg = SimpleNamespace(
        episode_length_s=20.0,
        curriculum={"velocity": object()},
        commands={"lift_height": command},
    )
    cfg = SimpleNamespace(
        episode_length_s=None,
        mjlab_disable_curriculum=True,
        mjlab_fixed_goal=True,
        mjlab_command_resampling_time=None,
    )

    mjlab_env.configure_lift_task(cfg, env_cfg)

    assert env_cfg.curriculum == {}
    assert command.difficulty == "fixed"
    assert command.resampling_time_range == (40.0, 40.0)


def test_command_resampling_must_exceed_episode_horizon():
    env_cfg = SimpleNamespace(
        episode_length_s=20.0,
        curriculum={},
        commands={"lift_height": SimpleNamespace(difficulty="dynamic")},
    )
    cfg = SimpleNamespace(
        episode_length_s=None,
        mjlab_disable_curriculum=False,
        mjlab_fixed_goal=False,
        mjlab_command_resampling_time=20.0,
    )

    with pytest.raises(ValueError, match="must exceed episode_length_s"):
        mjlab_env.configure_lift_task(cfg, env_cfg)
