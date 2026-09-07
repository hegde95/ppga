"""MJLab manipulation environments adapted to PPGA's QD contract."""

from __future__ import annotations

import torch

from ppga.envs.qd_env import (FINAL_MEASURES, FINAL_OBSERVATION,
                              FINAL_OBSERVATION_MASK, MEASURE_REWARDS,
                              MEASURES, TASK_METRICS, canonical_policy_observation,
                              policy_observation, validate_qd_info)


_TASK_NAMES = {
    "lift_cube": "Mjlab-Lift-Cube-Yam",
    "yam_lift_cube": "Mjlab-Lift-Cube-Yam",
    "Mjlab-Lift-Cube-Yam": "Mjlab-Lift-Cube-Yam",
}


def _resolve_task_name(env_name: str) -> str:
    return _TASK_NAMES.get(env_name, env_name)


def _raw_observation_term(env, term_name: str) -> torch.Tensor:
    """Evaluate a noiseless manipulation observation term for descriptors."""
    manager = env.observation_manager
    group = "critic" if "critic" in manager.active_terms else "actor"
    if term_name not in manager.active_terms[group]:
        raise KeyError(
            f"MJLab task needs observation term {term_name!r} in its "
            f"{group!r} group to compute PPGA descriptors")
    term_cfg = manager.get_term_cfg(group, term_name)
    return term_cfg.func(env, **term_cfg.params)


def lift_cube_measures(env, mode: str = "height_approach") -> torch.Tensor:
    """Return dense manipulation descriptors in ``[0, 1]``.

    ``height_approach`` separates task progress (normalized cube height) from
    behavior (which lateral side of the cube the end effector approaches).
    ``progress`` preserves the original reaching / goal-proximity measures for
    loading or reproducing older archives.
    """
    ee_to_cube = _raw_observation_term(env, "ee_to_cube").norm(dim=-1)
    cube_to_goal = _raw_observation_term(env, "cube_to_goal").norm(dim=-1)
    reaching = torch.exp(-ee_to_cube / 0.20)
    bringing = torch.exp(-cube_to_goal / 0.30)
    if mode == "progress":
        return torch.stack((reaching, bringing), dim=-1).to(torch.float32)
    if mode != "height_approach":
        raise ValueError(f"Unknown MJLab descriptor mode: {mode!r}")

    command = env.command_manager.get_term("lift_height")
    object_height = (command.object.data.root_link_pos_w[:, 2]
                     - env.scene.env_origins[:, 2])
    # The task samples the cube near 2--5 cm and goals up to 40 cm. Keeping
    # the initial cube height above zero reserves low archive bins for policies
    # that never establish a lift.
    height = ((object_height - 0.02) / 0.38).clamp(0.0, 1.0)

    ee_to_cube_vec = _raw_observation_term(env, "ee_to_cube")
    direction = ee_to_cube_vec / ee_to_cube_vec.norm(
        dim=-1, keepdim=True).clamp_min(1e-6)
    approach_side = (0.5 * (direction[:, 1] + 1.0)).clamp(0.0, 1.0)
    return torch.stack((height, approach_side), dim=-1).to(torch.float32)


class QDRewardMJLab:
    """Expose MJLab as an auto-resetting, terminal-aware PPGA environment.

    The wrapped MJLab environment runs with ``auto_reset=False``. This wrapper
    therefore receives the genuine terminal observation, records it, and then
    performs a partial reset before returning the next policy observation.
    """

    def __init__(self, env, measure_reward_scale=None,
                 descriptor_mode="height_approach"):
        self.env = env
        if env.cfg.auto_reset:
            raise ValueError("QDRewardMJLab requires env.cfg.auto_reset=False")
        if "actor" not in env.observation_space.spaces:
            raise KeyError("MJLab environment has no 'actor' observation group")
        self.observation_space = type(env.observation_space)(spaces={
            "policy": env.observation_space.spaces["actor"]
        })
        self.single_observation_space = type(env.single_observation_space)(
            spaces={
                "policy": env.single_observation_space.spaces["actor"]
            })
        self.action_space = env.action_space
        self.single_action_space = env.single_action_space
        self.measure_reward_scale = (
            float(measure_reward_scale) if measure_reward_scale is not None
            else float(env.step_dt))
        self.descriptor_mode = descriptor_mode

    @property
    def unwrapped(self):
        return self.env

    def close(self):
        return self.env.close()

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return canonical_policy_observation(observation), info

    def step(self, action):
        terminal_obs, reward, terminated, truncated, env_info = self.env.step(action)
        # MJLab reuses its extras dict and reset() clears extras["log"]. Keep
        # the terminal-step diagnostics independent of that mutation.
        info = dict(env_info)
        if isinstance(info.get("log"), dict):
            info["log"] = dict(info["log"])
        done = (terminated | truncated).reshape(-1)
        final_policy_obs = policy_observation(terminal_obs).clone()
        measures = lift_cube_measures(self.env, self.descriptor_mode)
        command = self.env.command_manager.get_term("lift_height")
        task_metrics = {
            name: value.clone()
            for name, value in command.metrics.items()
            if isinstance(value, torch.Tensor)
        }
        # Report height relative to each environment origin so it remains
        # meaningful if the scene layout uses vertical offsets.
        task_metrics["object_height"] = (
            command.object.data.root_link_pos_w[:, 2]
            - self.env.scene.env_origins[:, 2]).clone()
        reward_manager = self.env.reward_manager
        step_reward = getattr(reward_manager, "_step_reward", None)
        if isinstance(step_reward, torch.Tensor):
            # MJLab documents _step_reward as the unscaled weighted reward
            # rate underlying get_active_iterable_terms(). Exposing it in
            # batch form makes reach-only reward plateaus diagnosable.
            info["reward_terms"] = {
                name: step_reward[:, index].clone()
                for index, name in enumerate(reward_manager.active_terms)
            }

        if done.any():
            env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            next_obs, reset_info = self.env.reset(env_ids=env_ids)
            # Preserve episode logging produced by the terminal step.
            if "log" not in info and "log" in reset_info:
                info["log"] = reset_info["log"]
        else:
            next_obs = terminal_obs

        info[MEASURES] = measures
        info[MEASURE_REWARDS] = measures * self.measure_reward_scale
        info["measure_reward_scale"] = self.measure_reward_scale
        info[FINAL_OBSERVATION] = {"policy": final_policy_obs}
        info[FINAL_OBSERVATION_MASK] = done.clone()
        info[FINAL_MEASURES] = measures.clone()
        info[TASK_METRICS] = task_metrics
        validate_qd_info(info, self.env.num_envs, measures.shape[1])
        return (canonical_policy_observation(next_obs), reward, terminated,
                truncated, info)


def configure_lift_task(cfg, env_cfg) -> None:
    """Apply PPGA's stationary-task overrides to an MJLab config in place."""
    episode_length_s = getattr(cfg, "episode_length_s", None)
    if episode_length_s is not None:
        env_cfg.episode_length_s = float(episode_length_s)

    if getattr(cfg, "mjlab_disable_curriculum", False):
        env_cfg.curriculum = {}

    command_cfg = env_cfg.commands["lift_height"]
    if getattr(cfg, "mjlab_fixed_goal", False):
        command_cfg.difficulty = "fixed"
    resampling_time = getattr(cfg, "mjlab_command_resampling_time", None)
    if resampling_time is None and getattr(cfg, "mjlab_fixed_goal", False):
        # Resample once during reset, then keep the same cube and goal for the
        # entire episode. Mid-episode resampling teleports the cube.
        resampling_time = 2.0 * float(env_cfg.episode_length_s)
    if resampling_time is not None:
        resampling_time = float(resampling_time)
        if resampling_time <= float(env_cfg.episode_length_s):
            raise ValueError(
                "mjlab_command_resampling_time must exceed episode_length_s "
                "to prevent mid-episode cube teleportation")
        command_cfg.resampling_time_range = (resampling_time, resampling_time)


def make_vec_env_mjlab(cfg):
    """Create the state-based YAM lift task in a separate MJLab install."""
    try:
        import mjlab  # noqa: F401
        # Import the built-in task package explicitly so its registry entries
        # are available even when packaging entry-point discovery is disabled.
        import mjlab.tasks.manipulation.config.yam  # noqa: F401
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg
    except ImportError as exc:
        raise ImportError(
            "MJLab is optional. Create a separate Linux/WSL environment and "
            "install requirements-mjlab.txt before using --env_type=mjlab."
        ) from exc

    task_name = _resolve_task_name(cfg.env_name)
    env_cfg = load_env_cfg(task_name)
    env_cfg.scene.num_envs = int(getattr(cfg, "env_batch_size", 1))
    env_cfg.seed = int(getattr(cfg, "seed", 0))
    env_cfg.auto_reset = False
    configure_lift_task(cfg, env_cfg)

    device = getattr(cfg, "device", None) or (
        "cuda" if torch.cuda.is_available() else "cpu")
    render_mode = "rgb_array" if getattr(cfg, "capture_video", False) else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device,
                            render_mode=render_mode)
    env = QDRewardMJLab(
        env,
        measure_reward_scale=getattr(cfg, "measure_reward_scale", None),
        descriptor_mode=getattr(cfg, "mjlab_descriptor_mode",
                                "height_approach"))
    env.reset(seed=env_cfg.seed)
    return env
