"""MJLab manipulation environments adapted to PPGA's QD contract."""

from __future__ import annotations

import torch

from ppga.envs.qd_env import (FINAL_MEASURES, FINAL_OBSERVATION,
                              FINAL_OBSERVATION_MASK, MEASURE_REWARDS,
                              MEASURES, canonical_policy_observation,
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


def lift_cube_measures(env) -> torch.Tensor:
    """Dense [0, 1] lift descriptors: reaching and goal proximity."""
    ee_to_cube = _raw_observation_term(env, "ee_to_cube").norm(dim=-1)
    cube_to_goal = _raw_observation_term(env, "cube_to_goal").norm(dim=-1)
    reaching = torch.exp(-ee_to_cube / 0.20)
    bringing = torch.exp(-cube_to_goal / 0.30)
    return torch.stack((reaching, bringing), dim=-1).to(torch.float32)


class QDRewardMJLab:
    """Expose MJLab as an auto-resetting, terminal-aware PPGA environment.

    The wrapped MJLab environment runs with ``auto_reset=False``. This wrapper
    therefore receives the genuine terminal observation, records it, and then
    performs a partial reset before returning the next policy observation.
    """

    def __init__(self, env, measure_reward_scale=None):
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
        measures = lift_cube_measures(self.env)

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
        validate_qd_info(info, self.env.num_envs, measures.shape[1])
        return (canonical_policy_observation(next_obs), reward, terminated,
                truncated, info)


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
    episode_length_s = getattr(cfg, "episode_length_s", None)
    if episode_length_s is not None:
        env_cfg.episode_length_s = float(episode_length_s)

    device = getattr(cfg, "device", None) or (
        "cuda" if torch.cuda.is_available() else "cpu")
    render_mode = "rgb_array" if getattr(cfg, "capture_video", False) else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device,
                            render_mode=render_mode)
    env = QDRewardMJLab(
        env, measure_reward_scale=getattr(cfg, "measure_reward_scale", None))
    env.reset(seed=env_cfg.seed)
    return env
