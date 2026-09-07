from typing import Optional
import argparse

import gymnasium as gym
import torch

from ppga.envs.qd_env import (FINAL_MEASURES, FINAL_OBSERVATION,
                              FINAL_OBSERVATION_MASK, MEASURE_REWARDS,
                              MEASURES, replace_done_rows, validate_qd_info)

# Import Isaac Lab tasks and helpers
# from isaaclab_tasks.utils import parse_env_cfg

try:
    from isaaclab.app import AppLauncher
except Exception:
    AppLauncher = None  # type: ignore[assignment]


# Lazily launched global simulation app (created on first env construction)
_simulation_app = None


_TO_TASK_NAME = {
    # Extend as you add more tasks
    "ant": "Isaac-Ant-v0",
    "humanoid": "Isaac-Humanoid-v0",
    # "walker2d": "Isaac-Walker2d-v0",  # uncomment if available in your task set
    # "halfcheetah": "Isaac-HalfCheetah-v0",
}


def _resolve_task_name(env_name: str) -> str:
    # Try explicit mapping first
    if env_name in _TO_TASK_NAME:
        return _TO_TASK_NAME[env_name]
    # Fallback: Title-case common names, e.g., "ant" -> "Isaac-Ant-v0"
    return f"Isaac-{env_name.capitalize()}-v0"


def _infer_device(cfg) -> str:
    # Prefer cfg.device if present; else CUDA if available; otherwise CPU
    device: Optional[str] = getattr(cfg, "device", None)
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _maybe_launch_sim_app(headless: bool = True):
    global _simulation_app
    if _simulation_app is not None:
        return _simulation_app
    if AppLauncher is None:
        # If AppLauncher isn't available, assume sim app isn't required in this context.
        return None

    # Create and launch the Omniverse/Isaac app once.
    # Respect headless mode by default for training.
    args = argparse.Namespace(headless=headless)
    # args.video = True
    # args.video_length = 500
    # args.enable_cameras = True
    app_launcher = AppLauncher(args)
    _simulation_app = app_launcher.app
    return _simulation_app


def make_vec_env_isaac(cfg):
    """
    Create a vectorized Isaac Lab Gymnasium environment analogous to make_vec_env_brax.

    Expected cfg fields:
    - env_name: str            (e.g., 'ant', 'humanoid')
    - env_batch_size: int      (number of parallel envs; mapped to num_envs)
    - seed: int                (optional; env.reset(seed=...))
    - device: str              (optional; e.g., 'cuda' or 'cpu')
    - measure_reward_scale: float (optional; defaults to the control step dt)

    Returns:
        gym.Env: Vectorized Isaac Lab environment.
    """
    # Launch the simulator app in headless mode if available/needed.
    _maybe_launch_sim_app(headless=True)

    import isaaclab_tasks  # noqa: F401  - required to register tasks with gym
    from isaaclab_tasks.utils import parse_env_cfg

    task_name = _resolve_task_name(cfg.env_name)
    device = _infer_device(cfg)
    num_envs = int(getattr(cfg, "env_batch_size", 1))

    # Use the repository's humanoid config directly. This avoids the previous
    # non-reproducible instruction to overwrite a file inside Isaac Lab.
    if cfg.env_name == 'humanoid':
        from humanoid_env_cfg import HumanoidEnvCfg
        env_cfg = HumanoidEnvCfg()
        env_cfg.scene.num_envs = num_envs
        env_cfg.sim.device = device
        env_cfg.sim.use_fabric = True
    else:
        env_cfg = parse_env_cfg(
            task_name, device=device, num_envs=num_envs, use_fabric=True)

    episode_length_s = getattr(cfg, "episode_length_s", None)
    if episode_length_s is not None:
        env_cfg.episode_length_s = float(episode_length_s)

    # Instantiate the manager environment directly so PPGA can capture the
    # terminal state in the narrow hook immediately before Isaac auto-resets.
    render_mode = "rgb_array" if getattr(cfg, "capture_video", False) else None
    from ppga.envs.isaac_lab.terminal_env import PPGAManagerBasedRLEnv
    env = PPGAManagerBasedRLEnv(cfg=env_cfg, render_mode=render_mode)
    env = QDRewardIsaac(
        env, measure_reward_scale=getattr(cfg, "measure_reward_scale", None))

    # Seed and reset (if seed present)
    if hasattr(cfg, "seed"):
        try:
            env.reset(seed=int(cfg.seed))
        except TypeError:
            # Some versions may not accept seed in reset; ignore gracefully
            env.reset()
    else:
        env.reset()

    # Isaac Lab envs operate on torch tensors directly and expose action/observation spaces.
    # No torch wrapper is required (unlike Brax).

    # Optional: If you'd like to mimic clipping behavior from Brax wrapper,
    # you can add a simple wrapper here in the future.

    return env

class QDRewardIsaac(gym.Wrapper):
    """
    Feet contact, based on the QDReward class in reward.py
    """
    def __init__(self, env, measure_reward_scale=None):
        super().__init__(env)
        # Isaac Lab reward terms are multiplied by the control dt. Apply the
        # same convention to descriptor pseudo-rewards used for DQD gradients,
        # while preserving raw [0, 1] contacts for archive descriptors.
        self.measure_reward_scale = (
            float(measure_reward_scale)
            if measure_reward_scale is not None
            else float(self.env.unwrapped.step_dt))

    def step(self, action):
        env_returns = self.env.step(action)
        # https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/contact_sensor.html
        # print(self.env.unwrapped.scene["contact_forces_LF"].data.net_forces_w.shape) # [3000, 2, 3]
        # contact_forces_feet = self.env.unwrapped.scene["contact_forces_feet"].data.net_forces_w
        # contact_forces_norm = torch.norm(contact_forces_feet, dim=-1) # [3000, 2]
        # env_returns[-1]['measures'] = contact_forces_norm
        # https://github.com/isaac-sim/IsaacLab/blob/f4aa17f87e2e5db5484f0b5974918573e8918ce2/source/isaaclab/isaaclab/envs/mdp/rewards.py#L267
        from ppga.envs.isaac_lab.terminal_env import humanoid_foot_contacts
        contacts = humanoid_foot_contacts(self.env.unwrapped)
        terminated, truncated = env_returns[2], env_returns[3]
        done = (terminated | truncated).reshape(-1)
        final_observation = self.env.unwrapped.ppga_final_observation
        final_measures = self.env.unwrapped.ppga_final_measures
        if done.any() and (final_observation is None or final_measures is None):
            raise RuntimeError(
                "Isaac reset without PPGA terminal data; use "
                "PPGAManagerBasedRLEnv to construct this environment")
        if final_observation is None:
            final_observation = env_returns[0]["policy"].clone()
        if final_measures is None:
            final_measures = contacts.clone()
        contacts = replace_done_rows(contacts, final_measures, done)
        info = env_returns[-1]
        info[MEASURES] = contacts
        info[MEASURE_REWARDS] = contacts * self.measure_reward_scale
        info['measure_reward_scale'] = self.measure_reward_scale
        info[FINAL_OBSERVATION] = {"policy": final_observation.clone()}
        info[FINAL_OBSERVATION_MASK] = done.clone()
        info[FINAL_MEASURES] = final_measures.clone()

        # Expose weighted per-step terms without changing the task reward.
        reward_manager = getattr(self.env.unwrapped, 'reward_manager', None)
        if reward_manager is not None and hasattr(reward_manager, '_step_reward'):
            info['reward_terms'] = {
                name: reward_manager._step_reward[:, i] * self.env.unwrapped.step_dt
                for i, name in enumerate(reward_manager.active_terms)
            }

        # Isaac Lab retains the most recent terminal cause for reset envs in
        # this buffer. Consumers should count it only where done is true.
        termination_manager = getattr(self.env.unwrapped, 'termination_manager', None)
        if termination_manager is not None and hasattr(termination_manager, '_term_dones'):
            info['termination_terms'] = {
                name: termination_manager._term_dones[:, i]
                for i, name in enumerate(termination_manager.active_terms)
            }

        # feet_body_forces = env_returns[0]['policy'][:, 53:66] # this is the wrench force, which may not be viable
        # print(torch.norm(feet_body_forces, dim=-1), torch.norm(feet_body_forces, dim=-1).shape)
        # env_returns[-1]['measures'] = feet_contact.values()
        validate_qd_info(info, self.env.unwrapped.num_envs, contacts.shape[1])
        return env_returns

def close_isaac():
    """Close the Isaac environment and simulator app if it was launched here."""
    global _simulation_app
    # There's no single global env reference here; the caller should close envs directly.
    if _simulation_app is not None:
        try:
            _simulation_app.close()
        finally:
            _simulation_app = None


from ppga.envs.factory import reward_offset  # backwards-compatible import
