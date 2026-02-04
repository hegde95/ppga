import os
from typing import Optional

import gymnasium as gym
import torch

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
    class _Args:
        # Minimal set; add/extend if needed
        headless = headless

    app_launcher = AppLauncher(_Args())
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
    - clip_obs_rew: bool       (optional; ignored for Isaac by default)

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

    # Build Isaac Lab env configuration
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs, use_fabric=True)

    # Create the gym environment
    env = gym.make(task_name, cfg=env_cfg)

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


def close_isaac():
    """Close the Isaac environment and simulator app if it was launched here."""
    global _simulation_app
    # There's no single global env reference here; the caller should close envs directly.
    if _simulation_app is not None:
        try:
            _simulation_app.close()
        finally:
            _simulation_app = None


reward_offset = {
    "ant": 0.0,
    "humanoid": 0.0,
    "halfcheetah": 0.0,
    "hopper": 0.0,
    "walker2d": 0.0,
}