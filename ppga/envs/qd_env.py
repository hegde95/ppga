"""Backend-neutral transition helpers for PPGA vector environments.

QD backends may use different actor-observation keys and reset conventions.
This module defines the small contract consumed by PPO without importing a
particular simulator.
"""

from typing import Any, Protocol, runtime_checkable

import torch


FINAL_OBSERVATION = "final_observation"
FINAL_OBSERVATION_MASK = "final_observation_mask"
FINAL_MEASURES = "final_measures"
MEASURES = "measures"
MEASURE_REWARDS = "measure_rewards"


@runtime_checkable
class QDVectorEnv(Protocol):
    """Structural interface expected by :class:`ppga.RL.ppo.PPO`."""

    observation_space: Any
    action_space: Any
    unwrapped: Any

    def reset(self, **kwargs) -> tuple[Any, dict]: ...

    def step(self, action: torch.Tensor) -> tuple[Any, torch.Tensor,
                                                    torch.Tensor, torch.Tensor,
                                                    dict]: ...

    def close(self) -> None: ...


def policy_observation(observation: Any) -> torch.Tensor:
    """Extract the canonical actor observation from a backend observation."""
    if not isinstance(observation, dict):
        return observation
    for key in ("policy", "actor"):
        if key in observation:
            return observation[key]
    raise KeyError("Observation dict must contain a 'policy' or 'actor' group")


def policy_observation_space(observation_space: Any) -> Any:
    """Extract the unbatched-compatible policy space from a Dict or Box."""
    spaces = getattr(observation_space, "spaces", None)
    if spaces is None:
        return observation_space
    for key in ("policy", "actor"):
        if key in spaces:
            return spaces[key]
    raise KeyError("Observation space must contain a 'policy' or 'actor' group")


def canonical_policy_observation(observation: Any) -> dict[str, torch.Tensor]:
    """Return only the actor input under the backend-neutral ``policy`` key."""
    return {"policy": policy_observation(observation)}


def replace_done_rows(current: torch.Tensor, terminal: torch.Tensor,
                      done: torch.Tensor) -> torch.Tensor:
    """Replace reset rows with values captured immediately before reset."""
    done = done.to(device=current.device, dtype=torch.bool).reshape(-1)
    if terminal.shape != current.shape:
        raise ValueError(
            f"Terminal tensor shape {terminal.shape} does not match {current.shape}")
    result = current.clone()
    result[done] = terminal.to(current.device)[done]
    return result


def validate_qd_info(info: dict, num_envs: int, num_dims: int) -> None:
    """Validate the tensor shapes shared by all QD simulator adapters."""
    for key in (MEASURES, MEASURE_REWARDS):
        if key not in info:
            raise KeyError(f"QD environment did not provide info['{key}']")
        if tuple(info[key].shape) != (num_envs, num_dims):
            raise ValueError(
                f"info['{key}'] has shape {tuple(info[key].shape)}; "
                f"expected {(num_envs, num_dims)}")
    if FINAL_OBSERVATION_MASK in info:
        mask = info[FINAL_OBSERVATION_MASK]
        if mask.numel() != num_envs:
            raise ValueError(
                f"info['{FINAL_OBSERVATION_MASK}'] must have {num_envs} entries")
