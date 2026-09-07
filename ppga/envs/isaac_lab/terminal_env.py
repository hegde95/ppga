"""Isaac Lab environment which preserves transition data across auto-reset."""

import torch
from isaaclab.envs import ManagerBasedRLEnv


CONTACT_SENSOR_NAMES = (
    "contact_forces_left_foot",
    "contact_forces_right_foot",
)


def humanoid_foot_contacts(env) -> torch.Tensor:
    """Return left/right foot contact indicators for every Isaac environment."""
    return torch.cat([
        (env.scene[name].data.net_forces_w_history.norm(dim=-1).max(
            dim=1).values > 1.0)
        for name in CONTACT_SENSOR_NAMES
    ], dim=1).to(torch.float32)


class PPGAManagerBasedRLEnv(ManagerBasedRLEnv):
    """Capture observations and descriptors immediately before `_reset_idx`.

    Isaac Lab auto-resets inside ``step`` and otherwise exposes only the next
    episode's initial observation. Overriding the narrow reset hook avoids
    copying Isaac Lab's step implementation or enabling its per-step recorder
    observation pass.
    """

    ppga_final_observation: torch.Tensor | None = None
    ppga_final_measures: torch.Tensor | None = None

    def _reset_idx(self, env_ids):
        # Managers do not exist during the earliest part of construction.
        if hasattr(self, "observation_manager") and env_ids is not None:
            observations = self.observation_manager.compute(
                update_history=False)
            policy_obs = observations["policy"]
            contacts = humanoid_foot_contacts(self)
            if self.ppga_final_observation is None:
                self.ppga_final_observation = torch.zeros_like(policy_obs)
            if self.ppga_final_measures is None:
                self.ppga_final_measures = torch.zeros_like(contacts)
            self.ppga_final_observation[env_ids] = policy_obs[env_ids].clone()
            self.ppga_final_measures[env_ids] = contacts[env_ids].clone()
        super()._reset_idx(env_ids)
