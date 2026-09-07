import pytest
import torch

from ppga.RL.ppo import add_time_limit_bootstrap
from ppga.envs.qd_env import (policy_observation, replace_done_rows,
                              validate_qd_info)
from ppga.utils.normalize import ObsNormalizer


def test_time_limit_bootstrap_uses_terminal_value_only_for_truncations():
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    truncated = torch.tensor([[False, True], [True, False]])
    terminal_values = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    valid = truncated.clone()
    result = add_time_limit_bootstrap(rewards, truncated, terminal_values,
                                      valid, gamma=0.5)
    assert torch.equal(result, torch.tensor([[1.0, 12.0], [18.0, 4.0]]))


def test_time_limit_bootstrap_rejects_post_reset_fallback():
    with pytest.raises(RuntimeError, match="pre-reset final observations"):
        add_time_limit_bootstrap(
            torch.zeros(1, 2), torch.tensor([[False, True]]),
            torch.zeros(1, 2), torch.tensor([[False, False]]), gamma=0.99)


def test_policy_observation_accepts_backend_group_names():
    policy = torch.ones(2, 3)
    assert policy_observation({"policy": policy}) is policy
    assert policy_observation({"actor": policy}) is policy
    assert policy_observation(policy) is policy


def test_terminal_measures_replace_only_done_rows():
    post_reset = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    terminal = torch.tensor([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])
    result = replace_done_rows(post_reset, terminal,
                               torch.tensor([True, False, True]))
    assert torch.equal(result, torch.tensor([[3.0, 3.0], [1.0, 1.0],
                                             [5.0, 5.0]]))


def test_observation_snapshot_normalization_does_not_update_statistics():
    normalizer = ObsNormalizer((2,))
    normalizer(torch.tensor([[1.0, 3.0], [3.0, 5.0]]), update=True)
    count = normalizer.obs_rms.count.clone()
    normalizer(torch.tensor([[100.0, 200.0]]), update=False)
    assert torch.equal(normalizer.obs_rms.count, count)


def test_qd_info_shape_validation():
    info = {
        "measures": torch.zeros(4, 2),
        "measure_rewards": torch.zeros(4, 2),
        "final_observation_mask": torch.zeros(4, dtype=torch.bool),
    }
    validate_qd_info(info, num_envs=4, num_dims=2)
