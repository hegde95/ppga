import numpy as np
import torch

from ppga.models.actor_critic import Actor
from ppga.models.vectorized import VectorizedActor


def test_actor_tanh_actions_are_bounded_and_logprobs_recompute():
    torch.manual_seed(7)
    actor = Actor((3,), np.array([2]), action_transform="tanh")
    obs = torch.randn(8, 3)

    actions, old_logprob, _ = actor.get_action(obs)
    _, new_logprob, _ = actor.get_action(obs, action=actions)

    assert torch.all(actions.abs() < 1.0)
    torch.testing.assert_close(old_logprob, new_logprob, rtol=1e-5, atol=1e-5)


def test_actor_deterministic_action_is_squashed_mean():
    actor = Actor((3,), np.array([2]), action_transform="tanh")
    obs = torch.randn(4, 3)

    actions, _, _ = actor.get_action(obs, deterministic=True)

    torch.testing.assert_close(actions, torch.tanh(actor(obs)))


def test_vectorized_tanh_actions_are_bounded_and_logprobs_recompute():
    torch.manual_seed(11)
    actors = [
        Actor((3,), np.array([2]), action_transform="tanh")
        for _ in range(2)
    ]
    vectorized = VectorizedActor(actors, Actor, (3,), np.array([2]))
    obs = torch.randn(8, 3, device=vectorized.device)

    actions, old_logprob, _ = vectorized.get_action(obs)
    _, new_logprob, _ = vectorized.get_action(obs, action=actions)

    assert torch.all(actions.abs() < 1.0)
    torch.testing.assert_close(old_logprob, new_logprob, rtol=1e-5, atol=1e-5)


def test_clip_mode_executes_bounded_actions():
    actor = Actor((3,), np.array([2]), action_transform="clip")
    actor.actor_logstd.data.fill_(2.0)
    actions, _, _ = actor.get_action(torch.zeros(128, 3))

    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)
    assert (actor.last_raw_action.abs() > 1.0).any()


def test_direct_std_parameterization_matches_rsl_style_distribution():
    torch.manual_seed(19)
    actor = Actor((3,), np.array([2]),
                  action_std_parameterization="direct",
                  initial_action_std=0.25)
    obs = torch.zeros(4096, 3)

    actions, _, _ = actor.get_action(obs)

    residual = actions - actor(obs)
    torch.testing.assert_close(residual.std(dim=0), torch.full((2,), 0.25),
                               atol=0.015, rtol=0.0)

    vectorized = VectorizedActor([actor], Actor, (3,), np.array([2]))
    restored = vectorized.vec_to_models()[0]
    assert restored.action_std_parameterization == "direct"
