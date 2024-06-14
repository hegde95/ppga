import functools

import brax
import gym
import torch
from brax.envs.wrappers.torch import TorchWrapper
from jax.dlpack import to_dlpack

from envs import brax_custom

v = torch.ones(1, device='cuda' if torch.cuda.is_available() else
               'cpu')  # init torch cuda before jax

_to_custom_env = {
    'ant': {
        'custom_env_name': 'brax_custom-ant-v0',
        'kwargs': {
            'clip_actions': (-1, 1),
        }
    },
    'humanoid': {
        'custom_env_name': 'brax_custom-humanoid-v0',
        'kwargs': {
            'clip_actions': (-1, 1),
        }
    },
    'walker2d': {
        'custom_env_name': 'brax-custom-walker2d-v0',
        'kwargs': {
            'clip_actions': (-1, 1),
        }
    },
    'halfcheetah': {
        'custom_env_name': 'brax-custom-halfcheetah-v0',
        'kwargs': {
            'clip_actions': (-1, 1),
        }
    },
}


def make_vec_env_brax(cfg):
    entry_point = functools.partial(brax_custom.create_gym_env,
                                    env_name=cfg.env_name)
    brax_env_name = _to_custom_env[cfg.env_name]['custom_env_name']
    if brax_env_name not in gym.envs.registry:
        gym.register(brax_env_name, entry_point=entry_point)

    kwargs = _to_custom_env[cfg.env_name]['kwargs']
    if cfg.clip_obs_rew:
        kwargs['clip_obs'] = (-10, 10)
        kwargs['clip_rewards'] = (-10, 10)
    vec_env = gym.make(_to_custom_env[cfg.env_name]['custom_env_name'],
                       batch_size=cfg.env_batch_size,
                       seed=cfg.seed,
                       **kwargs)
    vec_env = TorchWrapper(
        vec_env, device=('cuda' if torch.cuda.is_available() else 'cpu'))

    return vec_env


def make_single_env_brax(cfg):
    entry_point = functools.partial(brax_custom.create_gym_env,
                                    env_name=cfg.env_name)
    brax_env_name = _to_custom_env[cfg.env_name]['custom_env_name']
    if brax_env_name not in gym.envs.registry:
        gym.register(brax_env_name, entry_point=entry_point)

    kwargs = _to_custom_env[cfg.env_name]['kwargs']
    if cfg.clip_obs_rew:
        kwargs['clip_obs'] = (-10, 10)
        kwargs['clip_rewards'] = (-10, 10)

    single_env = gym.make(
        _to_custom_env[cfg.env_name]['custom_env_name'],
        # batch_size=None makes the environment be single; see
        # brax_custom.create_gym_env
        batch_size=None,
        seed=cfg.seed,
        **kwargs)
    single_env = TorchWrapper(
        single_env, device=('cuda' if torch.cuda.is_available() else 'cpu'))

    return single_env
