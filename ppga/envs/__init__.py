try:
    import gym
    from gym.envs.registration import registry, make, spec
except ImportError:
    try:
        import gymnasium as gym
        from gymnasium.envs.registration import registry, make, spec
    except ImportError:
        # Simulator-only environments such as MJLab do not require either Gym
        # package. Keep package import lazy; Brax registration will explain the
        # missing dependency if it is actually requested.
        gym = None
        registry = {}
        make = spec = None


def register(id, *args, **kwargs):
    if gym is None:
        raise ImportError("Brax environments require gym or gymnasium")
    env_specs = getattr(registry, 'env_specs', registry)
    if id in env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kwargs)
