"""Simulator-independent vector environment construction."""


def make_vec_env(cfg):
    backend = getattr(cfg, "env_type", "isaac")
    if backend == "brax":
        from ppga.envs.brax_custom.brax_env import make_vec_env_brax
        return make_vec_env_brax(cfg)
    if backend == "isaac":
        from ppga.envs.isaac_lab.isaac_env import make_vec_env_isaac
        return make_vec_env_isaac(cfg)
    if backend == "mjlab":
        from ppga.envs.mjlab.mjlab_env import make_vec_env_mjlab
        return make_vec_env_mjlab(cfg)
    raise ValueError(f"Unknown environment backend: {backend!r}")


reward_offset = {
    "ant": 0.0,
    "humanoid": 0.0,
    "halfcheetah": 0.0,
    "hopper": 0.0,
    "walker2d": 0.0,
    "lift_cube": 0.0,
    "Mjlab-Lift-Cube-Yam": 0.0,
}
