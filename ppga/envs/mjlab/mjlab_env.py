"""MJLab manipulation environments adapted to PPGA's QD contract."""

from __future__ import annotations

import torch

from ppga.envs.qd_env import (FINAL_MEASURES, FINAL_OBSERVATION,
                              FINAL_OBSERVATION_MASK, MEASURE_REWARDS,
                              MEASURES, TASK_METRICS, canonical_policy_observation,
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


def _seed_mjlab_rng(seed: int) -> None:
    """Seed MJLab's reset-time random number generators."""
    from mjlab.utils.random import seed_rng

    seed_rng(int(seed))


def stable_lift_success(env, command_name="lift_height",
                        max_object_speed=0.15) -> torch.Tensor:
    """Return true when the cube is at the goal and no longer moving fast."""
    command = env.command_manager.get_term(command_name)
    position_error = torch.linalg.vector_norm(
        command.target_pos - command.object.data.root_link_pos_w, dim=-1)
    linear_speed = torch.linalg.vector_norm(
        command.object.data.root_link_vel_w[:, :3], dim=-1)
    return ((position_error < float(command.cfg.success_threshold))
            & (linear_speed <= float(max_object_speed)))


def terminal_lift_success_bonus(env, command_name="lift_height",
                                max_object_speed=0.15) -> torch.Tensor:
    """Return a dt-neutral indicator for a one-time episodic success bonus."""
    return stable_lift_success(
        env, command_name, max_object_speed).to(torch.float32) / env.step_dt


class ApproachTransportMeasures:
    """Track task-relative approach and cube-transport path descriptors."""

    def __init__(self, env, transport_start_distance=0.03,
                 approach_deviation_reference=0.05,
                 transport_deviation_reference=0.15):
        self.env = env
        self.transport_start_distance = float(transport_start_distance)
        self.approach_deviation_reference = float(
            approach_deviation_reference)
        self.transport_deviation_reference = float(
            transport_deviation_reference)
        if self.transport_start_distance <= 0:
            raise ValueError(
                "mjlab_transport_start_distance must be positive")
        if self.approach_deviation_reference <= 0:
            raise ValueError(
                "mjlab_approach_deviation_reference must be positive")
        if self.transport_deviation_reference <= 0:
            raise ValueError(
                "mjlab_transport_deviation_reference must be positive")

        reaching_cfg = env.reward_manager.get_term_cfg("lift")
        self.reaching_std = float(reaching_cfg.params["reaching_std"])
        device = env.scene.env_origins.device
        num_envs = env.num_envs
        self.initial_object_pos = torch.zeros(num_envs, 3, device=device)
        self.initial_goal_pos = torch.zeros(num_envs, 3, device=device)
        self.initial_ee_to_cube = torch.zeros(num_envs, 3, device=device)
        self.approach_sum = torch.zeros(num_envs, device=device)
        self.approach_weight = torch.zeros(num_envs, device=device)
        self.transport_peak = torch.zeros(num_envs, device=device)
        self.transport_started = torch.zeros(
            num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(
                self.env.num_envs, device=self.initial_object_pos.device)
        command = self.env.command_manager.get_term("lift_height")
        self.initial_object_pos[env_ids] = (
            command.object.data.root_link_pos_w[env_ids])
        self.initial_goal_pos[env_ids] = command.target_pos[env_ids]
        self.initial_ee_to_cube[env_ids] = _raw_observation_term(
            self.env, "ee_to_cube")[env_ids]
        self.approach_sum[env_ids] = 0.0
        self.approach_weight[env_ids] = 0.0
        self.transport_peak[env_ids] = 0.0
        self.transport_started[env_ids] = False

    def update(self) -> torch.Tensor:
        command = self.env.command_manager.get_term("lift_height")
        object_pos = command.object.data.root_link_pos_w
        ee_to_cube = _raw_observation_term(self.env, "ee_to_cube")
        distance = torch.linalg.vector_norm(ee_to_cube, dim=-1)
        initial_approach_xy = self.initial_ee_to_cube[:, :2]
        initial_approach_direction = initial_approach_xy / torch.linalg.vector_norm(
            initial_approach_xy, dim=-1, keepdim=True).clamp_min(1e-6)
        # Measure signed displacement perpendicular to the direct line from
        # the episode's initial end-effector pose to the cube. Unlike a fixed
        # robot-frame left/right coordinate, this stays meaningful when reset
        # randomization moves the cube.
        approach_deviation = (
            initial_approach_direction[:, 0] * ee_to_cube[:, 1]
            - initial_approach_direction[:, 1] * ee_to_cube[:, 0])
        approach_side = 0.5 + approach_deviation / (
            2.0 * self.approach_deviation_reference)
        approach_active = ~self.transport_started
        proximity = torch.exp(
            -distance.square() / self.reaching_std**2) * approach_active
        self.approach_sum += approach_side.clamp(0.0, 1.0) * proximity
        self.approach_weight += proximity

        object_displacement = torch.linalg.vector_norm(
            object_pos - self.initial_object_pos, dim=-1)
        self.transport_started |= (
            object_displacement >= self.transport_start_distance)

        direct_path = self.initial_goal_pos - self.initial_object_pos
        path_denominator = direct_path.square().sum(dim=-1).clamp_min(1e-6)
        progress = ((object_pos - self.initial_object_pos) * direct_path).sum(
            dim=-1) / path_denominator
        straight_path_pos = (
            self.initial_object_pos
            + progress.clamp(0.0, 1.0).unsqueeze(-1) * direct_path)
        direct_path_xy = direct_path[:, :2]
        direct_path_direction = direct_path_xy / torch.linalg.vector_norm(
            direct_path_xy, dim=-1, keepdim=True).clamp_min(1e-6)
        path_error_xy = object_pos[:, :2] - straight_path_pos[:, :2]
        lateral_deviation = (
            direct_path_direction[:, 0] * path_error_xy[:, 1]
            - direct_path_direction[:, 1] * path_error_xy[:, 0])
        farther_from_path = (
            self.transport_started
            & (lateral_deviation.abs() > self.transport_peak.abs()))
        self.transport_peak = torch.where(
            farther_from_path, lateral_deviation, self.transport_peak)
        return self.measures()

    def measures(self) -> torch.Tensor:
        approach = torch.where(
            self.approach_weight > 0,
            self.approach_sum / self.approach_weight.clamp_min(1e-6),
            torch.full_like(self.approach_sum, 0.5))
        transport = 0.5 + self.transport_peak / (
            2.0 * self.transport_deviation_reference)
        return torch.stack((approach.clamp(0.0, 1.0),
                            transport.clamp(0.0, 1.0)), dim=-1)


class GripOrientationArmLengthMeasures:
    """Track grasp-time wrist tilt and arm extension descriptors."""

    def __init__(self, env, arm_length_min=0.20, arm_length_max=0.50,
                 transport_start_distance=0.03):
        self.env = env
        self.arm_length_min = float(arm_length_min)
        self.arm_length_max = float(arm_length_max)
        self.transport_start_distance = float(transport_start_distance)
        if self.arm_length_min < 0:
            raise ValueError("mjlab_arm_length_min cannot be negative")
        if self.arm_length_max <= self.arm_length_min:
            raise ValueError(
                "mjlab_arm_length_max must exceed mjlab_arm_length_min")
        if self.transport_start_distance <= 0:
            raise ValueError(
                "mjlab_transport_start_distance must be positive")

        reaching_cfg = env.reward_manager.get_term_cfg("lift")
        asset_cfg = reaching_cfg.params["asset_cfg"]
        self.robot = env.scene[asset_cfg.name]
        if len(asset_cfg.site_ids) != 1:
            raise ValueError(
                "grip descriptors require exactly one end-effector site")
        self.grasp_site_id = asset_cfg.site_ids[0]

        device = env.scene.env_origins.device
        num_envs = env.num_envs
        self.initial_object_pos = torch.zeros(num_envs, 3, device=device)
        self.closest_distance = torch.full(
            (num_envs,), float("inf"), device=device)
        self.grip_orientation = torch.full(
            (num_envs,), 0.5, device=device)
        self.arm_length = torch.full((num_envs,), 0.5, device=device)
        self.transport_started = torch.zeros(
            num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(
                self.env.num_envs, device=self.initial_object_pos.device)
        command = self.env.command_manager.get_term("lift_height")
        self.initial_object_pos[env_ids] = (
            command.object.data.root_link_pos_w[env_ids])
        self.closest_distance[env_ids] = float("inf")
        self.grip_orientation[env_ids] = 0.5
        self.arm_length[env_ids] = 0.5
        self.transport_started[env_ids] = False

    @staticmethod
    def _rotate_local_z(quaternion: torch.Tensor) -> torch.Tensor:
        """Return local positive-z axis in world coordinates for wxyz quats."""
        w, x, y, z = quaternion.unbind(dim=-1)
        return torch.stack((2.0 * (x * z + w * y),
                            2.0 * (y * z - w * x),
                            1.0 - 2.0 * (x.square() + y.square())), dim=-1)

    def update(self) -> torch.Tensor:
        command = self.env.command_manager.get_term("lift_height")
        object_pos = command.object.data.root_link_pos_w
        ee_to_cube = _raw_observation_term(self.env, "ee_to_cube")
        distance = torch.linalg.vector_norm(ee_to_cube, dim=-1)

        ee_pos = self.robot.data.site_pos_w[:, self.grasp_site_id]
        ee_quat = self.robot.data.site_quat_w[:, self.grasp_site_id]
        grip_axis = self._rotate_local_z(ee_quat)
        # Zero means a downward-pointing grasp axis; one means upward. This
        # measures wrist approach geometry without rewarding slow motion.
        grip_tilt = torch.acos(
            (-grip_axis[:, 2]).clamp(-1.0, 1.0)) / torch.pi
        arm_length = torch.linalg.vector_norm(
            ee_pos - self.robot.data.root_link_pos_w, dim=-1)
        normalized_arm_length = (
            (arm_length - self.arm_length_min)
            / (self.arm_length_max - self.arm_length_min)).clamp(0.0, 1.0)

        closer = (~self.transport_started) & (distance < self.closest_distance)
        self.closest_distance = torch.where(
            closer, distance, self.closest_distance)
        self.grip_orientation = torch.where(
            closer, grip_tilt, self.grip_orientation)
        self.arm_length = torch.where(
            closer, normalized_arm_length, self.arm_length)

        object_displacement = torch.linalg.vector_norm(
            object_pos - self.initial_object_pos, dim=-1)
        self.transport_started |= (
            object_displacement >= self.transport_start_distance)
        return self.measures()

    def measures(self) -> torch.Tensor:
        return torch.stack((self.grip_orientation.clamp(0.0, 1.0),
                            self.arm_length.clamp(0.0, 1.0)), dim=-1)


def _motion_effort_parameters(env, speed_reference=None):
    """Resolve YAM arm joints and physical normalization constants."""
    robot = env.scene["robot"]
    arm_joint_ids, arm_joint_names = robot.find_joints(r"joint[1-6]")
    if len(arm_joint_ids) != 6:
        raise ValueError(
            "motion_effort descriptors require YAM arm joints joint1..joint6; "
            f"found {arm_joint_names}")

    if speed_reference is None:
        velocity_penalty = env.reward_manager.get_term_cfg("joint_vel_hinge")
        speed_reference = float(velocity_penalty.params["max_vel"])
    speed_reference = float(speed_reference)
    if speed_reference <= 0:
        raise ValueError("mjlab_motion_speed_reference must be positive")

    effort_by_joint = {}
    for actuator in robot.actuators:
        effort_limit = getattr(actuator.cfg, "effort_limit", None)
        if effort_limit is None:
            continue
        for joint_id in actuator.target_ids.tolist():
            effort_by_joint[int(joint_id)] = float(effort_limit)
    missing = [
        name for joint_id, name in zip(arm_joint_ids, arm_joint_names)
        if joint_id not in effort_by_joint
    ]
    if missing:
        raise ValueError(
            f"No actuator effort limits available for arm joints {missing}")
    effort_limits = torch.tensor(
        [effort_by_joint[joint_id] for joint_id in arm_joint_ids],
        dtype=torch.float32,
        device=robot.data.joint_vel.device)
    return arm_joint_ids, speed_reference, effort_limits


def lift_cube_measures(env, mode: str = "approach_transport", *,
                       arm_joint_ids=None, speed_reference=None,
                       effort_limits=None) -> torch.Tensor:
    """Return dense manipulation descriptors in ``[0, 1]``.

    ``approach_transport`` and ``grip_orientation_arm_length`` are stateful
    and handled by episode trackers. ``motion_effort`` captures manipulation
    style using normalized arm-joint speed and actuator effort.
    ``height_approach`` separates task progress (normalized cube height) from
    behavior (which lateral side of the cube the end effector approaches).
    ``progress`` preserves the original reaching / goal-proximity measures for
    loading or reproducing older archives.
    """
    if mode in {"approach_transport", "grip_orientation_arm_length"}:
        raise ValueError(
            f"{mode} measures require an episode tracker")
    if mode == "motion_effort":
        if (arm_joint_ids is None or speed_reference is None
                or effort_limits is None):
            arm_joint_ids, speed_reference, effort_limits = (
                _motion_effort_parameters(env, speed_reference))
        robot = env.scene["robot"]
        joint_speed = robot.data.joint_vel[:, arm_joint_ids].abs()
        actuator_effort = robot.data.qfrc_actuator[:, arm_joint_ids].abs()
        motion = (joint_speed / float(speed_reference)).mean(dim=-1)
        effort = (actuator_effort / effort_limits).mean(dim=-1)
        return torch.stack((motion.clamp(0.0, 1.0),
                            effort.clamp(0.0, 1.0)), dim=-1).to(torch.float32)

    if mode == "progress":
        ee_to_cube = _raw_observation_term(env, "ee_to_cube").norm(dim=-1)
        cube_to_goal = _raw_observation_term(env, "cube_to_goal").norm(dim=-1)
        reaching = torch.exp(-ee_to_cube / 0.20)
        bringing = torch.exp(-cube_to_goal / 0.30)
        return torch.stack((reaching, bringing), dim=-1).to(torch.float32)
    if mode != "height_approach":
        raise ValueError(f"Unknown MJLab descriptor mode: {mode!r}")

    command = env.command_manager.get_term("lift_height")
    object_height = (command.object.data.root_link_pos_w[:, 2]
                     - env.scene.env_origins[:, 2])
    # The task samples the cube near 2--5 cm and goals up to 40 cm. Keeping
    # the initial cube height above zero reserves low archive bins for policies
    # that never establish a lift.
    height = ((object_height - 0.02) / 0.38).clamp(0.0, 1.0)

    ee_to_cube_vec = _raw_observation_term(env, "ee_to_cube")
    direction = ee_to_cube_vec / ee_to_cube_vec.norm(
        dim=-1, keepdim=True).clamp_min(1e-6)
    approach_side = (0.5 * (direction[:, 1] + 1.0)).clamp(0.0, 1.0)
    return torch.stack((height, approach_side), dim=-1).to(torch.float32)


class QDRewardMJLab:
    """Expose MJLab as an auto-resetting, terminal-aware PPGA environment.

    The wrapped MJLab environment runs with ``auto_reset=False``. This wrapper
    therefore receives the genuine terminal observation, records it, and then
    performs a partial reset before returning the next policy observation.
    """

    def __init__(self, env, measure_reward_scale=None,
                 descriptor_mode="approach_transport",
                 motion_speed_reference=None,
                 transport_start_distance=0.03,
                 approach_deviation_reference=0.05,
                 transport_deviation_reference=0.15,
                 arm_length_min=0.20,
                 arm_length_max=0.50):
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
        self.descriptor_mode = descriptor_mode
        self.arm_joint_ids = None
        self.motion_speed_reference = None
        self.arm_effort_limits = None
        self.episode_measure_tracker = None
        if descriptor_mode == "motion_effort":
            (self.arm_joint_ids, self.motion_speed_reference,
             self.arm_effort_limits) = _motion_effort_parameters(
                 env, motion_speed_reference)
        elif descriptor_mode == "approach_transport":
            self.episode_measure_tracker = ApproachTransportMeasures(
                env,
                transport_start_distance=transport_start_distance,
                approach_deviation_reference=(
                    approach_deviation_reference),
                transport_deviation_reference=(
                    transport_deviation_reference))
        elif descriptor_mode == "grip_orientation_arm_length":
            self.episode_measure_tracker = GripOrientationArmLengthMeasures(
                env,
                arm_length_min=arm_length_min,
                arm_length_max=arm_length_max,
                transport_start_distance=transport_start_distance)
        if self.episode_measure_tracker is not None:
            self.episode_measure_tracker.reset()

    @property
    def unwrapped(self):
        return self.env

    def close(self):
        return self.env.close()

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        if self.episode_measure_tracker is not None:
            self.episode_measure_tracker.reset(kwargs.get("env_ids"))
        return canonical_policy_observation(observation), info

    def reset_with_common_random_numbers(self, num_groups: int, seed: int):
        """Reset contiguous policy groups onto identical random scenarios."""
        num_groups = int(num_groups)
        if num_groups < 1 or self.env.num_envs % num_groups != 0:
            raise ValueError(
                "num_groups must evenly divide the MJLab environment count")

        group_size = self.env.num_envs // num_groups
        info = {}
        policy_buffer = None
        for group_index in range(num_groups):
            # Rewinding every RNG before a same-sized partial reset gives each
            # policy block the same ordered cube/goal samples.
            _seed_mjlab_rng(seed)
            start = group_index * group_size
            env_ids = torch.arange(
                start, start + group_size,
                device=self.env.scene.env_origins.device)
            observation, info = self.env.reset(env_ids=env_ids)
            group_policy = policy_observation(observation)
            if policy_buffer is None:
                policy_buffer = torch.empty_like(group_policy)
            policy_buffer[env_ids] = group_policy[env_ids]

        if self.episode_measure_tracker is not None:
            self.episode_measure_tracker.reset()
        return {"policy": policy_buffer}, info

    def step(self, action):
        terminal_obs, reward, terminated, truncated, env_info = self.env.step(action)
        # MJLab reuses its extras dict and reset() clears extras["log"]. Keep
        # the terminal-step diagnostics independent of that mutation.
        info = dict(env_info)
        if isinstance(info.get("log"), dict):
            info["log"] = dict(info["log"])
        done = (terminated | truncated).reshape(-1)
        final_policy_obs = policy_observation(terminal_obs).clone()
        if self.episode_measure_tracker is not None:
            measures = self.episode_measure_tracker.update()
        else:
            measures = lift_cube_measures(
                self.env,
                self.descriptor_mode,
                arm_joint_ids=self.arm_joint_ids,
                speed_reference=self.motion_speed_reference,
                effort_limits=self.arm_effort_limits)
        final_measures = measures.clone()
        command = self.env.command_manager.get_term("lift_height")
        task_metrics = {
            name: value.clone()
            for name, value in command.metrics.items()
            if isinstance(value, torch.Tensor)
        }
        # Report height relative to each environment origin so it remains
        # meaningful if the scene layout uses vertical offsets.
        task_metrics["object_height"] = (
            command.object.data.root_link_pos_w[:, 2]
            - self.env.scene.env_origins[:, 2]).clone()
        if "task_success" in self.env.termination_manager.active_terms:
            task_metrics["episode_success"] = self.env.termination_manager.get_term(
                "task_success").to(torch.float32).clone()
        reward_manager = self.env.reward_manager
        step_reward = getattr(reward_manager, "_step_reward", None)
        if isinstance(step_reward, torch.Tensor):
            # MJLab documents _step_reward as the unscaled weighted reward
            # rate underlying get_active_iterable_terms(). Exposing it in
            # batch form makes reach-only reward plateaus diagnosable.
            info["reward_terms"] = {
                name: step_reward[:, index].clone()
                for index, name in enumerate(reward_manager.active_terms)
            }

        if done.any():
            env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            next_obs, reset_info = self.env.reset(env_ids=env_ids)
            if self.episode_measure_tracker is not None:
                self.episode_measure_tracker.reset(env_ids)
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
        info[FINAL_MEASURES] = final_measures
        info[TASK_METRICS] = task_metrics
        validate_qd_info(info, self.env.num_envs, measures.shape[1])
        return (canonical_policy_observation(next_obs), reward, terminated,
                truncated, info)


def configure_lift_task(cfg, env_cfg) -> None:
    """Apply PPGA's stationary-task overrides to an MJLab config in place."""
    episode_length_s = getattr(cfg, "episode_length_s", None)
    if episode_length_s is not None:
        env_cfg.episode_length_s = float(episode_length_s)

    if getattr(cfg, "mjlab_disable_curriculum", False):
        env_cfg.curriculum = {}

    command_cfg = env_cfg.commands["lift_height"]
    if getattr(cfg, "mjlab_fixed_goal", False):
        command_cfg.difficulty = "fixed"
    resampling_time = getattr(cfg, "mjlab_command_resampling_time", None)
    if resampling_time is None and getattr(cfg, "mjlab_fixed_goal", False):
        # Resample once during reset, then keep the same cube and goal for the
        # entire episode. Mid-episode resampling teleports the cube.
        resampling_time = 2.0 * float(env_cfg.episode_length_s)
    if resampling_time is not None:
        resampling_time = float(resampling_time)
        if resampling_time <= float(env_cfg.episode_length_s):
            raise ValueError(
                "mjlab_command_resampling_time must exceed episode_length_s "
                "to prevent mid-episode cube teleportation")
        command_cfg.resampling_time_range = (resampling_time, resampling_time)

    if getattr(cfg, "mjlab_terminate_on_success", False):
        from mjlab.managers.reward_manager import RewardTermCfg
        from mjlab.managers.termination_manager import TerminationTermCfg

        max_object_speed = float(getattr(
            cfg, "mjlab_success_max_object_speed", 0.15))
        success_bonus = float(getattr(cfg, "mjlab_success_bonus", 50.0))
        if max_object_speed <= 0:
            raise ValueError(
                "mjlab_success_max_object_speed must be positive")
        if success_bonus < 0:
            raise ValueError("mjlab_success_bonus cannot be negative")
        success_params = {
            "command_name": "lift_height",
            "max_object_speed": max_object_speed,
        }
        env_cfg.terminations["task_success"] = TerminationTermCfg(
            func=stable_lift_success, params=success_params)
        if success_bonus > 0:
            env_cfg.rewards["success_bonus"] = RewardTermCfg(
                func=terminal_lift_success_bonus,
                weight=success_bonus,
                params=success_params)


def make_base_env_mjlab(cfg):
    """Create an unwrapped state-based MJLab environment."""
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
    configure_lift_task(cfg, env_cfg)
    video_width = getattr(cfg, "video_width", None)
    video_height = getattr(cfg, "video_height", None)
    if video_width is not None:
        env_cfg.viewer.width = int(video_width)
    if video_height is not None:
        env_cfg.viewer.height = int(video_height)

    device = getattr(cfg, "device", None) or (
        "cuda" if torch.cuda.is_available() else "cpu")
    render_mode = "rgb_array" if getattr(cfg, "capture_video", False) else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device,
                            render_mode=render_mode)
    env.reset(seed=env_cfg.seed)
    return env


def make_vec_env_mjlab(cfg):
    """Create the state-based YAM lift task in a separate MJLab install."""
    env = make_base_env_mjlab(cfg)
    env = QDRewardMJLab(
        env,
        measure_reward_scale=getattr(cfg, "measure_reward_scale", None),
        descriptor_mode=getattr(cfg, "mjlab_descriptor_mode",
                                "approach_transport"),
        motion_speed_reference=getattr(
            cfg, "mjlab_motion_speed_reference", None),
        transport_start_distance=getattr(
            cfg, "mjlab_transport_start_distance", 0.03),
        approach_deviation_reference=getattr(
            cfg, "mjlab_approach_deviation_reference", 0.05),
        transport_deviation_reference=getattr(
            cfg, "mjlab_transport_deviation_reference", 0.15),
        arm_length_min=getattr(cfg, "mjlab_arm_length_min", 0.20),
        arm_length_max=getattr(cfg, "mjlab_arm_length_max", 0.50))
    return env
