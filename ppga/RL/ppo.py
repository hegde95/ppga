import copy
import random
import time
from collections import deque
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch import Tensor

from ppga.models.actor_critic import Actor, Critic, QDCritic
from ppga.models.vectorized import VectorizedActor
from ppga.envs.qd_env import (FINAL_MEASURES, FINAL_OBSERVATION,
                              FINAL_OBSERVATION_MASK, TASK_METRICS,
                              policy_observation)
from ppga.utils.utilities import log, save_checkpoint

# based off of the clean-rl implementation
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py


def calculate_discounted_sum_torch(x: Tensor,
                                   dones: Tensor,
                                   discount: float,
                                   x_last: Optional[Tensor] = None) -> Tensor:
    """
    Computing cumulative sum (of something) for the trajectory, taking episode termination into consideration.
    """
    if x_last is None:
        x_last = x[-1].clone().fill_(0.0)

    cumulative = x_last

    discounted_sum = torch.zeros_like(x)
    i = len(x) - 1
    while i >= 0:
        cumulative = x[i] + discount * cumulative * (1.0 - dones[i])
        discounted_sum[i] = cumulative
        i -= 1

    return discounted_sum


def add_time_limit_bootstrap(rewards: Tensor, truncated: Tensor,
                             terminal_values: Tensor,
                             terminal_observation_mask: Tensor,
                             gamma: float) -> Tensor:
    """Bootstrap artificial time limits from the true terminal observation.

    ``dones`` still cuts the GAE recursion because the following observation is
    from a new episode. Only the one-step reward receives ``gamma * V(s_T)``.
    """
    truncated = truncated.to(device=rewards.device, dtype=torch.bool)
    valid = terminal_observation_mask.to(device=rewards.device,
                                         dtype=torch.bool)
    missing = truncated & ~valid
    if missing.any():
        count = int(missing.sum().item())
        raise RuntimeError(
            f"Cannot bootstrap {count} truncated transitions: the backend did "
            "not provide their pre-reset final observations")
    return rewards + float(gamma) * truncated.to(rewards.dtype) * terminal_values


def aggregate_episode_measures(measures_acc: Tensor, traj_lengths: Tensor,
                               final_measures: Tensor,
                               final_measure_mask: Tensor) -> Tensor:
    """Prefer backend-provided terminal descriptors over timestep averages."""
    num_envs = measures_acc.shape[1]
    measures = torch.zeros(
        num_envs, measures_acc.shape[2], device=measures_acc.device)
    for index in range(num_envs):
        if final_measure_mask[index]:
            measures[index] = final_measures[index]
        else:
            length = int(traj_lengths[index].item())
            measures[index] = measures_acc[:length, index].mean(dim=0)
    return measures


class PPO:

    def __init__(self, cfg):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.seed = cfg.seed
        self.num_envs = cfg.num_envs
        self.obs_shape = cfg.obs_shape
        self.action_shape = cfg.action_shape
        self.action_transform = getattr(cfg, 'action_transform', 'none')

        # Seed before constructing modules so initial parameters—not only
        # rollout sampling—are reproducible.
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = cfg.torch_deterministic

        agent = Actor(self.obs_shape,
                      self.action_shape,
                      normalize_obs=cfg.normalize_obs,
                      normalize_returns=cfg.normalize_returns,
                      action_transform=self.action_transform,
                      action_std_parameterization=getattr(
                          cfg, 'action_std_parameterization', 'log'),
                      initial_action_std=getattr(
                          cfg, 'initial_action_std', 1.0),
                      hidden_dims=getattr(
                          cfg, 'actor_hidden_dims', (400, 200, 100))).to(
                              self.device)
        self._agents = [agent]
        critic = QDCritic(self.obs_shape,
                          measure_dim=cfg.num_dims).to(self.device)
        self.qd_critic = critic
        self.vec_inference = VectorizedActor(
            self._agents,
            Actor,
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            normalize_obs=cfg.normalize_obs,
            normalize_returns=cfg.normalize_returns,
            use_amp=getattr(cfg, 'mixed_precision', True)).to(self.device)
        self.vec_optimizer = torch.optim.Adam(self.vec_inference.parameters(),
                                              lr=cfg.learning_rate,
                                              eps=1e-5)
        self.qd_critic_optim = torch.optim.Adam(self.qd_critic.parameters(),
                                                lr=cfg.learning_rate,
                                                eps=1e-5)
        self._theta = None  # nn params. Used for compatibility with DQD side

        # critic for moving the mean solution point
        self.mean_critic = Critic(self.obs_shape).to(self.device)
        self.mean_critic_optim = torch.optim.Adam(self.mean_critic.parameters(),
                                                  lr=cfg.learning_rate,
                                                  eps=1e-5)

        # metrics for logging
        self.metric_last_n_window = 100
        self.episodic_returns = deque([], maxlen=self.metric_last_n_window)
        self.episodic_lengths = deque([], maxlen=self.metric_last_n_window)
        self._report_interval = cfg.report_interval
        self.num_intervals = 0
        self.total_rewards = torch.zeros(self.num_envs)
        self.ep_len = torch.zeros(self.num_envs)

        # initialize tensors for training
        self.obs = torch.zeros((cfg.rollout_length, self.num_envs) +
                               self.obs_shape).to(self.device)
        self.actions = torch.zeros((cfg.rollout_length, self.num_envs) +
                                   self.action_shape).to(self.device)
        self.logprobs = torch.zeros(
            (cfg.rollout_length, self.num_envs)).to(self.device)
        self.rewards = torch.zeros(
            (cfg.rollout_length, self.num_envs)).to(self.device)
        self.dones = torch.zeros(
            (cfg.rollout_length, self.num_envs)).to(self.device)
        self.truncated = torch.zeros(
            (cfg.rollout_length, self.num_envs)).to(self.device)
        self.values = torch.zeros(
            (cfg.rollout_length, self.num_envs)).to(self.device)
        self.measures = torch.zeros((cfg.rollout_length, self.num_envs,
                                     self.cfg.num_dims)).to(self.device)
        self.final_observations = torch.zeros(
            (cfg.rollout_length, self.num_envs) + self.obs_shape,
            device=self.device)
        self.final_observation_mask = torch.zeros(
            (cfg.rollout_length, self.num_envs), dtype=torch.bool,
            device=self.device)

        self.next_obs = None
        self._rollout_state_valid = False
        # for moving the mean solution point w/ ppo
        self._grad_coeffs = torch.zeros(cfg.num_dims + 1).to(self.device)
        self._grad_coeffs[
            0] = 1.0  # default grad coefficients optimizes objective only
        self.obs_measure_coeffs = torch.zeros(
            (cfg.rollout_length, self.num_envs,
             self.obs_shape[0] + self.cfg.num_dims + 1)).to(self.device)

    @staticmethod
    def _policy_obs(observation):
        """Return the actor observation for both Isaac Dict and flat envs."""
        return policy_observation(observation)

    @property
    def agents(self):
        return self.vec_inference.vec_to_models()

    @agents.setter
    def agents(self, agents):
        self._agents = agents
        self.vec_inference = VectorizedActor(self._agents, Actor,
                                             self.obs_shape, self.action_shape,
                                             self.cfg.normalize_obs,
                                             self.cfg.normalize_returns,
                                             getattr(self.cfg,
                                                     'mixed_precision', True))
        self.vec_optimizer = torch.optim.Adam(self.vec_inference.parameters(),
                                              lr=self.cfg.learning_rate,
                                              eps=1e-5)
        # A different policy grouping invalidates both the environment-policy
        # assignment and any observation normalization applied to next_obs.
        self.next_obs = None
        self._rollout_state_valid = False

    @property
    def grad_coeffs(self):
        return self._grad_coeffs

    @grad_coeffs.setter
    def grad_coeffs(self, coeffs):
        if isinstance(coeffs, np.ndarray):
            coeffs = torch.tensor(coeffs).to(self.device)
        assert isinstance(
            coeffs,
            torch.Tensor), "grad coefficients should be a pytorch tensor"
        repeats = self.cfg.num_envs // coeffs.shape[0]
        coeffs = torch.repeat_interleave(coeffs, dim=0, repeats=repeats)
        self._grad_coeffs = coeffs

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, new_theta):
        self._theta = np.copy(new_theta)

    def update_critics(self, critics_list: List[Critic]):
        self.qd_critic = QDCritic(self.obs_shape,
                                  measure_dim=self.cfg.num_dims,
                                  critics_list=critics_list).to(self.device)
        self.qd_critic_optim = torch.optim.Adam(self.qd_critic.parameters(),
                                                lr=self.cfg.learning_rate,
                                                eps=1e-5)

    def update_critics_params(self, mean_critic_params, qd_critic_params):
        self.mean_critic.deserialize(mean_critic_params).to(self.device)
        self.mean_critic_optim = torch.optim.Adam(self.mean_critic.parameters(),
                                                  lr=self.cfg.learning_rate,
                                                  eps=1e-5)
        self.qd_critic.deserialize(qd_critic_params).to(self.device)
        self.qd_critic_optim = torch.optim.Adam(self.qd_critic.parameters(),
                                                lr=self.cfg.learning_rate,
                                                eps=1e-5)

    # noinspection NonAsciiCharacters
    def calculate_rewards(self,
                          next_obs,
                          next_done,
                          rewards,
                          values,
                          dones,
                          rollout_length,
                          calculate_dqd_gradients=False,
                          move_mean_agent=False):
        del next_done, rollout_length
        # Bootstrap the rollout tail, and separately bootstrap artificial time
        # limits from the pre-reset terminal observations captured per step.
        with torch.no_grad():
            if calculate_dqd_gradients:
                next_obs = next_obs.reshape(
                    self.cfg.num_dims + 1,
                    self.cfg.num_envs // (self.cfg.num_dims + 1), -1)
                next_value = []
                for i, obs, in enumerate(next_obs):
                    val = self.qd_critic.get_value_at(obs, dim=i)
                    if self.cfg.normalize_returns:
                        # need to denormalize the values
                        mean, var = self.vec_inference.rew_normalizers[
                            i].return_rms.mean, self.vec_inference.rew_normalizers[
                                i].return_rms.var
                        val = (torch.clamp(val, -5.0, 5.0) * torch.sqrt(
                            var.to(self.device))) + mean.to(self.device)

                    next_value.append(val)
                next_value = torch.cat(next_value).reshape(1,
                                                           -1).to(self.device)

                if self.cfg.normalize_returns:
                    denormalized_values = torch.empty_like(values)
                    envs_per_dim = self.cfg.num_envs // (self.cfg.num_dims + 1)
                    for i in range(self.cfg.num_dims + 1):
                        env_slice = slice(i * envs_per_dim,
                                          (i + 1) * envs_per_dim)
                        mean = self.vec_inference.rew_normalizers[
                            i].return_rms.mean.to(self.device)
                        var = self.vec_inference.rew_normalizers[
                            i].return_rms.var.to(self.device)
                        denormalized_values[:, env_slice] = (
                            torch.clamp(values[:, env_slice], -5.0, 5.0) *
                            torch.sqrt(var) + mean)
                    values = denormalized_values

            else:
                if move_mean_agent:
                    next_value = self.mean_critic.get_value(next_obs).reshape(
                        1, -1).to(self.device)
                else:
                    # standard ppo
                    next_value = self.qd_critic.get_value(next_obs).reshape(
                        1, -1).to(self.device)

                if self.cfg.normalize_returns:
                    #  need to de-normalize values
                    mean, var = self.vec_inference.rew_normalizers[
                        0].return_rms.mean, self.vec_inference.rew_normalizers[
                            0].return_rms.var
                    next_value = (torch.clamp(next_value, -5.0, 5.0) *
                                  torch.sqrt(var)) + mean
                    values = (torch.clamp(values, -5.0, 5.0) *
                              torch.sqrt(var)) + mean

            if self.cfg.value_bootstrap:
                terminal_obs = self.final_observations
                if self.cfg.normalize_obs:
                    terminal_obs = terminal_obs.clone()
                    envs_per_model = self.num_envs // self.vec_inference.num_models
                    for i, normalizer in enumerate(
                            self.vec_inference.obs_normalizers):
                        env_slice = slice(i * envs_per_model,
                                          (i + 1) * envs_per_model)
                        obs = terminal_obs[:, env_slice].reshape(
                            -1, self.obs_shape[0])
                        terminal_obs[:, env_slice] = normalizer(
                            obs, update=False).reshape(
                                terminal_obs.shape[0], envs_per_model, -1)

                terminal_values = torch.zeros_like(rewards)
                if calculate_dqd_gradients:
                    envs_per_dim = self.num_envs // (self.cfg.num_dims + 1)
                    for i in range(self.cfg.num_dims + 1):
                        env_slice = slice(i * envs_per_dim,
                                          (i + 1) * envs_per_dim)
                        obs = terminal_obs[:, env_slice].reshape(
                            -1, self.obs_shape[0])
                        val = self.qd_critic.get_value_at(obs, dim=i).reshape(
                            terminal_obs.shape[0], envs_per_dim)
                        if self.cfg.normalize_returns:
                            mean = self.vec_inference.rew_normalizers[
                                i].return_rms.mean.to(self.device)
                            var = self.vec_inference.rew_normalizers[
                                i].return_rms.var.to(self.device)
                            val = torch.clamp(val, -5.0, 5.0) * torch.sqrt(
                                var) + mean
                        terminal_values[:, env_slice] = val
                else:
                    flat_terminal_obs = terminal_obs.reshape(
                        -1, self.obs_shape[0])
                    critic = self.mean_critic if move_mean_agent else self.qd_critic
                    terminal_values = critic.get_value(
                        flat_terminal_obs).reshape_as(rewards)
                    if self.cfg.normalize_returns:
                        mean = self.vec_inference.rew_normalizers[
                            0].return_rms.mean.to(self.device)
                        var = self.vec_inference.rew_normalizers[
                            0].return_rms.var.to(self.device)
                        terminal_values = (
                            torch.clamp(terminal_values, -5.0, 5.0) *
                            torch.sqrt(var) + mean)

                rewards = add_time_limit_bootstrap(
                    rewards, self.truncated, terminal_values,
                    self.final_observation_mask, self.cfg.gamma)

            values = torch.cat([values, next_value])

            # section 3 in GAE paper: calculating advantages
            γ = self.cfg.gamma
            λ = self.cfg.gae_lambda
            deltas = (rewards - values[:-1]) + (1 - dones) * (γ * values[1:])
            advantages = calculate_discounted_sum_torch(deltas, dones, γ * λ)
            returns = advantages + values[:-1]
        return advantages, returns

    def batch_update(self,
                     values,
                     batched_data,
                     calculate_dqd_gradients=False,
                     move_mean_agent=False):
        with torch.no_grad():
            b_values = values
            (b_obs, b_logprobs, b_actions, b_advantages,
             b_returns) = batched_data
            batch_size = b_obs.shape[1]
            minibatch_size = batch_size // self.cfg.num_minibatches

            if (self.cfg.norm_adv
                    and not getattr(self.cfg, 'norm_adv_per_minibatch', True)):
                b_advantages = (
                    b_advantages - b_advantages.mean(dim=1, keepdim=True)
                ) / (b_advantages.std(dim=1, keepdim=True) + 1e-8)

            obs_dim, action_dim = self.obs_shape[0], self.action_shape[0]

            clipfracs = []
            actor_grad_norms = []
            critic_grad_norms = []

            pg_loss = v_loss = entropy_loss = ratio = None

        for epoch in range(self.cfg.update_epochs):
            b_inds = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy = self.vec_inference.get_action(
                    b_obs[:, mb_inds].reshape(-1, obs_dim),
                    b_actions[:, mb_inds].reshape(-1, action_dim))

                if calculate_dqd_gradients:
                    newvalue = []
                    for i in range(self.cfg.num_dims + 1):
                        newvalue.append(
                            self.qd_critic.get_value_at(b_obs[i, mb_inds],
                                                        dim=i))
                    newvalue = torch.cat(newvalue).to(self.device)
                elif move_mean_agent:
                    newvalue = self.mean_critic.get_value(
                        b_obs[:, mb_inds].reshape(-1, obs_dim))
                else:
                    # standard ppo
                    newvalue = self.qd_critic.get_value(b_obs[:,
                                                              mb_inds].reshape(
                                                                  -1, obs_dim))

                logratio = newlogprob - b_logprobs[:, mb_inds].flatten()
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    # noinspection PyUnresolvedReferences
                    clipfracs += [((ratio - 1.0).abs()
                                   > self.cfg.clip_coef).float().mean().item()]

                    # Match the adaptive schedule used by MJLab's RSL-RL PPO.
                    # An epoch-level stop alone cannot prevent early
                    # minibatches at a fixed 1e-3 rate from overshooting.
                    if (getattr(self.cfg, 'adaptive_kl', False)
                            and self.cfg.target_kl is not None):
                        current_lr = self.vec_optimizer.param_groups[0]['lr']
                        if approx_kl > self.cfg.target_kl * 2.0:
                            learning_rate = max(1e-5, current_lr / 1.5)
                        elif (approx_kl < self.cfg.target_kl / 2.0
                              and approx_kl > 0.0):
                            learning_rate = min(1e-2, current_lr * 1.5)
                        else:
                            learning_rate = current_lr
                        for optimizer in (self.vec_optimizer,
                                          self.qd_critic_optim,
                                          self.mean_critic_optim):
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = learning_rate

                mb_advantages = b_advantages[:, mb_inds].flatten()
                if (self.cfg.norm_adv
                        and getattr(self.cfg, 'norm_adv_per_minibatch', True)):
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8)

                # policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                newvalue = newvalue.view(-1)
                if self.cfg.clip_vloss:
                    v_loss_unclipped = (newvalue -
                                        b_returns[:, mb_inds].flatten())**2
                    v_clipped = b_values[:, mb_inds].flatten() + torch.clamp(
                        newvalue - b_values[:, mb_inds].flatten(),
                        -self.cfg.clip_value_coef,
                        self.cfg.clip_value_coef,
                    )
                    v_loss_clipped = (v_clipped -
                                      b_returns[:, mb_inds].flatten())**2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = v_loss_max.mean()
                else:
                    v_loss = ((newvalue -
                               b_returns[:, mb_inds].flatten())**2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.cfg.entropy_coef * entropy_loss + v_loss * self.cfg.vf_coef

                for p in self.vec_inference.parameters():
                    p.grad = None
                for p in self.qd_critic.parameters():
                    p.grad = None
                for p in self.mean_critic.parameters():
                    p.grad = None

                loss.backward()
                actor_grad_norms.append(
                    nn.utils.clip_grad_norm_(self.vec_inference.parameters(),
                                             self.cfg.max_grad_norm).detach())
                self.vec_optimizer.step()
                if move_mean_agent:
                    critic_grad_norms.append(
                        nn.utils.clip_grad_norm_(self.mean_critic.parameters(),
                                                 self.cfg.max_grad_norm).detach())
                    self.mean_critic_optim.step()
                else:
                    # works for standard ppo or the dqd step
                    critic_grad_norms.append(
                        nn.utils.clip_grad_norm_(self.qd_critic.parameters(),
                                                 self.cfg.max_grad_norm).detach())
                    self.qd_critic_optim.step()

            if self.cfg.target_kl is not None:
                if approx_kl > self.cfg.target_kl:
                    # print(f"Early stopping at epoch {epoch} due to reaching max kl {approx_kl}")
                    break

        actor_grad_norm = torch.stack(actor_grad_norms).mean()
        critic_grad_norm = torch.stack(critic_grad_norms).mean()
        return (pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl,
                clipfracs, ratio, actor_grad_norm, critic_grad_norm)

    def train(self,
              vec_env,
              num_updates,
              rollout_length,
              calculate_dqd_gradients=False,
              move_mean_agent=False,
              negative_measure_gradients=False,
              reset_env=True,
              start_update=0,
              initial_global_step=0,
              checkpoint_callback=None,
              checkpoint_interval=0):
        global_step = int(initial_global_step)

        if calculate_dqd_gradients:
            solution_params = self._agents[0].serialize()
            original_obs_normalizer = None
            original_return_normalizer = None
            if self.cfg.normalize_obs:
                original_obs_normalizer = self._agents[0].obs_normalizer
            if self.cfg.normalize_returns:
                original_return_normalizer = self._agents[0].return_normalizer
            # create copy of agent for f and one of each m
            agent_original_params = [
                copy.deepcopy(solution_params)
                for _ in range(self.cfg.num_dims + 1)
            ]
            agents = [
                Actor(self.obs_shape, self.action_shape, self.cfg.normalize_obs,
                      self.cfg.normalize_returns,
                      self.action_transform,
                      getattr(self.cfg, 'action_std_parameterization',
                              'log'),
                      hidden_dims=getattr(
                          self.cfg, 'actor_hidden_dims',
                          (400, 200, 100))).deserialize(params)
                for params in agent_original_params
            ]
            for agent in agents:
                if self.cfg.normalize_obs:
                    agent.obs_normalizer = copy.deepcopy(
                        original_obs_normalizer)
                if self.cfg.normalize_returns:
                    agent.return_normalizer = copy.deepcopy(
                        original_return_normalizer)
            self.agents = agents

        num_agents = len(self._agents)
        if self.num_envs % num_agents != 0:
            raise ValueError(
                f"num_envs={self.num_envs} must be divisible by "
                f"num_agents={num_agents}")

        # Resets remain the default at DQD phase boundaries, where the policy
        # assignment changes. Callers may continue a rollout only when the
        # exact same policy grouping remains installed.
        if reset_env or not self._rollout_state_valid or self.next_obs is None:
            self.next_obs = self._policy_obs(vec_env.reset()[0]).to(self.device)
            if self.cfg.normalize_obs:
                self.next_obs = self.vec_inference.vec_normalize_obs(
                    self.next_obs)
            self.total_rewards.zero_()
            self.ep_len.zero_()
        self._rollout_state_valid = True

        train_start = time.time()
        for update in range(int(start_update) + 1, num_updates + 1):
            if self.cfg.anneal_lr:
                frac = 1.0 - (update - 1.0) / max(num_updates, 1)
                learning_rate = frac * self.cfg.learning_rate
                for optimizer in (self.vec_optimizer, self.qd_critic_optim,
                                  self.mean_critic_optim):
                    optimizer.param_groups[0]['lr'] = learning_rate

            raw_out_of_bounds = 0
            saturated_actions = 0
            action_elements = 0
            raw_measure_sum = torch.zeros(self.cfg.num_dims, device=self.device)
            measure_reward_sum = torch.zeros(self.cfg.num_dims, device=self.device)
            measure_samples = 0
            reward_term_sums = {}
            termination_counts = {}
            task_metric_sums = {}
            task_metric_maxima = {}
            task_metric_minima = {}
            task_metric_samples = {}
            with torch.no_grad():
                for step in range(rollout_length):
                    global_step += self.num_envs

                    self.obs[step] = self.next_obs

                    action, logprob, _ = self.vec_inference.get_action(
                        self.next_obs)
                    # b/c of torch amp, need to convert back to float32
                    action = action.to(torch.float32)
                    raw_action = self.vec_inference.last_raw_action
                    raw_out_of_bounds += (raw_action.abs() > 1.0).sum().item()
                    if self.action_transform in {'clip', 'tanh'}:
                        saturated_actions += (action.abs() > 0.99).sum().item()
                    action_elements += action.numel()
                    if calculate_dqd_gradients:
                        next_obs = self.next_obs.reshape(
                            num_agents, self.cfg.num_envs // num_agents, -1)
                        value = []
                        for i, obs in enumerate(next_obs):
                            value.append(self.qd_critic.get_value_at(obs, i))
                        value = torch.cat(value).reshape(-1).to(self.device)
                    elif move_mean_agent:
                        value = self.mean_critic.get_value(self.next_obs)
                    else:
                        # standard ppo. Maintains backwards compatibility
                        value = self.qd_critic.get_value(self.next_obs)

                    self.values[step] = value.flatten()
                    self.actions[step] = action
                    self.logprobs[step] = logprob

                    env_returns = vec_env.step(action)
                    self.next_obs = self._policy_obs(env_returns[0])
                    reward = env_returns[1]
                    dones = env_returns[2] | env_returns[3] # terminated and truncated
                    infos = env_returns[4]
                    if self.cfg.normalize_obs:
                        self.next_obs = self.vec_inference.vec_normalize_obs(
                            self.next_obs)

                    # self.truncated[step] = infos['truncation']
                    self.truncated[step] = env_returns[3]
                    self.dones[step] = dones.view(-1)
                    self.final_observations[step].zero_()
                    self.final_observation_mask[step].zero_()
                    if FINAL_OBSERVATION in infos:
                        final_obs = self._policy_obs(
                            infos[FINAL_OBSERVATION]).to(self.device)
                        if self.cfg.normalize_obs:
                            final_obs = self.vec_inference.vec_normalize_obs(
                                final_obs, update=False)
                        if final_obs.shape != self.final_observations[step].shape:
                            raise ValueError(
                                "Backend final observation has shape "
                                f"{tuple(final_obs.shape)}; expected "
                                f"{tuple(self.final_observations[step].shape)}")
                        self.final_observations[step].copy_(final_obs)
                        final_mask = infos.get(FINAL_OBSERVATION_MASK, dones)
                        self.final_observation_mask[step].copy_(
                            final_mask.to(self.device).bool().reshape(-1))
                    raw_measures = infos['measures'].to(self.device)
                    measure_rewards = infos.get('measure_rewards', raw_measures).to(self.device)
                    if negative_measure_gradients:
                        measure_rewards = -measure_rewards
                    self.measures[step] = measure_rewards
                    raw_measure_sum += raw_measures.sum(dim=0)
                    measure_reward_sum += measure_rewards.sum(dim=0)
                    measure_samples += raw_measures.shape[0]
                    for name, values in infos.get('reward_terms', {}).items():
                        reward_term_sums[name] = reward_term_sums.get(name, 0.0) + values.mean().item()
                    for name, values in infos.get('termination_terms', {}).items():
                        values = values.to(dones.device).bool() & dones.bool()
                        termination_counts[name] = termination_counts.get(name, 0) + values.sum().item()
                    for name, values in infos.get(TASK_METRICS, {}).items():
                        values = values.detach()
                        task_metric_sums[name] = (
                            task_metric_sums.get(name, 0.0)
                            + values.sum().item())
                        task_metric_samples[name] = (
                            task_metric_samples.get(name, 0)
                            + values.numel())
                        current_max = values.max().item()
                        task_metric_maxima[name] = max(
                            task_metric_maxima.get(name, -float('inf')),
                            current_max)
                        current_min = values.min().item()
                        task_metric_minima[name] = min(
                            task_metric_minima.get(name, float('inf')),
                            current_min)
                    if move_mean_agent:
                        rew_measures = torch.cat(
                            (reward.unsqueeze(1), measure_rewards), dim=1)
                        rew_measures *= self._grad_coeffs
                        reward = rew_measures.sum(dim=1)
                    reward = reward.cpu()
                    self.total_rewards += reward
                    self.ep_len += 1

                    self.next_obs = self.next_obs.to(self.device)
                    # if self.cfg.normalize_returns:
                    #     reward = self.vec_inference.vec_normalize_returns(reward, self.next_done)
                    self.rewards[step] = reward.squeeze()

                    if not calculate_dqd_gradients and not move_mean_agent:
                        if dones.any():
                            dones_bool = dones.bool()
                            dones_cpu = dones_bool.cpu()
                            self.episodic_returns.extend(
                                self.total_rewards[dones_cpu].tolist())
                            self.episodic_lengths.extend(
                                self.ep_len[dones_cpu].tolist())
                            self.total_rewards[dones_cpu] = 0
                            self.ep_len[dones_cpu] = 0
                        self.num_intervals += 1

                if calculate_dqd_gradients:
                    envs_per_dim = self.cfg.num_envs // (self.cfg.num_dims + 1)
                    mask = torch.eye(self.cfg.num_dims + 1)
                    mask = torch.repeat_interleave(
                        mask, dim=0,
                        repeats=envs_per_dim).unsqueeze(dim=0).to(self.device)

                    # concat the reward w/ measures and mask appropriately
                    rew_measures = torch.cat(
                        (self.rewards.unsqueeze(dim=2), self.measures), dim=2)
                    rew_measures = (rew_measures * mask).sum(dim=2)
                    advantages, returns = self.calculate_rewards(
                        self.next_obs,
                        dones,
                        rew_measures,
                        self.values,
                        self.dones,
                        rollout_length=rollout_length,
                        calculate_dqd_gradients=True)
                else:
                    advantages, returns = self.calculate_rewards(
                        self.next_obs,
                        dones,
                        self.rewards,
                        self.values,
                        self.dones,
                        rollout_length=rollout_length,
                        move_mean_agent=move_mean_agent)
                # normalize the returns
                if self.cfg.normalize_returns:
                    for i, single_step_returns in enumerate(returns):
                        returns[
                            i][:] = self.vec_inference.vec_normalize_returns(
                                single_step_returns)

                # flatten the batch
                b_obs = self.obs.transpose(0, 1).reshape((
                    num_agents,
                    -1,
                ) + self.obs_shape)
                b_logprobs = self.logprobs.transpose(0,
                                                     1).reshape(num_agents, -1)
                b_actions = self.actions.transpose(0, 1).reshape((
                    num_agents,
                    -1,
                ) + self.action_shape)
                b_advantages = advantages.transpose(0,
                                                    1).reshape(num_agents, -1)
                b_returns = returns.transpose(0, 1).reshape(num_agents, -1)
                b_values = self.values.transpose(0, 1).reshape(num_agents, -1)

            # end of nograd ctx
            # update the network
            (pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs,
             ratio, actor_grad_norm, critic_grad_norm) = self.batch_update(
                 b_values,
                 (b_obs, b_logprobs, b_actions, b_advantages, b_returns),
                 calculate_dqd_gradients=calculate_dqd_gradients,
                 move_mean_agent=move_mean_agent)

            with torch.inference_mode():
                y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(
                    y_true - y_pred) / var_y

                std_param = self.vec_inference.actor_logstd
                if self.vec_inference.action_std_parameterization == 'direct':
                    avg_log_stddev = torch.log(
                        std_param.clamp(1e-6, 1e6)).mean().detach().cpu().numpy()
                else:
                    avg_log_stddev = std_param.mean().detach().cpu().numpy()
                avg_obj_magnitude = self.rewards.mean()
                branch_adv_std = advantages.transpose(0, 1).reshape(
                    num_agents, -1).std(dim=1)
                branch_value_mse = (b_values - b_returns).square().mean(dim=1)

                train_elapse = time.time() - train_start
                fps = global_step / train_elapse
                if not calculate_dqd_gradients and not move_mean_agent:  # backwards compatibility for standard PPO
                    if update % self._report_interval == 0:
                        episodic_reward = (
                            np.mean(self.episodic_returns)
                            if self.episodic_returns else np.nan)
                        reward_terms = ', '.join(
                            f'reward_{name}={value / rollout_length:.3f}'
                            for name, value in reward_term_sums.items())
                        log.debug(
                            f'FPS={fps:.2f}, steps={global_step}, '
                            f'episodic_reward={episodic_reward:.3f}, '
                            f'task_success={task_metric_sums.get("episode_success", 0.0) / max(task_metric_samples.get("episode_success", 0), 1):.3f}, '
                            f'object_height_max={task_metric_maxima.get("object_height", np.nan):.3f}, '
                            f'position_error_min={task_metric_minima.get("position_error", np.nan):.3f}, '
                            f'avg_logstd={float(avg_log_stddev):.3f}, '
                            f'learning_rate={self.vec_optimizer.param_groups[0]["lr"]:.2e}, '
                            f'approx_kl={approx_kl.item():.5f}, '
                            f'entropy={entropy_loss.item():.3f}, '
                            f'raw_action_oob={raw_out_of_bounds / max(action_elements, 1):.3f}, '
                            f'action_saturation={saturated_actions / max(action_elements, 1):.3f}'
                            + (f', {reward_terms}' if reward_terms else '')
                        )

                if self.cfg.use_wandb:
                    diagnostics = {
                        "charts/actor_avg_logstd": avg_log_stddev,
                        "charts/learning_rate": self.vec_optimizer.param_groups[0]["lr"],
                        "charts/average_rew_magnitude": avg_obj_magnitude,
                        f"losses/{move_mean_agent=}/value_loss": v_loss.item(),
                        "losses/value_loss": v_loss.item(),
                        "losses/policy_loss": pg_loss.item(),
                        "losses/entropy": entropy_loss.item(),
                        "losses/old_approx_kl": old_approx_kl.item(),
                        "losses/approx_kl": approx_kl.item(),
                        "losses/clipfrac": np.mean(clipfracs),
                        "losses/explained_variance": explained_var,
                        "train/value_loss": v_loss.item(),
                        "train/policy_loss": pg_loss.item(),
                        "train/value": self.values.mean().item(),
                        "train/adv_mean": advantages.mean().item(),
                        "train/adv_std": advantages.std().item(),
                        "train/adv_max": advantages.max().item(),
                        "train/adv_min": advantages.min().item(),
                        "train/act_min": action.min().item(),
                        "train/act_max": action.max().item(),
                        "train/ratio_min": ratio.min().item(),
                        "train/ratio_max": ratio.max().item(),
                        "train/raw_action_oob_fraction": raw_out_of_bounds / max(action_elements, 1),
                        "train/action_saturation_fraction": saturated_actions / max(action_elements, 1),
                        "train/actor_grad_norm": actor_grad_norm.item(),
                        "train/critic_grad_norm": critic_grad_norm.item(),
                        "Env step": global_step,
                        "global_step": global_step,
                        "Update": update,
                        "FPS": fps,
                        "perf/_fps": fps,
                    }
                    if measure_samples:
                        for i in range(self.cfg.num_dims):
                            diagnostics[f"train/measure_{i}_occupancy"] = (
                                raw_measure_sum[i] / measure_samples).item()
                            diagnostics[f"train/measure_{i}_reward_mean"] = (
                                measure_reward_sum[i] / measure_samples).item()
                    for i in range(num_agents):
                        diagnostics[f"train/branch_{i}_adv_std"] = branch_adv_std[i].item()
                        diagnostics[f"train/branch_{i}_value_mse"] = branch_value_mse[i].item()
                    for name, value in reward_term_sums.items():
                        diagnostics[f"reward_terms/{name}"] = value / rollout_length
                    for name, value in termination_counts.items():
                        diagnostics[f"terminations/{name}"] = value
                    for name, value in task_metric_sums.items():
                        diagnostics[f"task/{name}_mean"] = (
                            value / max(task_metric_samples[name], 1))
                        diagnostics[f"task/{name}_max"] = task_metric_maxima[name]
                        diagnostics[f"task/{name}_min"] = task_metric_minima[name]
                    wandb.log(diagnostics)

                    if len(self.episodic_returns):
                        # only log if we have something to report
                        wandb.log({
                            f"Average Episodic Reward":
                                sum(self.episodic_returns) /
                                len(self.episodic_returns),
                            "reward/reward":
                                sum(self.episodic_returns) /
                                len(self.episodic_returns),
                            "reward/reward_min":
                                min(self.episodic_returns),
                            "reward/reward_max":
                                max(self.episodic_returns),
                            "len/len_max":
                                max(self.episodic_lengths),
                            "len/len_min":
                                min(self.episodic_lengths),
                            "len/len":
                                sum(self.episodic_lengths) /
                                len(self.episodic_lengths),
                            "Env step":
                                global_step,
                            "global_step":
                                global_step,
                            "Update":
                                update
                        })

                    if self.cfg.normalize_obs:
                        wandb.log({
                            "train/obs_running_std":
                                self.vec_inference.obs_normalizers[0].obs_rms.
                                var.sqrt().mean().item(),
                            "train/obs_running_mean":
                                self.vec_inference.obs_normalizers[0].obs_rms.
                                mean.mean().item(),
                        })

            if (checkpoint_callback is not None and checkpoint_interval > 0
                    and update % checkpoint_interval == 0):
                checkpoint_callback(self, update, global_step)

        train_elapse = time.time() - train_start
        log.debug(f'train() took {train_elapse:.2f} seconds to complete')
        fps = global_step / train_elapse
        log.debug(f'FPS: {fps:.2f}')
        if self.cfg.use_wandb:
            wandb.log({'FPS: ': fps})

        if not calculate_dqd_gradients and not move_mean_agent:
            # standard ppo
            # log.debug("Saving checkpoint...")
            # trained_models = self.vec_inference.vec_to_models()
            # for i in range(num_agents):
            #     save_checkpoint(
            #         'checkpoints',
            #         f'{self.cfg.env_name}_{self.cfg.env_type}_model_{i}_checkpoint',
            #         trained_models[i], self.vec_optimizer)
            # log.debug("Done!")
            pass
        elif calculate_dqd_gradients:
            trained_agents = self.vec_inference.vec_to_models()
            new_params = np.array(
                [agent.serialize() for agent in trained_agents])
            jacobian = (new_params - agent_original_params).reshape(
                self.cfg.num_emitters, self.cfg.num_dims + 1, -1)

            original_agent = [
                Actor(self.obs_shape, self.action_shape, self.cfg.normalize_obs,
                      self.cfg.normalize_returns,
                      self.action_transform,
                      getattr(self.cfg, 'action_std_parameterization',
                              'log'),
                      hidden_dims=getattr(
                          self.cfg, 'actor_hidden_dims',
                          (400, 200, 100))).deserialize(
                          agent_original_params[0]).to(self.device)
            ]
            self.vec_inference = VectorizedActor(original_agent, Actor,
                                                 self.obs_shape,
                                                 self.action_shape,
                                                 self.cfg.normalize_obs,
                                                 self.cfg.normalize_returns,
                                                 getattr(self.cfg,
                                                         'mixed_precision',
                                                         True))
            f, m, metadata = self.evaluate(
                self.vec_inference,
                vec_env=vec_env,
                obs_normalizer=original_obs_normalizer,
                return_normalizer=original_return_normalizer)
            return f.reshape(self.vec_inference.num_models, ), \
                m.reshape(self.vec_inference.num_models, -1), \
                jacobian, \
                metadata

    def evaluate(self,
                 vec_agent,
                 vec_env,
                 verbose=False,
                 obs_normalizer=None,
                 return_normalizer=None,
                 deterministic=None):
        '''
        Evaluate all agents for one episode
        :param vec_agent: Vectorized agents for vectorized inference
        :returns: Sum rewards and measures for all agents
        '''
        if deterministic is None:
            deterministic = getattr(self.cfg, 'eval_deterministic', True)
        # Evaluation resets and advances vec_env using a potentially different
        # policy assignment, so a later training call must start a fresh phase.
        self._rollout_state_valid = False
        num_envs = vec_env.unwrapped.num_envs
        if num_envs % vec_agent.num_models != 0:
            raise ValueError(
                'Evaluation env count must be divisible by the number of policies')
        total_reward = np.zeros(num_envs)
        traj_length = 0
        num_steps = int(getattr(self.cfg, 'eval_max_steps', 0) or
                        getattr(vec_env.unwrapped, 'max_episode_length', 1000))

        obs = self._policy_obs(vec_env.reset()[0])
        obs = obs.to(self.device)
        dones = torch.zeros(num_envs, dtype=torch.bool)
        all_dones = torch.zeros((num_steps, num_envs), dtype=torch.bool)
        measures_acc = torch.zeros(
            (num_steps, num_envs, self.cfg.num_dims), device=self.device)
        measures = torch.zeros(
            (num_envs, self.cfg.num_dims), device=self.device)
        final_measures = torch.zeros_like(measures)
        final_measure_mask = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device)
        task_metric_extrema = {}

        fixed_obs_stats = None
        if self.cfg.normalize_obs and obs_normalizer is not None:
            fixed_obs_stats = (obs_normalizer.obs_rms.mean,
                               obs_normalizer.obs_rms.var)

        while not torch.all(dones) and traj_length < num_steps:
            with torch.no_grad():
                if self.cfg.normalize_obs:
                    if fixed_obs_stats is not None:
                        mean, var = fixed_obs_stats
                        obs = (obs - mean) / (torch.sqrt(var) + 1e-8)
                    else:
                        obs = vec_agent.vec_normalize_obs(obs, update=False)
                acts, _, _ = vec_agent.get_action(
                    obs, deterministic=deterministic)
                acts = acts.to(torch.float32)
                env_returns = vec_env.step(acts)
                obs = self._policy_obs(env_returns[0])
                rew = env_returns[1]
                next_dones = env_returns[2] | env_returns[3] # terminated and truncated
                infos = env_returns[4]

                measures_acc[traj_length] = infos['measures']
                if FINAL_MEASURES in infos:
                    measure_mask = infos.get(
                        FINAL_OBSERVATION_MASK, next_dones).to(
                            self.device).bool().reshape(-1)
                    # Evaluation may continue resetting fast environments until
                    # every policy rollout has finished. Keep the descriptor
                    # from each environment's first completed episode only.
                    measure_mask &= ~dones.to(self.device)
                    final_measures[measure_mask] = infos[FINAL_MEASURES].to(
                        self.device)[measure_mask]
                    final_measure_mask |= measure_mask
                active = (~dones).to(self.device)
                for name, values in infos.get(TASK_METRICS, {}).items():
                    values = values.to(self.device).reshape(-1)
                    if name == 'position_error':
                        initial = torch.full_like(values, float('inf'))
                        previous = task_metric_extrema.setdefault(name, initial)
                        task_metric_extrema[name] = torch.where(
                            active, torch.minimum(previous, values), previous)
                    else:
                        initial = torch.full_like(values, -float('inf'))
                        previous = task_metric_extrema.setdefault(name, initial)
                        task_metric_extrema[name] = torch.where(
                            active, torch.maximum(previous, values), previous)
                obs = obs.to(self.device)
                total_reward += rew.detach().cpu().numpy(
                ) * ~dones.cpu().numpy()
                dones = torch.logical_or(dones, next_dones.cpu())
                all_dones[traj_length] = dones.clone()
                traj_length += 1

        if not torch.all(dones):
            unfinished = (~dones).sum().item()
            raise RuntimeError(
                f"Evaluation hit {num_steps} steps with {unfinished} unfinished environments. "
                "Increase --eval_max_steps or verify task termination settings.")

        # the first done in each env is where that trajectory ends
        # CPU argmax is not implemented for bool tensors in PyTorch 2.7.
        traj_lengths = torch.argmax(all_dones.to(torch.int64), dim=0) + 1
        measures = aggregate_episode_measures(
            measures_acc, traj_lengths.to(self.device), final_measures,
            final_measure_mask)
        measures = measures.reshape(vec_agent.num_models,
                                    num_envs // vec_agent.num_models,
                                    -1).mean(dim=1).detach().cpu().numpy()

        total_reward = total_reward.reshape(
            (vec_agent.num_models,
             num_envs // vec_agent.num_models)).mean(axis=1)
        avg_traj_lengths = traj_lengths.to(torch.float32).reshape((vec_agent.num_models, num_envs // vec_agent.num_models)).\
            mean(dim=1).cpu().numpy()
        metadata = np.array([{
            'traj_length': float(t)
        } for t in avg_traj_lengths]).reshape(-1,)
        for name, per_env_values in task_metric_extrema.items():
            per_policy = per_env_values.reshape(
                vec_agent.num_models, num_envs // vec_agent.num_models)
            per_policy_mean = per_policy.mean(dim=1).detach().cpu().numpy()
            metadata_name = {
                'at_goal': 'success_rate',
                'episode_success': 'episode_success_rate',
                'object_height': 'max_object_height',
                'position_error': 'min_position_error',
            }.get(name, name)
            for i, value in enumerate(per_policy_mean):
                metadata[i][metadata_name] = float(value)
        max_reward = np.max(total_reward)
        min_reward = np.min(total_reward)
        mean_reward = np.mean(total_reward)
        mean_traj_length = torch.mean(traj_lengths.to(
            torch.float64)).detach().cpu().numpy().item()
        objective_measures = np.concatenate(
            (total_reward.reshape(-1, 1), measures), axis=1)

        if self.cfg.normalize_obs:
            for i, data in enumerate(metadata):
                normalizer = (obs_normalizer if obs_normalizer is not None
                              else vec_agent.obs_normalizers[i])
                data['obs_normalizer'] = copy.deepcopy(
                    normalizer.state_dict())

        if self.cfg.normalize_returns:
            for i, data in enumerate(metadata):
                normalizer = (return_normalizer
                              if return_normalizer is not None else
                              vec_agent.rew_normalizers[i])
                data['return_normalizer'] = copy.deepcopy(
                    normalizer.state_dict())

        if verbose:
            np.set_printoptions(suppress=True)
            log.debug('Finished Evaluation Step')
            log.info(f"Evaluation policy: {'deterministic' if deterministic else 'stochastic'}")
            # log.info(f'Reward + Measures: {objective_measures}')
            log.info(f'Max Reward on eval: {max_reward}')
            log.info(f'Min Reward on eval: {min_reward}')
            log.info(f'Mean Reward across all agents: {mean_reward}')
            log.info(f'Average Trajectory Length: {mean_traj_length}')
            if len(metadata) and 'episode_success_rate' in metadata[0]:
                success_rates = np.array([
                    data['episode_success_rate'] for data in metadata
                ])
                log.info(
                    f'Task success rate: mean={success_rates.mean():.4f}, '
                    f'max={success_rates.max():.4f}')
            if len(metadata) and 'max_object_height' in metadata[0]:
                heights = np.array([
                    data['max_object_height'] for data in metadata
                ])
                log.info(
                    f'Max object height: mean={heights.mean():.4f}, '
                    f'max={heights.max():.4f}')

        return total_reward.reshape(-1,), measures.reshape(
            -1, self.cfg.num_dims), metadata
