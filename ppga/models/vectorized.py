from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

from ppga.models.policy import StochasticPolicy
from ppga.utils.normalize import ObsNormalizer, ReturnNormalizer


class VectorizedLinearBlock(nn.Module):

    def __init__(self,
                 weights: torch.Tensor,
                 biases=None,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.weight = nn.Parameter(weights).to(
            self.device
        )  # one slice of all the mlps we want to process as a batch
        self.bias = nn.Parameter(biases).to(
            self.device) if biases is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        obs_per_weight = x.shape[0] // self.weight.shape[0]
        x = torch.reshape(x, (-1, obs_per_weight, x.shape[1]))
        w_t = torch.transpose(self.weight, 1, 2).to(self.device)
        with autocast(device_type=self.device.type):
            y = torch.bmm(x, w_t)
        if self.bias is not None:
            y = torch.transpose(y, 0, 1)
            y += self.bias

        out_features = self.weight.shape[1]
        y = torch.transpose(y, 0, 1)
        y = torch.reshape(y, shape=(-1, out_features))
        return y


class VectorizedPolicy(StochasticPolicy, ABC):

    def __init__(self,
                 models,
                 model_fn,
                 obs_shape,
                 action_shape,
                 normalize_obs=False,
                 normalize_returns=False):
        StochasticPolicy.__init__(self,
                                  normalize_obs=normalize_obs,
                                  obs_shape=obs_shape,
                                  normalize_returns=normalize_returns)
        if not isinstance(models, np.ndarray):
            models = np.array(models)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.num_models = len(models)
        self.model_fn = model_fn
        self.blocks: List[VectorizedLinearBlock]
        self.actor_mean: nn.Sequential
        self.actor_logstd: nn.Parameter
        self.normalize_obs = normalize_obs
        self.normalize_returns = normalize_returns
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.action_transform = getattr(models[0], 'action_transform', 'none')
        if any(getattr(model, 'action_transform', 'none') != self.action_transform
               for model in models):
            raise ValueError('All vectorized actors must use the same action transform')
        self.last_raw_action = None

        if normalize_obs:
            self.obs_normalizers = [model.obs_normalizer for model in models]
        if normalize_returns:
            self.rew_normalizers = [model.return_normalizer for model in models]

    def _vectorize_layers(self, layer_name, models):
        '''
        Vectorize a specific nn.Sequential list of layers across all models of homogenous architecture
        :param layer_name: name of a nn.Sequential block
        :return: vectorized nn.Sequential block
        '''
        assert hasattr(models[0], layer_name), f'{layer_name=} not in the model'
        all_models_layers = [
            getattr(models[i], layer_name) for i in range(self.num_models)
        ]
        num_layers = len(getattr(models[0], layer_name))
        blocks = []
        for i in range(0, num_layers):
            if not isinstance(all_models_layers[0][i], nn.Linear):
                continue
            weights_slice = [
                all_models_layers[j][i].weight.to(self.device)
                for j in range(self.num_models)
            ]
            bias_slice = [
                all_models_layers[j][i].bias.to(self.device)
                for j in range(self.num_models)
            ]

            weights_slice = torch.stack(weights_slice)
            bias_slice = torch.stack(bias_slice)
            nonlinear = all_models_layers[0][i +
                                             1] if i + 1 < num_layers else None
            block = VectorizedLinearBlock(weights_slice, bias_slice)
            blocks.append(block)
            if nonlinear is not None:
                blocks.append(nonlinear)
        return blocks

    def vec_to_models(self):
        '''
        Returns a list of models view of the object
        '''
        models = [
            self.model_fn(self.obs_shape, self.action_shape, self.normalize_obs,
                          self.normalize_returns)
            for _ in range(self.num_models)
        ]
        for i, model in enumerate(models):
            if hasattr(model, 'action_transform'):
                model.action_transform = self.action_transform
            for l, layer in enumerate(self.actor_mean):
                # layer could be a nonlinearity
                if not isinstance(layer, VectorizedLinearBlock):
                    continue
                model.actor_mean[l].weight.data = layer.weight.data[i]
                model.actor_mean[l].bias.data = layer.bias.data[i]

            # update obs/rew normalizers
            if self.normalize_obs:
                model.obs_normalizer = self.obs_normalizers[i]
            if self.normalize_returns:
                model.return_normalizer = self.rew_normalizers[i]

            # update action logprobs
            model.actor_logstd.data = self.actor_logstd[i]
        return models

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_action(self, obs, action=None, deterministic=False):
        pass

    def vec_normalize_obs(self, obs, update=True):
        # TODO: make this properly vectorized
        obs = obs.reshape(self.num_models, obs.shape[0] // self.num_models, -1)
        for i, (model_obs,
                normalizer) in enumerate(zip(obs, self.obs_normalizers)):
            obs[i] = normalizer(model_obs, update=update)
        return obs.reshape(-1, obs.shape[-1])

    def vec_normalize_returns(self, rewards):
        # TODO: make this properly vectorized
        num_envs = rewards.shape[0]
        envs_per_model = num_envs // self.num_models
        rewards = rewards.reshape(self.num_models, envs_per_model)
        for i, (model_rews,
                normalizer) in enumerate(zip(rewards, self.rew_normalizers)):
            rewards[i] = normalizer(model_rews)
        return rewards.reshape(-1)


class VectorizedActor(VectorizedPolicy):

    def __init__(self,
                 models,
                 model_fn,
                 obs_shape,
                 action_shape,
                 normalize_obs=False,
                 normalize_returns=False):
        VectorizedPolicy.__init__(self,
                                  models,
                                  model_fn,
                                  obs_shape,
                                  action_shape,
                                  normalize_obs=normalize_obs,
                                  normalize_returns=normalize_returns)
        self.blocks = self._vectorize_layers('actor_mean', models)
        self.actor_mean = nn.Sequential(*self.blocks)
        action_logprobs = [model.actor_logstd for model in models]
        action_logprobs = torch.cat(action_logprobs).to(self.device)
        self.actor_logstd = nn.Parameter(action_logprobs)

    @autocast(device_type='cuda')
    def forward(self, x):
        return self.actor_mean(x)

    def get_action(self, obs, action=None, deterministic=False):
        with autocast(device_type=self.device.type):
            action_mean = self.actor_mean(obs)
        # Keep the probability calculation in float32. In particular, the
        # inverse tanh used by PPO to recompute stored action likelihoods is
        # too sensitive near +/-1 for float16 arithmetic.
        action_mean = action_mean.float()
        repeats = obs.shape[0] // self.num_models
        action_logstd = torch.repeat_interleave(self.actor_logstd,
                                                repeats,
                                                dim=0).float()
        action_logstd = action_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd) # may want to add nan fixes here, on action_mean and action_std
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            raw_action = action_mean if deterministic else probs.sample()
            if self.action_transform == 'tanh':
                action = torch.tanh(raw_action)
            elif self.action_transform == 'clip':
                action = raw_action.clamp(-1.0, 1.0)
            else:
                action = raw_action
        elif self.action_transform == 'tanh':
            eps = torch.finfo(action.dtype).eps
            raw_action = torch.atanh(action.clamp(-1.0 + eps, 1.0 - eps))
        else:
            raw_action = action

        self.last_raw_action = raw_action
        likelihood_action = action if self.action_transform == 'clip' else raw_action
        logprob = probs.log_prob(likelihood_action)
        if self.action_transform == 'tanh':
            logprob -= torch.log(1.0 - action.square() + 1e-6)
        logprob = logprob.sum(1)
        entropy = -logprob if self.action_transform == 'tanh' else probs.entropy().sum(1)
        return action, logprob, entropy
