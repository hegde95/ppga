"""Convert an MJLab RSL-RL actor checkpoint into PPGA's Actor format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ppga.models.actor_critic import Actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_checkpoint')
    parser.add_argument('output_checkpoint')
    args = parser.parse_args()

    source = torch.load(args.input_checkpoint, map_location='cpu',
                        weights_only=False)
    native = source['actor_state_dict']
    linear_indices = sorted({
        int(key.split('.')[1]) for key in native
        if key.startswith('mlp.') and key.endswith('.weight')
    })
    weights = [native[f'mlp.{index}.weight'] for index in linear_indices]
    hidden_dims = tuple(weight.shape[0] for weight in weights[:-1])
    obs_dim = weights[0].shape[1]
    action_dim = weights[-1].shape[0]

    actor = Actor(
        (obs_dim,), np.array([action_dim]), normalize_obs=True,
        action_transform='none', action_std_parameterization='direct',
        hidden_dims=hidden_dims)
    linear_layers = [layer for layer in actor.actor_mean
                     if isinstance(layer, torch.nn.Linear)]
    for layer, index in zip(linear_layers, linear_indices):
        layer.weight.data.copy_(native[f'mlp.{index}.weight'])
        layer.bias.data.copy_(native[f'mlp.{index}.bias'])
    actor.actor_logstd.data.copy_(
        native['distribution.std_param'].reshape_as(actor.actor_logstd))

    normalizer = actor.obs_normalizer.obs_rms
    normalizer.mean.copy_(native['obs_normalizer._mean'].squeeze(0))
    # RSL evaluates (x - mean) / (std + 1e-2). Encode the same denominator
    # in PPGA's sqrt(var + epsilon) normalizer.
    effective_std = native['obs_normalizer._std'].squeeze(0) + 1e-2
    normalizer.var.copy_(
        (effective_std.square() - actor.obs_normalizer.epsilon).clamp_min(0))
    normalizer.count.copy_(native['obs_normalizer.count'].float().reshape(1))

    output = Path(args.output_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'source_checkpoint': str(Path(args.input_checkpoint).resolve()),
        'action_std_parameterization': 'direct',
        'actor_hidden_dims': hidden_dims,
    }, output)
    print(f'Converted {args.input_checkpoint} -> {output}')
    print(f'obs_dim={obs_dim}, action_dim={action_dim}, hidden_dims={hidden_dims}')


if __name__ == '__main__':
    main()
