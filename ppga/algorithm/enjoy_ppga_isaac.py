import os
from pathlib import Path
import pickle
import numpy as np
from box import Box
from ppga.RL.ppo import *
from ppga.utils.utilities import log
from ppga.envs.isaac_lab.isaac_env import make_vec_env_isaac, reward_offset
# from ppga.utils.archive_utils_isaac import load_scheduler_from_checkpoint
from ppga.models.actor_critic import Actor
from pandas import DataFrame

from IPython.display import HTML, Image
from IPython.display import display

# params to config
device = torch.device('cuda')
env_name = 'humanoid'
seed = 2
normalize_obs = False
normalize_rewards = False
# non-configurable params
obs_shapes = {
    'humanoid': (87,),
}
action_shapes = {
    'humanoid': (21,),
}

# define the final config objects
actor_cfg = Box({
        'obs_shape': obs_shapes[env_name],
        'action_shape': action_shapes[env_name],
        'normalize_obs': normalize_obs,
        'normalize_rewards': normalize_rewards,
})
env_cfg = Box({
        'env_name': env_name,
        'env_batch_size': 1,
        'num_dims': 2,
        'seed': seed,
        'num_envs': 1,
        'num_steps': 32,
        'max_iterations': 1000,
        'learning_rate': 0.0005,
        'vf_coef': 1.0,
        'entropy_coef': 0.0,
        'num_minibatches': 4,
        'update_epochs': 5
})

# now lets load in a saved archive dataframe and scheduler
# change this to be your own checkpoint path
archive_path = '/home1/eh_352/ppga/experiments/paper_ppga_humanoid/2/checkpoints/cp_00001180/archive_df_00001180.pkl'
scheduler_path = '/home1/eh_352/ppga/experiments/paper_ppga_humanoid/2/checkpoints/cp_00001180/scheduler_00001180.pkl'
# each of these is ~4GB, so make sure more than 8GB of memory is allocated
with open(archive_path, 'rb') as f:
    archive_df = pickle.load(f)
with open(scheduler_path, 'rb') as f:
    scheduler = pickle.load(f)

# create the environment
env = make_vec_env_isaac(env_cfg)

def get_best_elite(scheduler):
    best_elite = scheduler.archive.best_elite
    # random_elite = scheduler.archive.sample_elites(1)
    print(f'Loading agent with reward {best_elite["objective"]} and measures {best_elite["measures"]}')
    agent = Actor(obs_shape=actor_cfg.obs_shape[0], action_shape=actor_cfg.action_shape, normalize_obs=normalize_obs, normalize_returns=normalize_rewards).deserialize(best_elite['solution']).to(device)
    if actor_cfg.normalize_obs:
        norm = best_elite.metadata['obs_normalizer']
        if isinstance(norm, dict):
            agent.obs_normalizer.load_state_dict(norm)
        else:
            agent.obs_normalizer = norm
    return agent

def enjoy_isaac(agent, render=True, deterministic=True):
    if actor_cfg.normalize_obs:
        obs_mean, obs_var = agent.obs_normalizer.obs_rms.mean, agent.obs_normalizer.obs_rms.var
        print(f'{obs_mean=}, {obs_var=}')

    obs = env.reset()[0]['policy']
    total_reward = 0
    measures = torch.zeros(env_cfg.num_dims).to(device)
    done = False
    steps = 0
    while not done:
        with torch.no_grad():
            obs = obs.unsqueeze(dim=0)
            if actor_cfg.normalize_obs:
                obs = (obs - obs_mean) / torch.sqrt(obs_var + 1e-8)

            if deterministic:
                act = agent.actor_mean(obs)
            else:
                act, _, _ = agent.get_action(obs)
            act = act.reshape(1, -1)
            env_returns = env.step(act)
            obs = env_returns[0]['policy']
            reward = env_returns[1]
            done = env_returns[2] or env_returns[3]
            info = env_returns[4]
            measures += info['measures'].squeeze()
            total_reward += reward
        steps += 1
    if render:
        pass
        # i = HTML(html.render(env.unwrapped._env.sys, [s.qp for s in rollout]))
        # display(i)
        # print(f'{total_reward=}')
        # print(f' Rollout length: {len(rollout)}')
        # measures /= len(rollout)
        # print(f'Measures: {measures.cpu().numpy()}')
    env.close()
    return total_reward.detach().cpu().numpy()[0], steps

agent = get_best_elite(scheduler)
reward, steps = enjoy_isaac(agent, render=False, deterministic=True)
print("Total Reward: ", reward)
print("Total steps: ", steps)