import sys
import os
import math
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.td3 import TD3
# from osim.env import ProstheticsEnv
from environment.prosthetics_env_with_history import ProstheticsEnvWithHistory
from environment.observations import prepare_model_observation, env_obs_history_to_model_obs
from environment.actions import prepare_env_action, reset_frameskip
from environment.rewards import env_obs_to_custom_reward
from distributed.database import persist_timesteps, persist_event, get_total_timesteps, clear_clients_for_thread
from distributed.db_history_sampler import DatabaseHistorySampler
from distributed.s3_checkpoints import load_s3_model_checkpoint, save_s3_model_checkpoint
import torch
import torch.utils.data


with open('config_distributed.json', 'r') as f:
    CONFIG = json.load(f)
print(json.dumps(CONFIG, indent=4))


env = ProstheticsEnvWithHistory(visualize=False, integrator_accuracy=CONFIG['env']['integrator_accuracy'])
env_step_kwargs = {'project': False}


# state_dim = env.observation_space.shape[0]
env.reset(**env_step_kwargs)
state_dim = prepare_model_observation(env).shape[0]
action_dim = env.action_space.shape[0]
max_action = int(env.action_space.high[0])
state_dim, action_dim, max_action


policy = TD3(state_dim, action_dim, max_action)


print(f"Loading policy checkpoints from {os.environ['CHECKPOINT_DIR']}{os.environ['CHECKPOINT_NAME']}*")
policy.load(os.environ['CHECKPOINT_DIR'], os.environ['CHECKPOINT_NAME'])
persist_event('eval_load_checkpoint', f"Loaded policy checkpoint {os.environ['CHECKPOINT_NAME']}*")


# Runs policy for X episodes and returns average reward
def evaluate_episode(policy):
    obs = env.reset(**env_step_kwargs)
    reset_frameskip(0)
    done = False
    total_reward = 0
    while not done:
        action = policy.select_action(prepare_model_observation(env))
        action = prepare_env_action(action)
        obs, reward, done, _ = env.step(action, **env_step_kwargs)
        
        # We don't use the custom rewards here because we want to evaluate our progress against the environment's reward.
        # obs_dict = env.get_state_desc()
        # custom_rewards = compute_rewards(obs_dict)
        # total_rewared += reward + sum(custom_rewards.values())

        total_reward += reward
    return total_reward


def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in tqdm(range(eval_episodes), desc="Evaluating policy", unit="episode"):
        avg_reward += evaluate_episode(policy)
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    
    return avg_reward


# Evaluate Policy
eval_reward = evaluate_policy(policy)

# Log evaluation history
persist_event('eval_completed', f"Evaluated policy @ checkpoint {os.environ['CHECKPOINT_NAME']}*, average reward: {eval_reward}")

# Persist evalutation timesteps to central database. (Why not?)
persist_timesteps(env.history())
env.reset_history()







