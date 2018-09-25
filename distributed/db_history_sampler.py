import uuid
import pandas as pd
import numpy as np
from .database import sample_timesteps

class DatabaseHistorySampler(object):
    def __init__(self, env_obs_history_to_model_obs_fn=None, n_obs_history=0, env_obs_custom_reward_fn=None, env_obs_custom_done_fn=None):
        self.env_obs_history_to_model_obs_fn = env_obs_history_to_model_obs_fn
        # Aggregate 1 extra due to needing `n_obs_history` timesteps of history for both the before obs and after obs.
        self.aggregate_obs_history = 1 + n_obs_history
        self.env_obs_custom_reward_fn = env_obs_custom_reward_fn
        self.env_obs_custom_done_fn = env_obs_custom_done_fn
        
    def sample(self, batch_size):
        docs_with_history = sample_timesteps(n=batch_size, n_obs_history=self.aggregate_obs_history)
        
        # Collect actions
        A = [d['action'] for d in docs_with_history]
        
        # Collect `done`
        if self.env_obs_custom_done_fn != None:
            D = [d['done'] | self.env_obs_custom_done_fn(d['obs']) for d in docs_with_history]
        else:
            D = [d['done'] for d in docs_with_history]
        
        # Collect rewards
        if self.env_obs_custom_reward_fn != None:
            R = [self.env_obs_custom_reward_fn(d['obs']) for d in docs_with_history]
        else:
            R = [d['reward'] for d in docs_with_history]
            
        # Collect X, Y
        if self.env_obs_history_to_model_obs_fn != None:
            obslist = lambda d: [d[f'obs_t-{i}'] for i in range(self.aggregate_obs_history,0,-1) if f'obs_t-{i}' in d] + [d['obs']]
            X = [self.env_obs_history_to_model_obs_fn(obslist(d)[:-1]) for d in docs_with_history]
            Y = [self.env_obs_history_to_model_obs_fn(obslist(d)[1:]) for d in docs_with_history]
        else:
            X = [d['obs_t-1'] for d in docs_with_history]
            Y = [d['obs'] for d in docs_with_history]
            
        return np.array(X), np.array(Y), np.array(A), np.array(R), np.array(D)
        