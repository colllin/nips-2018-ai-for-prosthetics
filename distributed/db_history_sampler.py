import uuid
import pandas as pd
import numpy as np

class DatabaseHistorySampler(object):
    def __init__(self, df_history, env_obs_history_to_model_obs_fn=None, env_obs_custom_reward_fn=None, env_obs_custom_done_fn=None):
        self.env_obs_history_to_model_obs_fn = env_obs_history_to_model_obs_fn
        self.env_obs_custom_reward_fn = env_obs_custom_reward_fn
        self.env_obs_custom_done_fn = env_obs_custom_done_fn
        
    def sample(self, batch_size):
        df_without_resets = self.df_history[~self.df_history['action'].isnull()]
        i_samples = df_without_resets.sample(batch_size).index
        
        # Get episodes corresponding to each sample
        episodes = [self.get_episode(self.df_history.at[i, 'episode_uuid']) for i in i_samples]
        # Slice episode to end at sampled step
        episodes = [ep[ep['i_step'] <= ep.at[i_sample, 'i_step']] for i_sample, ep in zip(i_samples, episodes)]
        
        # Collect actions
        A = [ep.at[i_sample, 'action'] for i_sample, ep in zip(i_samples, episodes)]
        
        # Collect `done`
        if self.env_obs_custom_done_fn != None:
            D = [ep.at[i_sample, 'done'] | self.env_obs_custom_done_fn(ep.at[i_sample, 'obs']) for i_sample, ep in zip(i_samples, episodes)]
        else:
            D = [ep.at[i_sample, 'done'] for i_sample, ep in zip(i_samples, episodes)]
        
        # Collect rewards
        if self.env_obs_custom_reward_fn != None:
            R = [self.env_obs_custom_reward_fn(ep.at[i_sample, 'obs']) for i_sample, ep in zip(i_samples, episodes)]
        else:
            R = [ep.at[i_sample, 'reward'] for i_sample, ep in zip(i_samples, episodes)]
            
        # Collect X, Y
        if self.env_obs_history_to_model_obs_fn != None:
            X = [self.env_obs_history_to_model_obs_fn(ep['obs'].iloc[:-1].tolist()) for ep in episodes]
            Y = [self.env_obs_history_to_model_obs_fn(ep['obs'].tolist()) for ep in episodes]
        else:
            X = [ep.iloc[-2]['obs'] for ep in episodes]
            Y = [ep.iloc[-1]['obs'] for ep in episodes]
            
        return np.array(X), np.array(Y), np.array(A), np.array(R), np.array(D)
        
    def get_episode(self, episode_uuid):
        return self.df_history[self.df_history['episode_uuid'] == episode_uuid]
