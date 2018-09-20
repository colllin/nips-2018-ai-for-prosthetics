import uuid
import pandas as pd
from osim.env import ProstheticsEnv

class ProstheticsEnvWithHistory(ProstheticsEnv):
    def __init__(self, *args, **kwargs):
        self.reset_history()
        super(ProstheticsEnvWithHistory, self).__init__(*args, **kwargs)
        
    def reset(self, *args, **kwargs):
        self.episode_id = str(uuid.uuid4())
        self.episode_step = 0
        obs = super(ProstheticsEnvWithHistory, self).reset(*args, **kwargs)
        self.append_history({
            'episode_uuid': self.episode_id,
            'i_step': self.episode_step,
            'action': None,
            'obs': self.get_state_desc(),
            'reward': 0,
            'done': False,
            'info': None,
        })
        return obs
    
    def step(self, action, *args, **kwargs):
        self.episode_step += 1
        obs,reward,done,info = super(ProstheticsEnvWithHistory, self).step(action, *args, **kwargs)
        self.append_history({
            'episode_uuid': self.episode_id,
            'i_step': self.episode_step,
            'action': action,
            'obs': self.get_state_desc(),
            'reward': reward,
            'done': done,
            'info': info,
        })       
        return obs,reward,done,info
    
    def append_history(self, history_item):
        self.df_history = self.df_history.append(history_item, ignore_index=True)
    
    def history(self, current_episode_only=False):
        if current_episode_only:
            return self.df_history[self.df_history['episode_uuid'] == self.episode_id]
        return self.df_history
    
    def reset_history(self):
        self.df_history = pd.DataFrame(columns=['episode_uuid','i_step','action','obs','reward','done','info'])

    def save_history(filename):
        pass
        
        
            
