# ### Action hacking

# - binary activations?  Or several "bins"?
# - Frameskip?
# - Muscles must remain "active" for at least 10 frames once activated?  Randomized?
# - Limited number of muscles can fire at one time?
# - Handle prosthetic or not (strip activations of nonexistent muscles)

import numpy as np

action_state = {}

def reset_frameskip(n=8):
    action_state['n_frameskip'] = n
    action_state['frames_to_skip'] = 0  # Start at 0 so we don't skip the first model action.
    action_state['frameskip_action'] = None
    
def apply_frameskip(model_action):
    if action_state.get('frames_to_skip', 0) == 0:
        # Already skipped enough frames.  Reset counter & cache unskipped frame.
        action_state['frames_to_skip'] = np.random.randint(action_state.get('n_frameskip', 0) + 1)
        action_state['frameskip_action'] = model_action
    else:
        # Need to skip this frame.  Decrement the counter & apply the cached unskipped frame.
        action_state['frames_to_skip'] -= 1
        model_action = action_state.get('frameskip_action')
    return model_action


def prepare_env_action(model_action, bins=2):
    # model_action is a list of muscle activations

    # Frame skipping
    model_action = apply_frameskip(model_action)

    model_action = np.array(model_action)
    
    # "Bin" the muscle activations:
    model_action *= bins-1
    model_action = model_action.round()
    model_action /= bins-1

    # Clip    
    model_action = np.clip(model_action, 0, 1)
    
    return model_action.tolist()
    