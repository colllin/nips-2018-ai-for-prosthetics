# ### Reward Hacking

# - Survival reward
# - Lean forward reward (to avod models which had torso too upright) (may need to be based on speed)
# - Reward for minimizing sideways velocity?
# - Reward for minimizing vertical velocity (COG)?

import numpy as np


def env_obs_to_custom_reward(obs):
    if type(obs) != dict:
        raise ValueError('obs must be a dict (project=False)')

    target_vel_x = 3
    target_vel_z = 0
    eps = 1e-8

    target_vel_theta = -np.arctan(target_vel_z/(target_vel_x+eps)) if target_vel_x >= 0 else np.pi - np.arctan(target_vel_z/(target_vel_x+eps))

    # Parabolic reward/penalty for tracking a target value within some tolerance. 
    # Unit reward: 1 at target value, 0 at `tolerance` distance from target value, negative outside of `tolerance`.
    # Multiply by desired scale factor / magnitude. Don't multiply by too large of a scale factor — it also amplifies the slope.
    # Make the tolerance bigger than you think — it makes the slope more gradual / less severe.
    # `tolerance`: reward is positive if within `tolerance` of target value, else negative.
    val_diff_reward = lambda val, target, tolerance: (1 - ((val - target)/tolerance)**2)
    def radians_diff_wrapped(a1, a2):
        # Make both angles within (-2pi, 2pi)
        a1, a2 = a1 % (2*np.pi), a2 % (2*np.pi)
        # Make both angles positive -- within [0, 2pi)
        a1 = a1 + 2*np.pi if a1 < 0 else a1
        a2 = a2 + 2*np.pi if a2 < 0 else a2
        # Make a1 the smaller angle
        a1, a2 = min(a1, a2), max(a1, a2)
        # Make sure a1 is within pi of a2 — two angles can't be greater than pi apart in relative (wrapped) sense.
        a1 = a1 + 2*np.pi if a2 - a1 > np.pi else a1
        return np.fabs(a2 - a1)
    
    avg_knee_joint_pos = (obs['joint_pos']['knee_l'][0] + obs['joint_pos']['knee_r'][0]) / 2
    target_avg_knee_joint_pos = 60*np.pi/180
    
    rewards = {
        'survival': 100,
        'target_velocity_x': 3 * val_diff_reward(obs['misc']['mass_center_vel'][0], target_vel_x, 5), # 3 at target velocity, 0 at 3m/s off-target, then negative
        'target_velocity_z': 3 * val_diff_reward(obs['misc']['mass_center_vel'][2], target_vel_z, 5), # 3 at target velocity, 0 at 3m/s off-target, then negative
        'head_velocity_x': 2 * val_diff_reward(obs['body_vel']['head'][0], target_vel_x, 5), # 2 at target velocity, 0 at 3m/s off-target, then negative
        'head_velocity_z': 2 * val_diff_reward(obs['body_vel']['head'][2], target_vel_z, 5), # 2 at target velocity, 0 at 3m/s off-target, then negative
        'lean_forward_x': 5 * val_diff_reward(obs['body_pos']['head'][0] - obs['body_pos']['pelvis'][0], .1 * target_vel_x, .4), # head in front of pelvis from perspective of velocity vector
        'lean_forward_z': 5 * val_diff_reward(obs['body_pos']['head'][2] - obs['body_pos']['pelvis'][2], .1 * target_vel_z, .4), # head in front of pelvis from perspective of velocity vector
        'hips_squared': 5 * val_diff_reward(radians_diff_wrapped(target_vel_theta, obs['body_pos_rot']['pelvis'][1]), 0, np.pi),
        'knee_bent_l': 5 * val_diff_reward(obs['joint_pos']['knee_l'][0], target_avg_knee_joint_pos, np.pi), # goal range of roughly [0,120] degrees
        'knee_bent_r': 5 * val_diff_reward(obs['joint_pos']['knee_r'][0], target_avg_knee_joint_pos, np.pi), # goal range of roughly [0,120] degrees
        'low_y_vel_pelvis': 5 * val_diff_reward(obs['body_vel']['pelvis'][1], 0, 1),
        'low_y_vel_head': 5 * val_diff_reward(obs['body_vel']['head'][1], 0, 1),
        'low_y_vel_toes_l': 5 * val_diff_reward(obs['body_vel']['toes_l'][1], 0, 1),
        'low_y_vel_pros_foot_r': 5 * val_diff_reward(obs['body_vel']['pros_foot_r'][1], 0, 1),
        'knees_opposite_joint_vel': 0 if avg_knee_joint_pos < (target_avg_knee_joint_pos - 15*np.pi/180) else 3 * val_diff_reward(obs['joint_vel']['knee_l'][0], -obs['joint_vel']['knee_r'][0], np.pi), # The left knee should be opening when the right knee is closing, and vice versa
        'feet_behind_mass_x': 5 * val_diff_reward(obs['misc']['mass_center_pos'][0] - (obs['body_pos']['toes_l'][0] + obs['body_pos']['pros_foot_r'][0])/2, .1 * target_vel_x, .4),
        'feet_behind_mass_z': 5 * val_diff_reward(obs['misc']['mass_center_pos'][2] - (obs['body_pos']['toes_l'][2] + obs['body_pos']['pros_foot_r'][2])/2, .1 * target_vel_z, .4),
        'one_foot_off_ground': 0,
        'femurs_parallel': 0,
        'absolute_foot_velocity': 0, # should be 0 at ground, 2x body velocity above ground
        'forefoot_strike': 0,
    }    

#     if should_abort_episode(obs, custom_rewards=rewards):
#         rewards['abort_episode'] = -100

    return rewards
    