# ### Observation Hacking

# - Rewrite all joint_pos, body_pos to be relative to mass_center_pos
# - Subtract mass_center_vel and mass_center_acc from joint_vel, body_vel, joint_acc, body_acc?
# - Either compute jounce/snap, or pass multiple timesteps, or just pass acceleration from past 3 timesteps?

# Initial Env Observation:
# ```
# {
#     'joint_pos': {
#         'ground_pelvis': [0.0, 0.0, 0.0, 0.0, 0.94, 0.0],
#         'hip_r': [0.0, 0.0, 0.0],
#         'knee_r': [0.0],
#         'ankle_r': [0.0],
#         'hip_l': [0.0, 0.0, 0.0],
#         'knee_l': [0.0],
#         'ankle_l': [0.0],
#         'subtalar_l': [],
#         'mtp_l': [],
#         'back': [-0.0872665],
#         'back_0': []
#     },
#     'joint_vel': {
#         'ground_pelvis': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         'hip_r': [0.0, 0.0, 0.0],
#         'knee_r': [0.0],
#         'ankle_r': [0.0],
#         'hip_l': [0.0, 0.0, 0.0],
#         'knee_l': [0.0],
#         'ankle_l': [0.0],
#         'subtalar_l': [],
#         'mtp_l': [],
#         'back': [0.0],
#         'back_0': []
#     },
#     'joint_acc': {
#         'ground_pelvis': [34.07237489546962, 3.219284560937942, 0.021285761200362296, 13.997154494145377, 0.8655672359505977, -0.6156967622871027],
#         'hip_r': [-194.74323476194263, -4.441803696780512, 1.5931700403370996e-14],
#         'knee_r': [305.46152469620915],
#         'ankle_r': [9636.363025843913],
#         'hip_l': [-208.86020665024324, 3.5702556374966354, -4.2521541843143495e-14],
#         'knee_l': [399.3192427973721],
#         'ankle_l': [809.4478175113452],
#         'subtalar_l': [],
#         'mtp_l': [],
#         'back': [-2.3092638912203256e-14],
#         'back_0': []
#     },
#     'body_pos': {
#         'pelvis': [0.0, 0.94, 0.0],
#         'femur_r': [-0.0707, 0.8738999999999999, 0.0835],
#         'pros_tibia_r': [-0.07519985651753601, 0.47807930355164957, 0.0835],
#         'pros_foot_r': [-0.07519985651753601, 0.04807930355164958, 0.0835],
#         'femur_l': [-0.0707, 0.8738999999999999, -0.0835],
#         'tibia_l': [-0.07519985651753601, 0.47807930355164957, -0.0835],
#         'talus_l': [-0.07519985651753601, 0.04807930355164958, -0.0835],
#         'calcn_l': [-0.123969856517536, 0.006129303551649576, -0.09142],
#         'toes_l': [0.05483014348246398, 0.004129303551649576, -0.0925],
#         'torso': [-0.1007, 1.0214999999999999, 0.0],
#         'head': [-0.052764320996907754, 1.5694070821576522, 0.0]
#     },
#     'body_vel': {
#         'pelvis': [0.0, 0.0, 0.0],
#         'femur_r': [0.0, 0.0, 0.0],
#         'pros_tibia_r': [0.0, 0.0, 0.0],
#         'pros_foot_r': [0.0, 0.0, 0.0],
#         'femur_l': [0.0, 0.0, 0.0],
#         'tibia_l': [0.0, 0.0, 0.0],
#         'talus_l': [0.0, 0.0, 0.0],
#         'calcn_l': [0.0, 0.0, 0.0],
#         'toes_l': [0.0, 0.0, 0.0],
#         'torso': [0.0, 0.0, 0.0],
#         'head': [0.0, 0.0, 0.0]
#     },
#     'body_acc': {
#         'pelvis': [13.997154494145377, 0.8655672359505977, -0.6156967622871027],
#         'femur_r': [16.25111583579615, -1.812159929997423, -0.826986568448235],
#         'pros_tibia_r': [-49.070641675940735, 0.12065763836075294, -0.34299240980632545],
#         'pros_foot_r': [13.18934420084581, 0.12065763836075294, 0.18269081860597952],
#         'femur_l': [16.24756111367569, -1.2745394083207864, -0.826986568448235],
#         'tibia_l': [-55.19198970892064, 1.093538716356541, -0.6879691696202773],
#         'talus_l': [41.356517039396714, 1.093538716356541, -0.537051606700039],
#         'calcn_l': [84.73177709400595, -49.336407951145645, -0.5212902634646581],
#         'toes_l': [86.79971256249173, 135.5386990655368, -0.5243942154141731],
#         'torso': [11.220255940164602, -2.565520916023193, -0.35118159441778396],
#         'head': [-7.448239570993795, -0.9322384901609437, 1.4116668685846663]
#     },
#     'body_pos_rot': {
#         'pelvis': [-0.0, 0.0, -0.0],
#         'femur_r': [-0.0, 0.0, -0.0],
#         'pros_tibia_r': [-0.0, 0.0, -0.0],
#         'pros_foot_r': [-0.0, 0.0, -0.0],
#         'femur_l': [-0.0, 0.0, -0.0],
#         'tibia_l': [-0.0, 0.0, -0.0],
#         'talus_l': [-0.0, 0.0, -0.0],
#         'calcn_l': [-0.0, 0.0, -0.0],
#         'toes_l': [-0.0, 0.0, -0.0],
#         'torso': [-0.0, 0.0, -0.0872665],
#         'head': [-0.0, 0.0, -0.0872665]
#     },
#     'body_vel_rot': {
#         'pelvis': [0.0, 0.0, 0.0],
#         'femur_r': [0.0, 0.0, 0.0],
#         'pros_tibia_r': [0.0, 0.0, 0.0],
#         'pros_foot_r': [0.0, 0.0, 0.0],
#         'femur_l': [0.0, 0.0, 0.0],
#         'tibia_l': [0.0, 0.0, 0.0],
#         'talus_l': [0.0, 0.0, 0.0],
#         'calcn_l': [0.0, 0.0, 0.0],
#         'toes_l': [0.0, 0.0, 0.0],
#         'torso': [0.0, 0.0, 0.0],
#         'head': [0.0, 0.0, 0.0]
#     },
#     'body_acc_rot': {
#         'pelvis': [3.219284560937942, 0.021285761200362296, 34.07237489546962],
#         'femur_r': [-1.2225191358425698, 0.021285761200378228, -160.670859866473],
#         'pros_tibia_r': [-1.2225191358425698, 0.021285761200378228, 144.79066482973616],
#         'pros_foot_r': [-1.2225191358425698, 0.021285761200378228, 9781.15369067365],
#         'femur_l': [-0.35097107655869353, 0.021285761200404818, -174.7878317547736],
#         'tibia_l': [-0.35097107655869353, 0.021285761200404818, 224.5314110425985],
#         'talus_l': [-0.35097107655869353, 0.021285761200404818, 1033.9792285539438],
#         'calcn_l': [-0.35097107655869353, 0.021285761200404818, 1033.9792285539438],
#         'toes_l': [-0.35097107655869353, 0.021285761200404818, 1033.9792285539438],
#         'torso': [3.219284560937942, 0.021285761200362296, 34.0723748954696],
#         'head': [3.219284560937942, 0.021285761200362296, 34.0723748954696]
#     },
#     'forces': {
#         'abd_r': [219.6613927253564],
#         'add_r': [144.87433100305103],
#         'hamstrings_r': [194.30030504346755],
#         'bifemsh_r': [42.728811234363775],
#         'glut_max_r': [171.7873509605573],
#         'iliopsoas_r': [158.01207984383657],
#         'rect_fem_r': [99.0329705435046],
#         'vasti_r': [436.79388413623326],
#         'abd_l': [219.6613927253564],
#         'add_l': [144.87433100305103],
#         'hamstrings_l': [194.30030504346755],
#         'bifemsh_l': [42.728811234363775],
#         'glut_max_l': [171.7873509605573],
#         'iliopsoas_l': [158.01207984383657],
#         'rect_fem_l': [99.0329705435046],
#         'vasti_l': [436.79388413623326],
#         'gastroc_l': [273.0178325689043],
#         'soleus_l': [370.0059951709156],
#         'tib_ant_l': [104.05059952034294],
#         'ankleSpring': [-0.0],
#         'pros_foot_r_0': [-1.3573320551122304e-12, -388.7553514927188, 0.0, 32.46107184964201, -1.1333722660187127e-13, 3.726164264482634, 1.3573320551122304e-12, 388.7553514927188, 0.0, 0.0, 0.0, 25.50818238819416, 1.3573320551122304e-12, 388.7553514927188, 0.0, 0.0, 0.0, 25.50818238819416],
#         'foot_l': [-1.7615592933859115e-12, -504.53063397142085, 0.0, -46.45915575255036, 1.622112753284362e-13, -4.943148065393853, 6.786660275561278e-13, 194.37767574636297, 0.0, 0.0, 0.0, 5.831330272390875, 1.0828932658297836e-12, 310.1529582250579, 0.0, 0.0, 0.0, 6.203059164501164, 1.0828932658297836e-12, 310.1529582250579, 0.0, 0.0, 0.0, 6.203059164501164],
#         'HipLimit_r': [0.0, 0.0],
#         'HipLimit_l': [0.0, 0.0],
#         'KneeLimit_r': [-0.0, 0.0],
#         'KneeLimit_l': [-0.0, 0.0],
#         'AnkleLimit_r': [0.0, 0.0],
#         'AnkleLimit_l': [0.0, 0.0],
#         'HipAddLimit_r': [0.0, 0.0],
#         'HipAddLimit_l': [0.0, 0.0]
#     },
#     'muscles': {
#         'abd_r': {
#             'activation': 0.05,
#             'fiber_length': 0.07752306863700548,
#             'fiber_velocity': 1.1700156898117815e-13,
#             'fiber_force': 219.6613927253564
#         },
#         'add_r': {
#             'activation': 0.05,
#             'fiber_length': 0.05526137592854144,
#             'fiber_velocity': 5.531257930905764e-11,
#             'fiber_force': 146.25768888087705
#         },
#         'hamstrings_r': {
#             'activation': 0.05,
#             'fiber_length': 0.06355896214015513,
#             'fiber_velocity': 2.1056261406660054e-14,
#             'fiber_force': 202.45627069225887
#         },
#         'bifemsh_r': {
#             'activation': 0.05,
#             'fiber_length': 0.13434264681417835,
#             'fiber_velocity': 9.542198984660805e-17,
#             'fiber_force': 45.09919197278584
#         },
#         'glut_max_r': {
#             'activation': 0.05,
#             'fiber_length': 0.16084824667171801,
#             'fiber_velocity': 1.0181982508865008e-12,
#             'fiber_force': 171.7873509605573
#         },
#         'iliopsoas_r': {
#             'activation': 0.05,
#             'fiber_length': 0.13005768603600326,
#             'fiber_velocity': 3.347183497651294e-11,
#             'fiber_force': 159.26525950285387
#         },
#         'rect_fem_r': {
#             'activation': 0.05,
#             'fiber_length': 0.06027044615978444,
#             'fiber_velocity': 2.2362024955832438e-15,
#             'fiber_force': 99.63652479982161
#         },
#         'vasti_r': {
#             'activation': 0.05,
#             'fiber_length': 0.07890756873654925,
#             'fiber_velocity': 6.168233828156989e-15,
#             'fiber_force': 437.7385693769557
#         },
#         'abd_l': {
#             'activation': 0.05,
#             'fiber_length': 0.07752306863700548,
#             'fiber_velocity': 1.1700156898117815e-13,
#             'fiber_force': 219.6613927253564
#         },
#         'add_l': {
#             'activation': 0.05,
#             'fiber_length': 0.05526137592854144,
#             'fiber_velocity': 5.531257930905764e-11,
#             'fiber_force': 146.25768888087705
#         },
#         'hamstrings_l': {
#             'activation': 0.05,
#             'fiber_length': 0.06355896214015513,
#             'fiber_velocity': 2.1056261406660054e-14,
#             'fiber_force': 202.45627069225887
#         },
#         'bifemsh_l': {
#             'activation': 0.05,
#             'fiber_length': 0.13434264681417835,
#             'fiber_velocity': 9.542198984660805e-17,
#             'fiber_force': 45.09919197278584
#         },
#         'glut_max_l': {
#             'activation': 0.05,
#             'fiber_length': 0.16084824667171801,
#             'fiber_velocity': 1.0181982508865008e-12,
#             'fiber_force': 171.7873509605573
#         },
#         'iliopsoas_l': {
#             'activation': 0.05,
#             'fiber_length': 0.13005768603600326,
#             'fiber_velocity': 3.347183497651294e-11,
#             'fiber_force': 159.26525950285387
#         },
#         'rect_fem_l': {
#             'activation': 0.05,
#             'fiber_length': 0.06027044615978444,
#             'fiber_velocity': 2.2362024955832438e-15,
#             'fiber_force': 99.63652479982161
#         },
#         'vasti_l': {
#             'activation': 0.05,
#             'fiber_length': 0.07890756873654925,
#             'fiber_velocity': 6.168233828156989e-15,
#             'fiber_force': 437.7385693769557
#         },
#         'gastroc_l': {
#             'activation': 0.05,
#             'fiber_length': 0.05720257668702345,
#             'fiber_velocity': 5.718949639274886e-14,
#             'fiber_force': 282.79456495087237
#         },
#         'soleus_l': {
#             'activation': 0.05,
#             'fiber_length': 0.04494814124106819,
#             'fiber_velocity': 3.4120643802478774e-10,
#             'fiber_force': 406.4161372187392
#         },
#         'tib_ant_l': {
#             'activation': 0.05,
#             'fiber_length': 0.06288000983990409,
#             'fiber_velocity': 8.525642140971546e-14,
#             'fiber_force': 104.5158690359632
#         }
#     },
#     'markers': {},
#     'misc': {
#         'mass_center_pos': [-0.08466565561225976, 0.9952730567231536, -0.003576087446004414],
#         'mass_center_vel': [0.0, 0.0, 0.0],
#         'mass_center_acc': [4.4008799039045146e-14, 2.570832209970726, -1.0237645330147003e-15]
#     }
# }
# ```


import numpy as np

# Copied & modified from https://github.com/stanfordnmbl/osim-rl/blob/master/osim/env/osim.py#L452
def single_step_env_obs_to_model_obs(env_obs, target_vel=[3,0,0], exclude_lower_order_values=False):
    env_obs = env_obs.copy()
    has_prosthetic = 'pros_foot_r' in env_obs['body_pos']

    target_vel_x = target_vel[0]
    target_vel_z = target_vel[2]
    eps = 1e-8

    frame = {
        'pos': np.array(env_obs['misc']['mass_center_pos']),
        'vel': np.array(env_obs['misc']['mass_center_vel']),
        'acc': np.array(env_obs['misc']['mass_center_acc']),
    }

    # Transform reference frame from 0,0,0 to center of mass:
    for k, pos in env_obs['body_pos'].items():
        env_obs['body_pos'][k] = list(np.array(pos) - frame['pos'])
    for k, vel in env_obs['body_vel'].items():
        env_obs['body_vel'][k] = list(np.array(vel) - frame['vel'])
    for k, acc in env_obs['body_acc'].items():
        env_obs['body_acc'][k] = list(np.array(acc) - frame['acc'])

    # Normalize body vel/acc based on center of mass vel/acc:
#     for k, vel in env_obs['body_vel'].items():
#         env_obs['body_vel'][k] = list(np.array(vel) / (frame['vel'] + eps))
#     for k, acc in env_obs['body_acc'].items():
#         env_obs['body_acc'][k] = list(np.array(acc) / (frame['acc'] + eps))

    # Collect observation vector
    lower_order = []
    highest_order = []    
    for body_part in ["head","torso","pelvis","femur_l","femur_r","tibia_l","tibia_r","pros_tibia_r","talus_l","talus_r","toes_l","toes_r","pros_foot_r","calcn_l","calcn_r"]:
        if has_prosthetic and body_part in ["toes_r","tibia_r","talus_r","calcn_r"]:
            lower_order += [0] * 12
            highest_order += [0] * 6
            continue
        if not has_prosthetic and body_part in ["pros_foot_r","pros_tibia_r"]:
            lower_order += [0] * 12
            highest_order += [0] * 6
            continue
        lower_order += env_obs["body_pos"][body_part][0:2]
        lower_order += env_obs["body_vel"][body_part][0:2]
        highest_order += env_obs["body_acc"][body_part][0:2]
        lower_order += env_obs["body_pos_rot"][body_part][0:2]
        lower_order += env_obs["body_vel_rot"][body_part][0:2]
        highest_order += env_obs["body_acc_rot"][body_part][0:2]

    for joint in ["ground_pelvis","ankle_l","ankle_r","back","hip_l","hip_r","knee_l","knee_r"]:
        lower_order += env_obs["joint_pos"][joint]
        lower_order += env_obs["joint_vel"][joint]
        highest_order += env_obs["joint_acc"][joint]

    for muscle in sorted(env_obs["muscles"].keys()):
        # Possible osim-rl bug: It appears that muscle values are ommitted when they are 0.
        highest_order += [env_obs["muscles"][muscle].get("activation", 0.)]
        highest_order += [env_obs["muscles"][muscle].get("fiber_length", 0.)]
        highest_order += [env_obs["muscles"][muscle].get("fiber_velocity", 0.)]
        highest_order += [env_obs["muscles"][muscle].get("fiber_force", 0.)] 

    for force in ['abd_l', 'add_l', 'hamstrings_l', 'bifemsh_l', 'glut_max_l', 'iliopsoas_l', 'rect_fem_l', 'vasti_l', 'gastroc_l', 'soleus_l', 'tib_ant_l', 'ankleSpring', 'pros_foot_r_0', 'foot_l', 'HipLimit_l', 'KneeLimit_l', 'AnkleLimit_l', 'HipAddLimit_l']:
        highest_order += env_obs['forces'][force]
        if not '_l' in force:
            continue
        force = force.replace('_l', '_r')
        if has_prosthetic:
            if force in ['gastroc_r', 'soleus_r', 'tib_ant_r']:
                highest_order += [0]
                continue
            if force in ['foot_r']:
                highest_order += [0] * 24
                continue
        else:
            if force in ['pros_foot_r_0']:
                highest_order += [0] * 18
                continue
        highest_order += env_obs['forces'][force.replace('_l', '_r')]

    # Center of mass
    lower_order += list(frame['pos'])
    lower_order += list(frame['vel'])
    highest_order += list(frame['acc'])

    # Target velocity
    highest_order += [frame['vel'][0] - target_vel_x, frame['vel'][2] - target_vel_z]

    result = highest_order
    if not exclude_lower_order_values:
        result = lower_order + result
    return result

def env_obs_history_to_model_obs(env_obs_history):
    env_obs_history = env_obs_history[-4:]
    # Duplicate first env_obs to ensure we have at least 4 steps of history.
    env_obs_history = env_obs_history[:1] * (4 - len(env_obs_history)) + env_obs_history

    model_obs_steps = [single_step_env_obs_to_model_obs(env_obs_history[-1])]
    model_obs_steps += [single_step_env_obs_to_model_obs(env_obs, exclude_lower_order_values=True) for env_obs in env_obs_history[:-1]][::-1]
    return np.concatenate(model_obs_steps)

def prepare_model_observation(env):
    df_history = env.history(current_episode_only=True)
    model_obs = env_obs_history_to_model_obs(df_history['obs'].tolist())
    return model_obs

