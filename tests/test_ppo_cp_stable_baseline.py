import os
import sys
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(directory))
sys.path.insert(0,desktop_path+'\ARM_IRL')
import gym
import numpy as np
from envs.simpy_env.CyberWithChannelEnvSB_123 import CyberEnv
from envs.openDSSenvSB_DiscreteSpace import openDSSenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from envs.simpy_env.generate_network import create_network,draw_cyber_network,create_network2
from envs.simpy_dss.CPEnv_DiscreteDSS_RtrDropRate import CyberPhysicalEnvMT,CyberPhysicalMapping
import random
import networkx as nx
import opendssdirect as dss

bi, li = CyberPhysicalMapping()
comp_zones = bi
for i,(k,v) in enumerate(li.items()):
    comp_zones[k] = bi[v[0]]

comp_zones['C83'] = bi['83']
comp_zones['C88a'] = bi['88']
comp_zones['C90b'] = bi['90']
comp_zones['C92c'] = bi['92']

# Create the vectorized environment
G = create_network2()
cenv = CyberEnv(provided_graph=G, channelModel=False,envDebug=False, R2_qlimit=130, ch_bw = 2500,with_threat=False,comp_zones=comp_zones)

# Stable Baselines provides you with make_vec_env() helper
# which does exactly the previous steps for you.
# You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
# env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

# Create the Physical Network
dss_data_dir = desktop_path+'\\ARM_IRL\\cases\\123Bus_SimpleMod\\'
dss_master_file_dir = 'Redirect ' + dss_data_dir + 'IEEE123Master.dss'

dss.run_command(dss_master_file_dir)
circuit = dss.Circuit
critical_loads_bus = ['58','59','99','100','88','93','94','78','48','50', '111','114', '37','39']
#critical_loads_bus = ['57','60']
capacitor_banks =['C83', 'C88a', 'C90b','C92c']
# switch from and two buses, with the first 6 are normally closed and the last two are normally open
switches = { 0: ['150r','149'], 1: ['13','152'], 2: ['18','135'], 3: ['60','160'], 4: ['97','197'], 5: ['61','61s'], 6: ['151','300'], 7: ['54','94'] }

switch_names =[]
for k,sw in enumerate(switches):
    switch_names.append('Sw'+str(k+1))

line_faults = ['L55','L68', 'L58', 'L77', 'L45', 'L101', 'L41']
#line_faults = ['L55']

penv = openDSSenv(_dss = dss, _critical_loads=critical_loads_bus, _line_faults =line_faults, _switch_names = switch_names, _capacitor_banks = capacitor_banks)

cyber_phy_env= CyberPhysicalEnvMT(cenv, penv,comp_zones)

test_episode = 100
max_t=100
acc_len = 0
acc_reward = 0
for i in range(test_episode):
    state = obs = cyber_phy_env.reset()
    done = False
    episode_length = 0
    episodic_reward = 0
    actions = []
    while not done and episode_length < max_t:
        phy_action = random.choice(switch_names[0:])
        router_id = random.randint(0,cenv.deviceCount-1)
                    
        # currently random:  to implement  get the next hop from the shortest path algorithm
        rnd_action_index = random.randint(0, len(cenv.routers[router_id].out)-1)

        #shortest_path_action_index = nx.single_source_shortest_path(env.G, router_id)['PS'][1]
        path_to_receiver =nx.single_source_shortest_path(cenv.G, router_id)['PS']

        shortest_path_action_index = path_to_receiver[1]

        #rnd_action = {'device':router_id, 'next_hop':rnd_action_index}
        rnd_action = [router_id,rnd_action_index]

        if shortest_path_action_index != 'PS' and False:
            rtr_id = 'R'+str(shortest_path_action_index)
            rtr_ix = [ix for (ix,item) in enumerate(cenv.routers[router_id].out) if item.id == rtr_id][0]
            #action = {'device':router_id, 'next_hop':rtr_ix}
            action = [router_id,rtr_ix]
        else:
            action = rnd_action
        #print(action)
        actions.append(action)
        action.append(list(cyber_phy_env.map_sw.values()).index(phy_action))
        actions = action
        next_state, reward, done, info = cyber_phy_env.step(actions)
        episodic_reward+=reward
        #print('obs {0} reward {1} done {2} '.format(obs,reward,done))
        episode_length+=1
    #print('Episode Length ',episode_length)
    acc_len+=episode_length
    acc_reward +=episodic_reward
print('Average Episode Length Before Training : {0}, Avg Reward : {1}'.format(acc_len/test_episode, acc_reward/test_episode))
    #env.render()

model = PPO('MlpPolicy',
            env=cyber_phy_env,
            seed=0,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.003,
            n_epochs=10,
            n_steps=64,)
model.learn(total_timesteps=1000000)
acc_len = 0
test_episode = 100
max_t=100
acc_reward = 0
for i in range(test_episode):
    state = obs = cyber_phy_env.reset()
    done = False
    episode_length = 0
    episodic_reward=0
    while not done and episode_length < max_t:
        action, _states = model.predict(obs)
        print(action)
        obs, reward, done, info = cyber_phy_env.step(action)
        #print('obs {0} reward {1} done {2} '.format(obs,reward,done))
        episodic_reward+=reward
        episode_length+=1
    #print('Episode Length ',episode_length)
    #env.render()
    acc_len+=episode_length
    acc_reward+=episodic_reward
print('Average Episode Length After Training : {0} Avg Reward : {1}'.format(acc_len/test_episode,acc_reward/test_episode))