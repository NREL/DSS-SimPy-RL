# testing open dss env
import os
import sys
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(directory))
sys.path.insert(0,desktop_path+'\ARM_IRL')
from envs.openDSSenv import openDSSenv
import opendssdirect as dss
import random

from agents.dqn import *
from collections import namedtuple, deque 
import torch

dss_data_dir = desktop_path+'\\ARM_IRL\\cases\\123Bus_Simple\\'
dss_master_file_dir = 'Redirect ' + dss_data_dir + 'IEEE123Master.dss'

dss.run_command(dss_master_file_dir)
circuit = dss.Circuit
#critical_loads_bus = ['58','59','99','100','88','93','94','78','48','50', '111','114', '37','39']
#critical_loads_bus = ['99','100','88','94']
critical_loads_bus = ['57','60']
capacitor_banks =['C83', 'C88a', 'C90b','C92c']
# switch from and two buses, with the first 6 are normally closed and the last two are normally open
switches = { 0: ['150r','149'], 1: ['13','152'], 2: ['18','135'], 3: ['60','160'], 4: ['97','197'], 5: ['61','61s'], 6: ['151','300'], 7: ['54','94'] }

switch_names =[]
for k,sw in enumerate(switches):
    switch_names.append('Sw'+str(k+1))

#line_faults = ['L55','L68', 'L58', 'L77', 'L45', 'L101', 'L41']
line_faults = ['L55']

env = openDSSenv(_dss = dss, _critical_loads=critical_loads_bus, _line_faults =line_faults, _switch_names = switch_names, _capacitor_banks = capacitor_banks)

n_episodes = 200
max_t = 100

eps_start = 1.0
eps_end = 0.01
eps_decay =0.996
scores = [] # list containing score from each episode
scores_window = deque(maxlen=100) # last 100 scores
eps = eps_start

agent = Agent(state_size=env.observation_spaces.shape[0],action_size=len(switch_names) - 1,seed=0)

for i_episode in range(1, n_episodes+1):
    print ('EPISODE: {} =======>'.format(i_episode))
    #print(env.reset())
    state = np.array(env.reset())
    print(state)
    score = 0
    for t in range(max_t):
        action = agent.act(state,eps)
        next_state,reward,done,info,_ = env.step(switch_names[action+1], result={})
        print('State : {0}, Next-State : {1}, Reward : {2}, Done : {3}, Info :{4}'.format(state, next_state, reward, done, info))
        agent.step(state,action,reward[0],np.array(next_state),done)
        state = np.array(next_state)
        score += reward[0]
        if done:
            print('Episode length '+str(t+1))
            break
        scores_window.append(score) ## save the most recent score
        scores.append(score) ## sae the most recent score
        eps = max(eps*eps_decay,eps_end)## decrease the epsilon
    #print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)), end="")
    if i_episode %20==0:
        print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)))
        
    if np.mean(scores_window)>=2.0:
        print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100,
                                                                                    np.mean(scores_window)))
        torch.save(agent.qnetwork_local.state_dict(),'checkpoint.pth')
        break
    




