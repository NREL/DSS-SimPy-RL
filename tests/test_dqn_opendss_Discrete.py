# testing open dss env
import os
import sys
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(directory))
sys.path.insert(0,desktop_path+'\ARM_IRL')
from envs.openDSSenvSB_DiscreteSpace import openDSSenv
import opendssdirect as dss
import random

from agents.dqn import *
from collections import namedtuple, deque 
import torch
import statistics

dss_data_dir = desktop_path+'\\ARM_IRL\\cases\\123Bus_SimpleMod\\'
dss_master_file_dir = 'Redirect ' + dss_data_dir + 'IEEE123Master.dss'

dss.run_command(dss_master_file_dir)

circuit = dss.Circuit
critical_loads_bus = ['58','59','99','100','88','93','94','78','48','50', '111','114', '37','39']
capacitor_banks =['C83', 'C88a', 'C90b','C92c']

# switch from and two buses, with the first 6 are normally closed and the last two are normally open
switches = { 0: ['150r','149'], 1: ['13','152'], 2: ['18','135'], 3: ['60','160'], 4: ['97','197'], 5: ['61','61s'], 6: ['151','300'], 7: ['54','94'] }

switch_names =[]
for k,sw in enumerate(switches):
    switch_names.append('Sw'+str(k+1))

line_faults = ['L55', 'L58', 'L77','L68','L45', 'L101','L41']

ders = ['35','48','64','78','95','108']
env = openDSSenv(_dss = dss, _critical_loads=critical_loads_bus, _line_faults =line_faults, _switch_names = switch_names, _capacitor_banks = capacitor_banks,load_ub=11,_ders=ders)
agg_episode_len = []
env.contingency = 4
success = 0

max_t = 100
for i_episode in range(1, 100):
    state = obs = env.reset()
    done = False
    episode_length = 0
    episodic_reward = 0
    switch_selected = []
    ctr=0
    while not done and ctr < max_t:
        ctr+=1
        #print(switch_names[0:])
        action = random.choice(switch_names[0:])
        next_state,reward,done,info,_ = env.step(action, result={})
        #print('State : {0}, Next-State : {1}, Reward : {2}, Done : {3}, Info :{4}'.format(state, next_state, reward, done, info))
        if action in switch_selected:
            continue
        else:
            switch_selected.append(action)
        next_state, reward, done,info,_ = env.step(action, result={})
    if ctr < max_t:
        success+=1
        agg_episode_len.append(ctr)
    print('Episode Len {0}'.format(ctr))
print('Case: Contingency: {0}, avg episode len: {1}, Success rate: {2}'.format(env.contingency, statistics.mean(agg_episode_len), success/100))


n_episodes = 5000


eps_start = 1.0
eps_end = 0.01
eps_decay =0.996
scores = [] # list containing score from each episode
scores_window = deque(maxlen=100) # last 100 scores
eps = eps_start

agent = Agent(state_size=env.observation_space.shape[0],action_size=len(switch_names) - 1,seed=0)

for i_episode in range(1, n_episodes+1):
    #print ('EPISODE: {} =======>'.format(i_episode))
    state = env.reset()
    #print(state)
    score = 0
    for t in range(max_t):
        action = agent.act(state,eps)
        next_state,reward,done,info,_ = env.step(action, result={})
        #print('State : {0}, Next-State : {1}, Reward : {2}, Done : {3}, Info :{4}'.format(state, next_state, reward, done, info))
        #print('Reward : {0}'.format(reward))
        agent.step(state,action,reward,next_state,done)
        state = next_state
        score += reward
        if done:
            #print('Episode length '+str(t+1))
            break
        eps = max(eps*eps_decay,eps_end)## decrease the epsilon
    scores_window.append(score) ## save the most recent score
    scores.append(score) ## sae the most recent score
        
    #print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)), end="")
    if i_episode %100==0:
        print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)))
        
    if np.mean(scores_window)>=0.0:
        print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100,
                                                                                    np.mean(scores_window)))
        torch.save(agent.qnetwork_local.state_dict(),'checkpoint.pth')
        break
    

testing = 5
#model = PPO.load("a2c_opendss_disc_lf", env=env)
for k in range(testing): 
    print('Test '+str(k+1))
    acc_len = 0
    test_episode = 100
    max_t=100
    acc_reward = 0
    for i in range(test_episode):
        state = obs = env.reset()
        #print('Episode ===========================> {0}'.format(i+1))
        done = False
        episodic_reward=0
        episode_len = 0
        switch_selected = []
        ctr = 0
        while not done and ctr < max_t:
            ctr+=1
            action = agent.act(obs)
            #print(action,obs)
            if action in switch_selected:
                continue
            else:
                switch_selected.append(action)
            obs, reward, done, info,_ = env.step(action,result={})
            #print('obs {0} reward {1} done {2} '.format(obs,reward,done))
            episodic_reward+=reward
            episode_len+=1
        #print('Episode Length ',ctr)
        #env.render()
        if ctr < max_t:
            success+=1
            agg_episode_len.append(ctr)
            acc_reward+=episodic_reward
    print('Average Episode Length After Training : {0} Avg Reward : {1}'.format(statistics.mean(agg_episode_len),acc_reward/test_episode))


