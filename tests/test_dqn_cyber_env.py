import os
import sys
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(directory))
sys.path.insert(0,desktop_path+'\ARM_IRL')
from agents.dqn import *
from envs.simpy_env.CyberWithChannelEnvSB_123 import CyberEnv
import random
import networkx as nx
import numpy as np
from envs.simpy_env.generate_network import create_network,draw_cyber_network,create_network2
from collections import namedtuple, deque 
import torch

G = create_network2()
env = CyberEnv(provided_graph=G,channelModel=True,envDebug=False, R2_qlimit=100, ch_bw = 2000,with_threat=True)

n_episodes = 10000
max_t = 30

eps_start = 1.0
eps_end = 0.01
eps_decay =0.996
scores = [] # list containing score from each episode
scores_window = deque(maxlen=100) # last 100 scores
eps = eps_start

# currently lets initialize the action_size to be limited based on the connectivity among the routers
possible_actions = 0
action_index_boundary = []
for routers in env.routers:
    possible_actions += len(routers.out)
    action_index_boundary.append(possible_actions)

print(possible_actions)
agent = Agent(state_size=env.observation_space.shape[0],action_size=possible_actions,seed=0)

print(action_index_boundary)

for i_episode in range(1, n_episodes+1):
    print ('EPISODE: {} =======>'.format(i_episode))
    state = env.reset()
    #print(state)
    score = 0
    for t in range(max_t):
        action = agent.act(state,eps)
        # map the action into specific dictionary in the environment action
        action_dict=[0,0]
        prev = 0
        for i,j in enumerate(action_index_boundary):
            if action >= j:
                prev = j
                continue
            else:
                action_dict[0] = i
                action_dict[1] = action - prev
                break
        
        #print(action_dict)
        next_state,reward,done,info = env.step(action_dict, result={})
        #print('State : {0}, Next-State : {1}, Reward : {2}, Done : {3}, Info :{4}'.format(state, next_state, reward, done, info))
        agent.step(state,action,reward,next_state,done)
        ## above step decides whether we will train(learn) the network
        ## actor (local_qnetwork) or we will fill the replay buffer
        ## if len replay buffer is equal to the batch size then we will
        ## train the network or otherwise we will add experience tuple in our 
        ## replay buffer.
        utilization_rates = next_state
        G_to_update = env.G
        for i,data in enumerate(G_to_update.edges(data=True)):
            G_to_update[data[0]][data[1]].update({'weight':utilization_rates[i]})
        env.G = G_to_update

        state = next_state
        score += reward
        if done:
            break
        scores_window.append(score) ## save the most recent score
        scores.append(score) ## sae the most recent score
        eps = max(eps*eps_decay,eps_end)## decrease the epsilon
        #print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)), end="")
    if i_episode %20==0:
        print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)))
        
    if np.mean(scores_window)>=0.0:
        print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100,
                                                                                    np.mean(scores_window)))
        torch.save(agent.qnetwork_local.state_dict(),'checkpoint.pth')
        break
    
