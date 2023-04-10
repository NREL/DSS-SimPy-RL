import os
import sys
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(directory))
sys.path.insert(0,desktop_path+'\ARM_IRL')
from agents.dqn import *
from envs.simpy_env.CyberWithChannelEnvSB import CyberEnv
import random
import networkx as nx
import numpy as np
from envs.simpy_env.generate_network import create_network,draw_cyber_network
from collections import namedtuple, deque 
import torch

#G = create_network()
#env = CyberEnv(provided_graph=G)
channelModel=False

env = CyberEnv(channelModel=channelModel,envDebug=False, R2_qlimit=90, ch_bw = 500)

n_episodes = 5000
max_t = 100

eps_start = 1.0
eps_end = 0.01
eps_decay =0.996
scores = [] # list containing score from each episode
scores_window = deque(maxlen=100) # last 100 scores
eps = eps_start

track_ideal = deque(maxlen=100)

# currently lets initialize the action_size to be limited based on the connectivity among the routers
possible_actions = 0
action_index_boundary = []
for routers in env.routers:
    possible_actions += len(routers.out)
    action_index_boundary.append(possible_actions)

print(env.observation_space.shape[0])
possible_actions = 2
agent = Agent(state_size=env.observation_space.shape[0],action_size=possible_actions,seed=0)
print(possible_actions)
action_index_boundary= [2]
for i_episode in range(1, n_episodes+1):
    #print ('EPISODE: {} =======>'.format(i_episode))
    state = env.reset()
    score = 0
    ideal=0
    step_cnt = 0
    for t in range(max_t):
        step_cnt+=1
        action = agent.act(state,eps)
        # map the action into specific dictionary in the environment action
        #print(action)
        action_dict=[0,0]
        prev = 0
        for i,j in enumerate(action_index_boundary):
            if action >= j:
                prev = j
                continue
            else:
                #action_dict[0] = i
                action_dict[1] = action - prev
                break
        #print(action_dict)
        if action_dict[1] == 1:
            ideal +=1
        next_state,reward,done,info = env.step(action_dict, result={})
        #print('State : {0}, Next-State : {1}, Reward : {2}, Done : {3}, Info :{4}'.format(state, next_state, reward, done, info))
        agent.step(state,action,reward,next_state,done)
        ## above step decides whether we will train(learn) the network
        ## actor (local_qnetwork) or we will fill the replay buffer
        ## if len replay buffer is equal to the batch size then we will
        ## train the network or otherwise we will add experience tuple in our 
        ## replay buffer.
        if channelModel:
            utilization_rates = next_state
            G_to_update = env.G
            for i,data in enumerate(G_to_update.edges(data=True)):
                G_to_update[data[0]][data[1]].update({'weight':utilization_rates[i]})
            env.G = G_to_update

        state = next_state
        score += reward
        if done:
            #print ('Episode Length ',str(t),' , score ',str(score))
            #print('ratio of R3 over total',(ideal/t+1))
            break
        scores_window.append(score) ## save the most recent score
        scores.append(score) ## sae the most recent score
        eps = max(eps*eps_decay,eps_end)## decrease the epsilon
        #print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)), end="")
    #print('ratio of R3 over total',(ideal/t+1))
    track_ideal.append(ideal/step_cnt)
    if i_episode %100==0:
        print('\rEpisode {}\tAverage Score {:.2f} \t ideal selection {:.2f}'.format(i_episode,np.mean(scores_window),np.mean(track_ideal)))
        
    if np.mean(scores_window)>=200.0:
        print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100,
                                                                                    np.mean(scores_window)))
        torch.save(agent.qnetwork_local.state_dict(),'checkpoint.pth')
        break
    
