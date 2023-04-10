import os
import sys
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(directory))
sys.path.insert(0,desktop_path+'\ARM_IRL')
from agents.per_dqn import *
from envs.simpy_env.CyberWithChannelEnv import CyberEnv
import random
import networkx as nx
import numpy as np
from envs.simpy_env.generate_network import create_network,draw_cyber_network
from collections import namedtuple, deque 
import torch

G = create_network()
env = CyberEnv(provided_graph=G)

n_episodes = 200
max_t = 30

scores = [] # list containing score from each episode
scores_window = deque(maxlen=100) # last 100 scores


# currently lets initialize the action_size to be limited based on the connectivity among the routers
possible_actions = 0
action_index_boundary = []
for routers in env.routers:
    possible_actions += len(routers.out)
    action_index_boundary.append(possible_actions)
batch_size = 32
agent = Agent(state_size=env.observation_space.shape[0],action_size=possible_actions,per=True)

replay_index = 0
c = 1
for i_episode in range(1, n_episodes+1):
    print ('EPISODE: {} =======>'.format(i_episode))
    state = env.reset()
    score = 0
    for t in range(max_t):
        action = agent.act(state)
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

        next_state,reward,done,info = env.step(action_dict, result={})[0]
        agent.remember(state, action, reward, next_state, done)
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            replay_index += 1
            if replay_index % c == 0:
                agent.target_train()

        utilization_rates = next_state[env.deviceCount:]
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
        #print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)), end="")
    if i_episode %20==0:
        print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)))
        
    if np.mean(scores_window)>=2.0:
        print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100,
                                                                                    np.mean(scores_window)))
        torch.save(agent.qnetwork_local.state_dict(),'checkpoint.pth')
        break
    
