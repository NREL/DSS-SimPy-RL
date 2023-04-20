# This will generate the expert demonstrations sequences till an episode is terminated

import sys
from numpy import average
import os
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(directory)))))
sys.path.insert(0,desktop_path+'\ARM_IRL')
sys.path.insert(1,os.path.dirname(directory))
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import gym

import rollout
from rollout import TrajectoryAccumulator
from wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from airl import AIRL
from reward_nets import BasicShapedRewardNet
from networks import RunningNorm
import random
import networkx as nx
from envs.simpy_env.CyberWithChannelEnvSB_123_MidSizeNW_Two import CyberEnv
#from envs.simpy_env.CyberWithChannelEnvSB_123_Experimentation import CyberEnv

def get_parent_routers(env, comp_rtr_obj):
    parent_routers = []
    for router in env.routers:
        for ix,item in enumerate(router.out):
            if item.id == comp_rtr_obj.id:
                parent_routers.append(router)
    return parent_routers

def get_next_child_router(env, packet_drop_rates, pdr_child_nodes_ix):
    print(packet_drop_rates)
    ix_rtr_to_redirect = pdr_child_nodes_ix[packet_drop_rates.index(min(packet_drop_rates))]
    for rtr in env.routers:
        if rtr.id == 'R'+str(ix_rtr_to_redirect+1):
            return rtr

#env = CyberEnv(channelModel=False,envDebug=False, R2_qlimit=100, ch_bw = 2000, with_threat=True)

def expert_rollouts(env,episodes,rtrs_comp,_expert=True):
    # Collect rollout tuples.
    trajectories = []
    # accumulator for incomplete trajectories
    trajectories_accum = TrajectoryAccumulator()

    test_episode = episodes
    max_t=100
    acc_len = 0
    acc_reward = 0
    expert = _expert
    # for the first midsize network 2
    #router_ids_to_target = ['R3','R4','R5']
    # for the first midsize network 2
    router_ids_to_target = rtrs_comp
    #router_ids_to_target = ['R4','R5']
    for i in range(test_episode):
        obs = env.reset()
        rtr_comp = env.rtr_compromised
        done = False
        episode_length = 0
        episodic_reward = 0
        action=[]
        trajectories_accum.add_step(dict(obs=obs),0)
        while not done and episode_length < max_t:
            if len(rtr_comp) > 0 and expert:
                comp_rtr_obj = [x for x in env.routers if x.id == router_ids_to_target[rtr_comp[0]]][0]

                print('Compromised Router {0}'.format(comp_rtr_obj.id))
                
                # get the precursor node of this router
                parent_routers = get_parent_routers(env,comp_rtr_obj)

                action_possible = []

                for parent_router in parent_routers:
                    print('Parent Router {0}'.format(parent_router.id))

                    router_id = int(parent_router.id[1:])-1

                    # set that as the router to modify if the drop rate of the compromised router is more than a limit
                    pdr_child_nodes_ix = []
                    packet_drop_rates = []
                    
                    for ix,item in enumerate(parent_router.out):
                        name = item.id
                        pdr_child_nodes_ix.append(int(name[1:])-1)

                    for ix,ob in enumerate(obs):
                        if ix in pdr_child_nodes_ix:
                            packet_drop_rates.append(ob)

                    # select the child router index with the least drop rate
                    # threshold = 10 for cyber n/w 2
                    print('packet drop comp router '+str(comp_rtr_obj.packets_drop))
                    if comp_rtr_obj.packets_drop > 10:
                        child_rtr = get_next_child_router(env,packet_drop_rates,pdr_child_nodes_ix)
                        print('Next Selected Router {0}'.format(child_rtr.id))
                        rtr_ix = [ix for (ix,item) in enumerate(parent_router.out) if item.id == child_rtr.id][0]
                        action = [router_id,rtr_ix]
                    else:
                        rtr_ix = [ix for (ix,item) in enumerate(parent_router.out) if item.id == comp_rtr_obj.id][0]
                        action = [router_id, rtr_ix]
                    action_possible.append(action)

                action = random.choice(action_possible)

            else:
                router_id = random.randint(0,env.deviceCount-1)
                # currently random:  to implement  get the next hop from the shortest path algorithm
                rnd_action_index = random.randint(0, len(env.routers[router_id].out)-1)
                action = [router_id,rnd_action_index]
            
            
            #print(action)
            obs, reward, done, info = env.step(action,result={})
            if expert:
                #print('Step {0}'.format(episode_length))
                new_trajs = trajectories_accum.add_steps_and_auto_finish(
                        [action],
                        [obs],
                        [reward],
                        [done],
                        [info],
                        True
                    )
            
                trajectories.extend(new_trajs)
            episodic_reward+=reward
            #print('obs {0} reward {1} done {2} '.format(obs,reward,done))
            #print('Trajectory {0}'.format(trajectories))
            episode_length+=1
        #print('Episode Length >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {0}'.format(episode_length))
        acc_len+=episode_length
        acc_reward +=episodic_reward
    print('Average Episode Length Before Training : {0}, Avg Reward : {1}'.format(acc_len/test_episode, acc_reward/test_episode))
    return acc_len/test_episode,trajectories
