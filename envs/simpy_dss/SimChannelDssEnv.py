# -*- coding: utf-8 -*-
"""
Created on Fri July 1 09:17:05 2022

@author: abhijeetsahu

This environment would merge both Simpy and OpenDSS environment.. Current implementation is a dummy merge..Still need to update
"""
import sys
import os
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(os.path.dirname(directory)))
sys.path.insert(0,desktop_path+'\ARM_IRL')
import collections
from envs.openDSSenv import openDSSenv
from envs.simpy_env.CyberWithChannelEnv import CyberEnv
from envs.simpy_env.generate_network import create_network,draw_cyber_network
import opendssdirect as dss
import random
import networkx as nx
import concurrent.futures
from functools import partial
import threading
import pandas as pd
from queue import Queue

# TO IMPLEMENT
# This class will be used to map the cyber components into the physical space.
# All the Data Concentrator would be connected to certain buses, lines, capacitor banks in the system 
# Depending on the line fault, the cyber simulator would be initiating an event packet...
# And later when the switches are opened or closed, every zone would acknowledge the operation of the switch.

# How to incorporate the inter-env interaction. The best method is to use an interconnecting Queues... One for phy communicating cyb another vice-versa
# What are the information we want to pass across these simulator?
# Depending on the Line fault and switch loc, the cyber emulator will pick the Data concentrator to first forward the line info, followed by ACK for switching action (PHY=>CYB)
# Based on the succesful packet transmission, the phy side action will be executed. (CYB===>PHY)
def CyberPhysicalMapping():
    # read the bus file
    fp = desktop_path+'\\ARM_IRL\\cases\\123Bus_Simple\\Buses_Pyomo.csv'
    bi = pd.read_csv(fp)
    bus_info = bi.set_index('Buses')['Zone'].to_dict()

    fp2 = desktop_path+'\\ARM_IRL\\cases\\123Bus_Simple\\Lines_data_Pyomo.csv'
    li = pd.read_csv(fp2)
    line_info = li.set_index('Lines').T.to_dict('list')


    return bus_info, line_info


#### Dummy Variant Serial Execution###################

class CyberPhysicalEnvDummy:
    def __init__(self, cenv, penv, compzones):
        self.envs = []
        self.envs.append(cenv)
        self.envs.append(penv)
        self.compzones = comp_zones

    def reset(self):
        for env in self.envs:
            env.reset()

    def step(self, actions):
        obs = []
        rewards = []
        dones = []
        infos = []

        for env, ac in zip(self.envs, actions):
            ob, rew, done, info = env.step(ac)
            obs.append(ob)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)

            if done:
                env.reset()

        return obs, rewards, all(dones), infos

# use multi threading parallel execution#####################
class CyberPhysicalEnvMT:
    def __init__(self, cenv, penv, comp_zones):
        self.envs = []
        cenv.comp_zones = comp_zones
        self.envs.append(cenv)
        self.envs.append(penv)
        self.comp_zones = comp_zones
        self.pc_queue = Queue() # pass data from phy env to cyb env
        self.cp_queue = Queue() # pass data from cyb env to phy env

    def reset(self):
        for env in self.envs:
            env.reset()

    def step(self, actions):
        obs = []
        rewards = []
        dones = []
        infos = []

        result={}
        phy_thread =threading.Thread(target=penv.step, args= (actions[1],result, self.pc_queue, self.cp_queue))
        phy_thread.start()

        cyb_thread = threading.Thread(target=cenv.step, args= (actions[0],result, self.pc_queue, self.cp_queue))
        cyb_thread.start()

        phy_thread.join()
        cyb_thread.join()

        res= []
        od_res = collections.OrderedDict(sorted(result.items()))
        for k, v in od_res.items():
            res.append(v)

        for ob, rew, done, info in res:
            obs.append(ob)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)

        return obs, rewards, all(dones), infos

if __name__ == '__main__':

    bi, li = CyberPhysicalMapping()
    comp_zones = bi
    for i,(k,v) in enumerate(li.items()):
        comp_zones[k] = bi[v[0]]

    comp_zones['C83'] = bi['83']
    comp_zones['C88a'] = bi['88']
    comp_zones['C90b'] = bi['90']
    comp_zones['C92c'] = bi['92']

    # Create the Cyber Network
    G = create_network()
    cenv = CyberEnv(provided_graph=G)
    
    # Create the Physical Network
    dss_data_dir = desktop_path+'\\ARM_IRL\\cases\\123Bus_Simple\\'
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

    line_faults = ['L55','L68', 'L58', 'L77', 'L45', 'L101', 'L41']

    penv = openDSSenv(_dss = dss, _critical_loads=critical_loads_bus, _line_faults =line_faults, _switch_names = switch_names, _capacitor_banks = capacitor_banks)

    # This is the creation of mixed environment
    #cyber_phy_env = CyberPhysicalEnvDummy(cenv, penv, comp_zones)

    cyber_phy_env= CyberPhysicalEnvMT(cenv, penv,comp_zones)

    episodes = 100
    max_episode_len = 50

    for i in range(episodes):
        print('Episode {}'.format(i+1))
        print('****************')
        cyber_phy_env.reset()
        #print('observation : {}'.format(state))
        action_index = random.randint(1,2)
        done = False
        ctr = 0
        episodic_reward = []
        
        while not done and  ctr < max_episode_len:
            actions = []
            ctr+=1
            # phy
            # randomly select an action for time-being until we train an agent
            phy_action = random.choice(switch_names[1:])
            
            # randomly pick a router to modify the next_hop
            router_id = random.randint(0,cenv.deviceCount-1)
            
            # currently random:  to implement  get the next hop from the shortest path algorithm
            rnd_action_index = random.randint(0, len(cenv.routers[router_id].out)-1)

            shortest_path_action_index = nx.single_source_shortest_path(cenv.G, router_id)['PS'][1]

            rnd_action = {'device':router_id, 'next_hop':rnd_action_index}

            if shortest_path_action_index != 'PS':
                cyb_action = {'device':router_id, 'next_hop':shortest_path_action_index}
            else:
                cyb_action = rnd_action

            # mixed environment step
            actions.append(cyb_action)
            actions.append(phy_action)

            next_state, reward, done, info = cyber_phy_env.step(actions)
            print('Fusion State {0} Episode Termination State : {1}'.format(next_state, done))

            episodic_reward.append(reward)
        