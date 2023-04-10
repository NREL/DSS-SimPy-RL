import gym
import collections
from powergym.env_register import make_env,remove_parallel_dss
from simpy_env.CyberVoltVar_IEEE123 import CyberEnv
from simpy_env.generate_network import create_network,create_network2
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from queue import Queue
import threading
import random
import statistics
import networkx as nx
import pandas as pd


def CyberPhysicalMapping():
    # read the bus file
    fp = r'D:\GithubRepos\SimPyDSS\systems\123Bus\Buses_Pyomo.csv'
    bi = pd.read_csv(fp)
    bus_info = bi.set_index('Buses')['Zone'].to_dict()

    fp2 = r'D:\GithubRepos\SimPyDSS\systems\123Bus\Lines_data_Pyomo.csv'
    li = pd.read_csv(fp2)
    line_info = li.set_index('Lines').T.to_dict('list')
    return bus_info, line_info

class CyberPhysicalEnv(gym.Env):

    def __init__(self, cenv, penv, comp_zones, train_profiles):
        self.envs = []
        cenv.comp_zones = comp_zones
        self.envs.append(cenv)
        self.envs.append(penv)
        self.comp_zones = comp_zones
        self.pc_queue = Queue()  # pass data from phy env to cyb env
        self.cp_queue = Queue()  # pass data from cyb env to phy env
        self.load_profile_idx = train_profiles
        action_list = [cenv.deviceCount, 5]
        action_list.extend([2] * penv.cap_num)
        action_list.extend([penv.reg_act_num] * penv.reg_num)
        action_list.extend([penv.bat_act_num] * penv.bat_num)
        if cenv.channelModel == False:
            self.observation_space = spaces.Box(low=0, high=1000000.0, shape=(
            cenv.deviceCount + penv.observation_space.shape[0],), dtype=np.float32)
            self.action_space = spaces.MultiDiscrete(action_list)
        else:
            self.observation_space = spaces.Box(low=0, high=1000000.0, shape=(
            len(cenv.channels) + penv.observation_space.shape[0],), dtype=np.float32)
            self.action_space = spaces.MultiDiscrete(action_list)

    def reset(self):
        obs = []
        counter = 0
        for env in self.envs:
            if counter == 0:
                # this is for the reset of the cyber environment
                obs = list(env.reset())
            else:
                # resets the physical environment with a random load profile
                obs.extend(env.reset(load_profile_idx = random.choice(self.load_profile_idx)))
            counter += 1
        return np.array(obs)

    def step(self, actions):
        #print(actions)
        obs = []
        rewards = []
        dones = []
        infos = ''

        result={}
        #print(self.map_sw[actions[2]])
        #phy_thread =threading.Thread(target=self.envs[1].step, args= (actions[2:][0],result, self.pc_queue, self.cp_queue))
        phy_thread = threading.Thread(target=self.envs[1].step,
                                      args=(actions[2:], result, self.pc_queue, self.cp_queue))
        phy_thread.start()

        cyb_thread = threading.Thread(target=self.envs[0].step, args= (actions[0:2],result, self.pc_queue, self.cp_queue))
        cyb_thread.start()

        phy_thread.join()
        cyb_thread.join()

        res= []
        od_res = collections.OrderedDict(sorted(result.items()))
        for k, v in od_res.items():
            res.append(v)
        counter = 0
        for ob, rew, done, info in res:
            if counter == 0:
                obs= list(ob)
            else:
                obs.extend(ob)
            rewards.append(rew)
            dones.append(done)
            if counter == 0:
                infos=str(info)
            else:
                infos+=str(info)
            counter+=1
        #print(dones)
        #print('{0} {1} {2} {3}'.format(obs, rewards[0]+rewards[1], all(dones), infos))
        information = {'terminal_observation':obs}
        #information["episode"] = infos
        return np.array(obs), rewards[0]+rewards[1], all(dones), information


if __name__ == '__main__':

    bi, li = CyberPhysicalMapping()
    comp_zones = bi
    for i, (k, v) in enumerate(li.items()):
        comp_zones[k] = bi[v[0]]

    comp_zones['c83'] = bi['83']
    comp_zones['c88a'] = bi['88']
    comp_zones['c90b'] = bi['90']
    comp_zones['c92c'] = bi['92']
    comp_zones['batt1'] = bi['33']
    comp_zones['batt2'] = bi['114']
    comp_zones['batt3'] = bi['67']
    comp_zones['batt4'] = bi['300']
    comp_zones['reg1a'] = bi['150']
    comp_zones['reg2a'] = bi['9']
    comp_zones['reg3a'] = bi['25']
    comp_zones['reg4a'] = bi['160']
    comp_zones['reg3c'] = bi['25']
    comp_zones['reg4b'] = bi['160']
    comp_zones['reg4c'] = bi['160']

    penv = make_env('123Bus', worker_idx=None)
    train_profiles = list(range(penv.num_profiles))

    G = create_network2()
    cenv = CyberEnv(provided_graph=G, channelModel=False,envDebug=False, R2_qlimit=200, ch_bw = 2500,with_threat=False)

    cyber_phy_env = CyberPhysicalEnv(cenv,penv,comp_zones,train_profiles)

    episodes = 100
    max_episode_len = 100

    for i in range(episodes):
        print('Episode {}'.format(i + 1))
        # print('****************')
        cyber_phy_env.reset()
        # print('observation : {}'.format(state))
        action_index = random.randint(1, 2)
        done = False
        ctr = 0
        episodic_reward = []
        episode_length = 0

        while not done and ctr < max_episode_len:
            actions = []
            ctr += 1
            # phy
            # randomly select an action for time-being until we train an agent
            phy_action = penv.random_action()

            # randomly pick a router to modify the next_hop
            router_id = random.randint(0, cenv.deviceCount - 1)

            # currently random:  to implement  get the next hop from the shortest path algorithm
            rnd_action_index = random.randint(0, len(cenv.routers[router_id].out) - 1)

            shortest_path_action_index = nx.single_source_shortest_path(cenv.G, router_id)['PS'][1]

            rnd_action = [router_id, rnd_action_index]

            if shortest_path_action_index != 'PS' and False:
                rtr_id = 'R' + str(shortest_path_action_index)
                rtr_ix = [ix for (ix, item) in enumerate(cenv.routers[router_id].out) if item.id == rtr_id][0]
                # action = {'device':router_id, 'next_hop':rtr_ix}
                cyb_action = [router_id, rtr_ix]
            else:
                cyb_action = rnd_action

            # mixed environment step
            actions.append(cyb_action)
            cyb_action.append(phy_action)
            actions = cyb_action

            next_state, reward, done, info = cyber_phy_env.step(actions)
            # print('Fusion State {0} Episode Termination State : {1}'.format(next_state, done))
            # print(reward[1])
            episodic_reward.append(reward)
            episode_length += 1
        print('Average Episode Reward {0} and Episode Length {1}'.format(statistics.mean(episodic_reward),
                                                                         episode_length))
