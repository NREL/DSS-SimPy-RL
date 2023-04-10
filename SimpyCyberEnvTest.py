import os
import sys
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(directory))
sys.path.insert(0,desktop_path+'\ARM_IRL')
import gym
import numpy as np
from envs.simpy_env.CyberWithChannelEnvSB_123_Experimentation import CyberEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from envs.simpy_env.generate_network import create_network,draw_cyber_network,create_network2
import random
import networkx as nx

def create_env_run(graph, channelModel, envDebug, queueLimit, channelBW, withThreat):
    """Initialize the Simpy based cyber environment and run with a random agent..
    
        :param graph: Networkx graph model
        :type graph: graph
        :param channelModel: select if there is channel considered within state space
        :type channelModel: bool
        :param envDebug: enable/disable the debugging of the cyber environment
        :type envDebug: bool
        :param queueLimit: defined the queue limit of the routers in the network
        :type queueLimit: int
        :param channelBW: set the channel bandwidth
        :type channelBW: int
        :param withThreat: indicates if threat model is enabled 
        :type ders: bool
        :return: env
        :rtype: gym.Env
    """
    env = CyberEnv(provided_graph=graph, channelModel=channelModel,envDebug=envDebug, R2_qlimit=queueLimit, ch_bw = channelBW,with_threat=withThreat)

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    test_episode = 100
    max_t=100
    acc_len = 0
    acc_reward = 0
    for i in range(test_episode):
        state = obs = env.reset()
        done = False
        episode_length = 0
        episodic_reward = 0
        while not done and episode_length < max_t:
            router_id = random.randint(0,env.deviceCount-1)
                        
            # currently random:  to implement  get the next hop from the shortest path algorithm
            rnd_action_index = random.randint(0, len(env.routers[router_id].out)-1)

            #shortest_path_action_index = nx.single_source_shortest_path(env.G, router_id)['PS'][1]
            path_to_receiver =nx.single_source_shortest_path(env.G, router_id)['PS']

            shortest_path_action_index = path_to_receiver[1]

            #rnd_action = {'device':router_id, 'next_hop':rnd_action_index}
            rnd_action = [router_id,rnd_action_index]

            if shortest_path_action_index != 'PS':
                rtr_id = 'R'+str(shortest_path_action_index)
                rtr_ix = [ix for (ix,item) in enumerate(env.routers[router_id].out) if item.id == rtr_id][0]
                #action = {'device':router_id, 'next_hop':rtr_ix}
                action = [router_id,rtr_ix]
            else:
                action = rnd_action
            #print(action)
            obs, reward, done, info = env.step(action,result={})
            episodic_reward+=reward
            #print('obs {0} reward {1} done {2} '.format(obs,reward,done))
            episode_length+=1
        #print('Episode Length ',episode_length)
        acc_len+=episode_length
        acc_reward +=episodic_reward
    print('Average Episode Length Before Training : {0}, Avg Reward : {1}'.format(acc_len/test_episode, acc_reward/test_episode))
    return env
    #env.render()

def trainAgent(env):
    """Train the model using either A2C or PPO..

        :param env: The openDss RL environment
        :type env: gym.Env
        :return: The trained model
        :rtype: Torch model
    """
    model = PPO('MlpPolicy',
                env=env,
                seed=0,
                batch_size=64,
                ent_coef=0.0,
                learning_rate=0.0003,
                n_epochs=10,
                n_steps=64,)
    model.learn(total_timesteps=1000000)
    return model

def testAgent(env,model):
    """Test the trained model..

        :param env: The openDss RL environment
        :type env: gym.Env
        :param model: The trained policy network
        :type model: gym.Env
        :return: Nothing
        :rtype: None
    """
    acc_len = 0
    test_episode = 100
    max_t=100
    acc_reward = 0
    for i in range(test_episode):
        state = obs = env.reset()
        done = False
        episode_length = 0
        episodic_reward=0
        while not done and episode_length < max_t:
            action, _states = model.predict(obs)
            #print(action)
            obs, reward, done, info = env.step(action,result={})
            #print('obs {0} reward {1} done {2} '.format(obs,reward,done))
            episodic_reward+=reward
            episode_length+=1
        #print('Episode Length ',episode_length)
        #env.render()
        acc_len+=episode_length
        acc_reward+=episodic_reward
    print('Average Episode Length After Training : {0} Avg Reward : {1}'.format(acc_len/test_episode,acc_reward/test_episode))


if __name__ == "__main__":
    G = create_network2()
    env = create_env_run(desktop_path,graph=G,channelModel=False,envDebug=False, queueLimit=130, channelBW= 2500,withThreat=False)
    model = trainAgent(env)
    testAgent(env,model)
