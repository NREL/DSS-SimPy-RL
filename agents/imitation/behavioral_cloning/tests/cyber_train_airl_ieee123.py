import sys
from numpy import average
import os
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(directory)))))
sys.path.insert(0,desktop_path+'\ARM_IRL')
sys.path.insert(1,os.path.dirname(directory))
sys.path.insert(2,os.path.dirname(directory)+'\experts')
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import gym
import scipy.io

import rollout
from wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
#from stable_baselines3.common.evaluation import evaluate_policy

from airl import AIRL
from reward_nets import BasicShapedRewardNet
from networks import RunningNorm

#from envs.simpy_env.CyberWithChannelEnvSB_123_MidSizeNW_Two import CyberEnv
from envs.simpy_env.CyberWithChannelEnvSB_123_Experimentation import CyberEnv
from envs.simpy_env.generate_network import create_network2
import torch
from generate_expert_demonstrations_ieee123 import expert_rollouts
import time

def evaluate_policy(env,model):
    """Evaluating the learned policy by running experiments in the environment
    
        :param cpenv: The cyber RL environment
        :type cpenv: Gym.Env
        :param model: Trained policy network model
        :type model: torch.nn.Module
        :return: average episode length, average reward
        :rtype: float
    """
    acc_len = 0
    succ_len = 0
    test_episode = 100
    succ_episode = 0
    max_t=100
    acc_reward = 0
    for i in range(test_episode):
        state = obs = env.reset()
        #print('Episode ===========================> {0}'.format(i+1))
        done = False
        episode_length = 0
        episodic_reward = 0
        while not done and episode_length < max_t:
            #print(obs)
            action, _states = model.predict(obs)
            #print(action)
            obs, reward, done, info = env.step(action,result={})
            #print('obs {0} reward {1} done {2} '.format(obs,reward,done))
            episodic_reward+=reward
            episode_length+=1
            if done:
                succ_episode+=1
                succ_len+=episode_length
        #print('Episode Length ',episode_length)
        #env.render()
        acc_len+=episode_length
        acc_reward+=episodic_reward
    print('Avg Epi {0} Succ Avg Epi {1} Avg Reward {2}'.format(acc_len/test_episode, succ_len/succ_episode,acc_reward/test_episode))
    return acc_len/test_episode, succ_len/succ_episode,acc_reward/test_episode

def train_and_evaluate(channel_bws, router_qlimits, policy_net_train_len,airl_train_len,exp_trajectory_len):
    """For different combination of channel bandwidths and router queue limits and expert demonstrations, train and test the policy and saves the results, reward and policy network.

        
        :param channel_bws: List of the channel bandwidths value considered in the communication network
        :type channel_bws: list
        :param router_qlimits: List of the router queue upper bound considered in the network
        :type router_qlimits: list
        :param policy_net_train_len: Samples considered for training the policy network fed in the generator network as initial policy
        :type policy_net_train_len: int
        :param airl_train_len: Samples considered for training AIRL network
        :type airl_train_len: int
        :param exp_tajectory_len: Number of expert demonstration samples to be considered
        :type exp_tajectory_len: int
        :return: Nothing
        :rtype: None
    """
    for c in channel_bws:
        print('Channel BW : {0}'.format(c))
        for r in router_qlimits:
            results = {}
            G = create_network2()
            env = CyberEnv(provided_graph=G,channelModel=False,envDebug=False, R2_qlimit=r, ch_bw = c,with_threat=True)
            random_epi_len,_ = expert_rollouts(env,episodes=50,rtrs_comp=['R10','R7','R4'],_expert=False)
            results['avg_episode_length_random']= random_epi_len

            time.sleep(5)

            expert_epi_len,rollouts = expert_rollouts(env,episodes=exp_trajectory_len,rtrs_comp=['R10','R7','R4'])
            results['avg_episode_length_expert']= expert_epi_len

            venv = DummyVecEnv([lambda: CyberEnv(provided_graph=G,channelModel=False,envDebug=False, R2_qlimit=r, ch_bw = c,with_threat=True)])

            # this is the generator or the policy optimizer 
            ppo_model = PPO(
                env=venv,
                policy=MlpPolicy,
                batch_size=64,
                ent_coef=0.0,
                learning_rate=0.0003,
                n_epochs=10,
            )

            ppo_model.learn(policy_net_train_len)

            print('Defined the expert policy')

            # this is the discriminator Network
            # normalize_input_layer defines how the input feature of the reward network/discriminator is performed
            reward_net = BasicShapedRewardNet(
                venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
            )
            print('Defined the reward model')

            airl_trainer = AIRL(
                demonstrations=rollouts,
                #demo_batch_size=1024,
                #gen_replay_buffer_capacity=2048,
                demo_batch_size=512,
                gen_replay_buffer_capacity=1024,
                n_disc_updates_per_round=4,
                venv=venv,
                gen_algo=ppo_model,
                reward_net=reward_net,
            )
            print('Initialized the AIRL agent')

            episode_len_before_training,succ_epi_len_before_training,learner_rewards_before_training = evaluate_policy(env,ppo_model)

            print(f"Reward before training: {learner_rewards_before_training}, avg : {average(learner_rewards_before_training)}")
            print(f"Episode length before training: {episode_len_before_training}, avg : {average(episode_len_before_training)}")
            results['rw_before_training'] = learner_rewards_before_training
            results['succ_epi_len_before_training'] = succ_epi_len_before_training
            results['episode_length_before_training'] = episode_len_before_training
            airl_trainer.train(airl_train_len)  # Note: set to 300000 for better results

            episode_len_after_training,succ_epi_len_after_training,learner_rewards_after_training = evaluate_policy(env,airl_trainer.gen_algo)
            print(f"Reward after training: {learner_rewards_after_training}, avg : {average(learner_rewards_after_training)}")
            print(f"Episode length after training: {episode_len_after_training} , avg : {average(episode_len_after_training)}")
            results['rw_after_training'] = learner_rewards_after_training
            results['succ_epi_len_after_training'] = succ_epi_len_after_training
            results['episode_length_after_training'] = episode_len_after_training
            scipy.io.savemat('AIRL_Actual_Expert_C_ieee123_long_training_qlimit_'+str(r)+'_ch_BW_'+str(c)+'goal_5pkt_each_single_threat.mat',results)
            discrim ='case_C_ieee123_reward_qlimit_long_training_'+str(r)+'_ch_bw_'+str(c)+'after_training_single_threat.pt'
            generat='case_C_ieee123_policy_qlimit_long_training_'+str(r)+'_ch_bw_'+str(c)+'after_training_single_threat.pt'
            torch.save(airl_trainer._reward_net,desktop_path+'\\Results\\models\\'+discrim)
            torch.save(airl_trainer.gen_algo.policy,desktop_path+'\\Results\\models\\'+generat)


if __name__ == "__main__":
    exp_tajectory_len = 2500
    channel_bws = [2500]
    router_qlimits = [120]
    policy_network_train_len = 20000
    airl_train_len = 30000
    train_and_evaluate(channel_bws, router_qlimits,policy_network_train_len,airl_train_len,exp_tajectory_len)