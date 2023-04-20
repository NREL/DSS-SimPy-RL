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
from stable_baselines3.common.evaluation import evaluate_policy

from airl import AIRL
from gail import GAIL
from reward_nets import BasicShapedRewardNet,BasicRewardNet
from networks import RunningNorm

from envs.simpy_env.CyberWithChannelEnvSB_123_Experimentation import CyberEnv
from envs.simpy_env.generate_network import create_network2
import torch
# bigger case
#G = create_network()
#env = CyberEnv(provided_graph=G)
for c in range(2500,2600,500):
    print('Channel BW : {0}'.format(c))
    for r in range(120,140,20):
        results = {}
        G = create_network2()
        env = CyberEnv(provided_graph=G, channelModel=False,envDebug=False, R2_qlimit=r, ch_bw = c,with_threat=False)

        expert = PPO(
            policy=MlpPolicy,
            env=env,
            seed=0,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.0003,
            n_epochs=10,
            n_steps=64,
        )
        expert.learn(100000)  # Note: set to 100000 to train a proficient expert

        print(' ppo expert learned')
        rollouts = rollout.rollout(
            expert,
            DummyVecEnv([lambda: RolloutInfoWrapper(CyberEnv(provided_graph=G, channelModel=False,envDebug=False, R2_qlimit=r, ch_bw = c,with_threat=False))]),
            rollout.make_sample_until(min_timesteps=None, min_episodes=500),
        )
        print('executing rollout')

        venv = DummyVecEnv([lambda: CyberEnv(provided_graph=G,channelModel=False,envDebug=False, R2_qlimit=r, ch_bw = c,with_threat=False)])

        # this is the generator or the policy optimizer 
        learner = PPO(
            env=venv,
            policy=MlpPolicy,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.0003,
            n_epochs=10,
        )

        print('Defined the expert policy')

        # this is the discriminator Network
        # normalize_input_layer defines how the input feature of the reward network/discriminator is performed
        reward_net = BasicRewardNet(
            venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
        )
        gail_trainer = GAIL(
            demonstrations=rollouts,
            demo_batch_size=1024,
            gen_replay_buffer_capacity=2048,
            n_disc_updates_per_round=4,
            venv=venv,
            gen_algo=learner,
            reward_net=reward_net,
        )

        #print("Reward Network/Discriminator Parameter before training")
        #print(list(reward_net.parameters()))

        #print("Policy Network/Generator Parameter before training")
        #print(learner.get_parameters())

        learner_rewards_before_training, episode_len_before_training = evaluate_policy(
            learner, venv, 100, return_episode_rewards=True
        )

        print(f"Reward before training: {learner_rewards_before_training}, avg : {average(learner_rewards_before_training)}")
        print(f"Episode length before training: {episode_len_before_training}, avg : {average(episode_len_before_training)}")
        results['rw_before_training'] = learner_rewards_before_training
        results['episode_length_before_training'] = episode_len_before_training
        gail_trainer.train(300000)  # Note: set to 300000 for better results

        #print("Reward Network/Discriminator Parameter after training")
        #print(list(airl_trainer._reward_net.parameters()))

        #print("Policy Network/Generator Parameter before training")
        #print(airl_trainer.gen_algo.get_parameters())

        """ learner_rewards_after_training, _ = evaluate_policy(
            learner, venv, 100, return_episode_rewards=True
        ) """

        learner_rewards_after_training, episode_len_after_training = evaluate_policy(
            gail_trainer.gen_algo, venv, 100, return_episode_rewards=True
        )

        print(f"Reward after training: {learner_rewards_after_training}, avg : {average(learner_rewards_after_training)}")
        print(f"Episode length after training: {episode_len_after_training} , avg : {average(episode_len_after_training)}")
        results['rw_after_training'] = learner_rewards_after_training
        results['episode_length_after_training'] = episode_len_after_training
        scipy.io.savemat('GAIL_qlimit_'+str(r)+'_ch_BW_'+str(c)+'goal_2_each_DC_pkt_exp_size_50_random_rtr_normal_threat.mat',results)
        discrim ='reward_qlimit_'+str(r)+'_ch_bw_'+str(c)+'after_training_normal_threat.pt'
        generat='policy_qlimit_'+str(r)+'_ch_bw_'+str(c)+'after_training_normal_threat.pt'
        torch.save(gail_trainer._reward_net,desktop_path+'\\Results\\models\\'+discrim)
        torch.save(gail_trainer.gen_algo.policy,desktop_path+'\\Results\\models\\'+generat)