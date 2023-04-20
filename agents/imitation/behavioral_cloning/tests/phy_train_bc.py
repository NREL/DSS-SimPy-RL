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
import opendssdirect as dss
import bc
from reward_nets import BasicShapedRewardNet
from networks import RunningNorm

#from envs.simpy_env.CyberWithChannelEnvSB_123_MidSizeNW_Two import CyberEnv
from envs.openDSSenvSB_DiscreteSpace import openDSSenv
import torch
from generate_expert_demonstrations_feeder import expert_rollouts
import time

def evaluate_policy(env,model):
    """Evaluating the learned policy by running experiments in the environment
    
        :param env: The Open DSS RL environment
        :type env: Gym.Env
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

def train_and_evaluate(env,bc_train_epoch_lens,exp_trajectory_len):
    """For different combination of channel bandwidths, router queue limits and expert demonstrations, train and test the policy and saves the results, reward and policy network.

        :param env: The Open DSS RL environment
        :type env: Gym.Env
        :param bc_train_epoch_lens: List of the number of epochs the behavioral cloning agent need to be trained
        :type bc_train_epoch_lens: list
        :param exp_tajectory_len: The number of expert demonstrations steps considered for AIRL training
        :type exp_tajectory_len: int
        :return: Nothing
        :rtype: None
    """
    for batch in bc_train_epoch_lens:
        
        results={}
        env.contingency = 1 # causes only one line fault
        avg_epi_len_random, succ_avg_epi_len_random, rollouts_random = expert_rollouts(env,episodes=500,_expert = False)
        results['avg_episode_length_random']= avg_epi_len_random
        avg_epi_len_expert, succ_avg_epi_len_expert, rollouts_expert = expert_rollouts(env,episodes=500,_expert = True)
        results['avg_episode_length_expert']= avg_epi_len_expert

        bc_trainer = bc.BC(observation_space=env.observation_space,action_space=env.action_space,demonstrations=rollouts_expert,)

        episode_len_before_training,succ_epi_len_before_training,learner_rewards_before_training = evaluate_policy(env,bc_trainer.policy)

        print(f"Reward before training: {learner_rewards_before_training}, avg : {average(learner_rewards_before_training)}")
        print(f"Episode length before training: {episode_len_before_training}, avg : {average(episode_len_before_training)}")
        results['rw_before_training'] = learner_rewards_before_training
        results['succ_epi_len_before_training'] = succ_epi_len_before_training
        results['episode_length_before_training'] = episode_len_before_training
        bc_trainer.train(n_batches=batch)  # Note: set to 300000 for better results


        episode_len_after_training,succ_epi_len_after_training,learner_rewards_after_training = evaluate_policy(env,bc_trainer.policy)
        print(f"Reward after training: {learner_rewards_after_training}, avg : {average(learner_rewards_after_training)}")
        print(f"Episode length after training: {episode_len_after_training} , avg : {average(episode_len_after_training)}")
        results['rw_after_training'] = learner_rewards_after_training
        results['succ_epi_len_after_training'] = succ_epi_len_after_training
        results['episode_length_after_training'] = episode_len_after_training
        scipy.io.savemat('BC_ieee123_distribution_feeder_min_batch_'+str(batch)+'.mat',results)

if __name__ == "__main__":
    dss_data_dir = desktop_path+'\\ARM_IRL\\cases\\123Bus_SimpleMod\\'
    dss_master_file_dir = 'Redirect ' + dss_data_dir + 'IEEE123Master.dss'

    dss.run_command(dss_master_file_dir)
    circuit = dss.Circuit
    critical_loads_bus = ['58','59','99','100','88','93','94','78','48','50', '111','114', '37','39']
    #critical_loads_bus = ['57','60']
    capacitor_banks =['C83', 'C88a', 'C90b','C92c']
    # switch from and two buses, with the first 6 are normally closed and the last two are normally open
    switches = { 0: ['150r','149'], 1: ['13','152'], 2: ['18','135'], 3: ['60','160'], 4: ['97','197'], 5: ['61','61s'], 6: ['151','300'], 7: ['54','94'] }

    switch_names =[]
    for k,sw in enumerate(switches):
        switch_names.append('Sw'+str(k+1))

    line_faults = ['L55','L68', 'L58', 'L77', 'L45', 'L101', 'L41']
    results={}
    bc_train_epoch_lens =[100,200,300,400,500]
    exp_trajectory_len=2000
    env = openDSSenv(_dss = dss, _critical_loads=critical_loads_bus, _line_faults =line_faults, _switch_names = switch_names, _capacitor_banks = capacitor_banks)
    train_and_evaluate(env, bc_train_epoch_lens,exp_trajectory_len)