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
import opendssdirect as dss
import rollout
from wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from torch import jit,tensor
from airl import AIRL
from reward_nets import BasicShapedRewardNet
from networks import RunningNorm
from generate_expert_demonstrations_feeder import expert_rollouts
from envs.openDSSenvSB_DiscreteSpace import openDSSenv
#from envs.simpy_env.generate_network import create_network2
#from CPEnv_DiscreteDSS_RtrDropRate import CyberPhysicalEnvMT,CyberPhysicalMapping
import torch
import numpy as np
# bigger case
#G = create_network()
#env = CyberEnv(provided_graph=G)
def sample_input_for_reward_network(env):
    """Sample an input to save the reward network 
    
        :param env: The Open DSS RL environment
        :type env: Gym.Env
        :return: returns the state,input,next_state,done concacted to be fed to the reward network
        :rtype: torch
    """
    state = env.reset()
    action = env.action_space.sample()
    print(action)
    obs, reward, done, info = env.step(action,result={})
    #action=[0.0,1.0]
    action_one_hot_encoding = []
    for r in range(8):
        if r == action:
            action_one_hot_encoding.append(1.0)
        else:
            action_one_hot_encoding.append(0.0)

    inputs = []
    
    print(torch.flatten(tensor([state]).float(),1))
    inputs.append(torch.flatten(tensor([state]).float(),1))
    inputs.append(torch.flatten(tensor([[np.asarray(action_one_hot_encoding)]]).float(),1))
    inputs.append(torch.flatten(tensor([obs]).float(),1))
    inputs.append(torch.reshape(tensor([done]).float(), [-1, 1]))
    print(inputs)
    inputs_concat = torch.cat(inputs,dim=1)
    return inputs_concat


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
    test_episode = 500
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
            action = 'Sw'+str(action+1)
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


def train_and_evaluate(env,airl_train_lens,policy_net_train_len,exp_trajectory_len):
    """For different combination of AIRL training length evaluate the learned policy and saves the reward network.
    
        :param env: The Open DSS RL environment
        :type env: Gym.Env
        :param exp_tajectory_len: The number of expert demonstrations steps considered for AIRL training
        :type exp_tajectory_len: int
        :param policy_net_train_len: Samples considered for training the policy network fed in the generator network as initial policy
        :type policy_net_train_len: int
        :param airl_train_lens: List of combination of AIRL training length evaluated
        :type airl_train_lens: list
        :return: Nothing
        :rtype: None
    """
    env.contingency = 1 # causes only one line fault
    avg_epi_len_random, succ_avg_epi_len_random, rollouts_random = expert_rollouts(env,episodes=500,_expert = False)
    results['avg_episode_length_random']= avg_epi_len_random
    results['succ_episode_length_random']= succ_avg_epi_len_random
    avg_epi_len_expert, succ_avg_epi_len_expert, rollouts_expert = expert_rollouts(env,episodes=exp_trajectory_len,_expert = True)
    results['avg_episode_length_expert']= avg_epi_len_expert
    results['succ_episode_length_expert']=succ_avg_epi_len_expert 

    venv = DummyVecEnv([lambda: env])

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

    episode_len_before_training,succ_epi_len_before_training,learner_rewards_before_training = evaluate_policy(env,ppo_model)

    print(f"Reward before training: {learner_rewards_before_training}, avg : {average(learner_rewards_before_training)}")
    print(f"Episode length before training: {episode_len_before_training}, avg : {average(episode_len_before_training)}")
    results['rw_before_training'] = learner_rewards_before_training
    results['succ_epi_len_before_training'] = succ_epi_len_before_training
    results['episode_length_before_training'] = episode_len_before_training

    # this is the discriminator Network
    # normalize_input_layer defines how the input feature of the reward network/discriminator is performed
    reward_net = BasicShapedRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
    )
    print('Defined the reward model')

    for tr_len in airl_train_lens:
        airl_trainer = AIRL(
            demonstrations=rollouts_expert,
            demo_batch_size=512,
            gen_replay_buffer_capacity=1024,
            n_disc_updates_per_round=4,
            venv=venv,
            gen_algo=ppo_model,
            reward_net=reward_net,
        )
        print('Initialized the AIRL agent')


        airl_trainer.train(tr_len)  # Note: set to 300000 for better results

        episode_len_after_training,succ_epi_len_after_training,learner_rewards_after_training = evaluate_policy(env,airl_trainer.gen_algo)
        print(f"Reward after training: {learner_rewards_after_training}, avg : {average(learner_rewards_after_training)}")
        print(f"Episode length after training: {episode_len_after_training} , avg : {average(episode_len_after_training)}")
        results['rw_after_training'] = learner_rewards_after_training
        results['succ_epi_len_after_training'] = succ_epi_len_after_training
        results['episode_length_after_training'] = episode_len_after_training
        #scipy.io.savemat('AIRL_ieee123_feeder_tr_len_'+str(tr_len)+'.mat',results)
        discrim ='case_ieee123_reward_feeder_tr_len_'+str(tr_len)+'.pt'
        generat='case_ieee123_policy_feeder_tr_len_'+str(tr_len)+'.pt'
        net_trace = jit.trace(airl_trainer._reward_net.base.mlp.forward,example_inputs = (sample_input_for_reward_network(env)))
        jit.save(net_trace,desktop_path+'\\Results\\models\\model_airl_distribution_feeder_nw.zip')


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
    env = openDSSenv(_dss = dss, _critical_loads=critical_loads_bus, _line_faults =line_faults, _switch_names = switch_names, _capacitor_banks = capacitor_banks)
    airl_train_lens = [30000,50000,70000]
    exp_trajectory_len=2000
    policy_net_train_len=2000
    train_and_evaluate(env, airl_train_lens,policy_net_train_len,exp_trajectory_len)
