import sys
from numpy import average
import os
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(directory)))))
sys.path.insert(0,desktop_path+'\ARM_IRL')
sys.path.insert(1,os.path.dirname(directory))
sys.path.insert(2,os.path.dirname(directory)+'\experts')
from torch import jit,tensor
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import gym
import torch
import rollout
from wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from generate_expert_demonstrations_simplest_nw import expert_rollouts
from airl import AIRL
from reward_nets import BasicShapedRewardNet
from networks import RunningNorm
import numpy as np
from envs.simpy_env.CyberWithChannelEnvSB import CyberEnv



def evaluate_policy(env,model):
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

# bigger case
#G = create_network()
#env = CyberEnv(provided_graph=G)

# smaller case
env = CyberEnv(channelModel=False,envDebug=False, R2_qlimit=200, ch_bw = 2500)

print(' ppo expert learned')


random_epi_len,_ = expert_rollouts(env,episodes=50,rtrs_comp=['R2'],_expert=False)

expert_epi_len,rollouts = expert_rollouts(env,episodes=100,rtrs_comp=['R2'])
print('executing rollout')

venv = DummyVecEnv([lambda: CyberEnv(channelModel=False,envDebug=False, R2_qlimit=200, ch_bw = 2500)])

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
reward_net = BasicShapedRewardNet(
    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm)
airl_trainer = AIRL(
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

episode_len_before_training,succ_epi_len_before_training,learner_rewards_before_training = evaluate_policy(env,learner)

print(f"Reward before training: {learner_rewards_before_training}, avg : {average(learner_rewards_before_training)}")
print(f"Episode length before training: {episode_len_before_training}, avg : {average(episode_len_before_training)}")
airl_trainer.train(30000)  # Note: set to 300000 for better results

episode_len_after_training,succ_epi_len_after_training,learner_rewards_after_training = evaluate_policy(env,airl_trainer.gen_algo)
print(f"Reward after training: {learner_rewards_after_training}, avg : {average(learner_rewards_after_training)}")
print(f"Episode length after training: {episode_len_after_training} , avg : {average(episode_len_after_training)}")

state = env.reset()
action = env.action_space.sample()
print(action)
next_state, reward, done, _ = env.step(action)
#action=[0.0,1.0]

inputs = []
action_one_hot_encoding = []
for r in range(env.deviceCount):
    if r == action[0]:
        action_one_hot_encoding.append(1.0)
    else:
        action_one_hot_encoding.append(0.0)
for k in range(2):
    if k == action[1]:
        action_one_hot_encoding.append(1.0)
    else:
        action_one_hot_encoding.append(0.0)
print(torch.flatten(tensor([state]).float(),1))
inputs.append(torch.flatten(tensor([state]).float(),1))
inputs.append(torch.flatten(tensor([[np.asarray(action_one_hot_encoding)]]).float(),1))
inputs.append(torch.flatten(tensor([next_state]).float(),1))
inputs.append(torch.reshape(tensor([done]).float(), [-1, 1]))
print(inputs)
inputs_concat = torch.cat(inputs,dim=1)

net_trace = jit.trace(airl_trainer._reward_net.base.mlp.forward,example_inputs = (inputs_concat))
jit.save(net_trace,desktop_path+'\\Results\\models\\model_airl_simple_nw.zip')
