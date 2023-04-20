from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import gym
import sys
import os
directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,os.path.dirname(directory))
sys.path.insert(1,os.path.dirname(directory)+'\experts')
desktop_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(directory)))))
import rollout
from wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from torch import jit, tensor
from airl import AIRL
from reward_nets import BasicShapedRewardNet
from networks import RunningNorm
import torch
import util
env = gym.make("CartPole-v0")


venv = DummyVecEnv([lambda: gym.make("CartPole-v0")] * 8)
print(venv.action_space)
print(venv.observation_space)
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

rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(gym.make("CartPole-v0"))] * 5),
    rollout.make_sample_until(min_timesteps=None, min_episodes=60),
)


learner = PPO(
    env=venv,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
)

reward_net = BasicShapedRewardNet(
    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
)
airl_trainer = AIRL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)

learner_rewards_before_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)

print(f"Reward before training: {learner_rewards_before_training}")

airl_trainer.train(300000)  # Note: set to 300000 for better results
learner_rewards_after_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)

print(f"Reward after training: {learner_rewards_after_training}")
state = env.reset()
action = 1
next_state, reward, done, _ = env.step(action)
action=[0.0,1.0]
inputs = []
print(torch.flatten(tensor([state]).float(),1))
inputs.append(torch.flatten(tensor([state]).float(),1))
inputs.append(torch.flatten(tensor([[action]]).float(),1))
inputs.append(torch.flatten(tensor([next_state]).float(),1))
inputs.append(torch.reshape(tensor([done]).float(), [-1, 1]))
print(inputs)
inputs_concat = torch.cat(inputs,dim=1)

print(inputs_concat)
net_trace = jit.trace(airl_trainer._reward_net.base.mlp.forward,example_inputs = (inputs_concat))
#net_trace = jit.trace(airl_trainer._reward_net.base.mlp.forward,example_inputs = (torch.flatten(tensor([state]),1),torch.flatten(tensor([[action]]),1),torch.flatten(tensor([next_state]),1),torch.reshape(tensor([done]), [-1, 1]),torch.reshape(tensor([done]), [-1, 1])))
#net_trace = jit.trace(airl_trainer._reward_net.forward,example_inputs = (airl_trainer._reward_net,torch.flatten(tensor([state]),1),torch.flatten(tensor([[action]]),1),torch.flatten(tensor([next_state]),1),torch.reshape(tensor([done]), [-1, 1])))
jit.save(net_trace,desktop_path+'\Results\models\model_airl_cartpole_v2.zip')
