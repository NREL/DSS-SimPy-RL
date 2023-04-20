from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import gym
import sys
import os
directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,os.path.dirname(directory))
sys.path.insert(1,os.path.dirname(directory)+'\experts')
import rollout
from wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from gail import GAIL
from reward_nets import BasicShapedRewardNet,BasicRewardNet
from networks import RunningNorm


env = gym.make("CartPole-v0")
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

venv = DummyVecEnv([lambda: gym.make("CartPole-v0")] * 8)
learner = PPO(
    env=venv,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
)
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

learner_rewards_before_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)

print(f"Reward before training: {learner_rewards_before_training}")

gail_trainer.train(300000)  # Note: set to 300000 for better results
learner_rewards_after_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)

print(f"Reward after training: {learner_rewards_after_training}")