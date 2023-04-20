"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import sys
import os
directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,os.path.dirname(directory))
sys.path.insert(1,os.path.dirname(directory)+'\experts')
import gym
import tempfile

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import bc
from dagger import SimpleDAggerTrainer

env = gym.make("CartPole-v1")
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
expert.learn(1000)  # Note: set to 100000 to train a proficient expert

venv = DummyVecEnv([lambda: gym.make("CartPole-v1")])


bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
)

with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
    print(tmpdir)
    dagger_trainer = SimpleDAggerTrainer(
        venv=venv, scratch_dir=tmpdir, expert_policy=expert, bc_trainer=bc_trainer
    )

    dagger_trainer.train(2000)

reward, _ = evaluate_policy(dagger_trainer.policy, env, 10)
print(reward)