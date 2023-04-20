from distutils.log import debug
import sys
import os
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(directory)))))
sys.path.insert(0,desktop_path+'\ARM_IRL')
sys.path.insert(1,os.path.dirname(directory))
sys.path.insert(2,os.path.dirname(directory)+'\experts')
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from envs.simpy_env.generate_network import create_network,create_network2

import bc
import rollout
from wrappers import RolloutInfoWrapper
from envs.simpy_env.CyberWithChannelEnvSB_123 import CyberEnv

# bigger case
G = create_network2()
env = CyberEnv(provided_graph=G,channelModel=True,envDebug=False, R2_qlimit=100, ch_bw = 2500,with_threat=True)

# smaller case
#env = CyberEnv(channelModel=True,envDebug=False)

def train_expert():
    print("Training a expert.")
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
    expert.learn(1000000)  # Note: change this to 100000 to trian a decent expert.
    return expert


def sample_expert_transitions():
    expert = train_expert()

    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        #rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rollout.make_sample_until(min_timesteps=None, min_episodes=10000),
    )
    return rollout.flatten_trajectories(rollouts)

# collect samples of transition from expert
transitions = sample_expert_transitions()

print('No of transitions'+ str(len(transitions)))
""" for ts in transitions:
    print(ts )"""

# initialize the behavior cloning trainer agent
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
)

# evaluate the policy first without the training using behavioral cloning
reward, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=50, render=False)
print(f"Reward before training: {reward}")

print("Training a policy using Behavior Cloning")
bc_trainer.train(n_epochs=20)

# then  again re-evaluate the policy after training the behavioral cloning
reward, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=50, render=False)
print(f"Reward after training: {reward}")
