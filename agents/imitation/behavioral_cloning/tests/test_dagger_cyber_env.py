import sys
import os
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(directory)))))
sys.path.insert(0,desktop_path+'\ARM_IRL')
sys.path.insert(1,os.path.dirname(directory))
sys.path.insert(2,os.path.dirname(directory)+'\experts')
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from envs.simpy_env.generate_network import create_network,create_network2
import gym
import tempfile

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import bc
from dagger import SimpleDAggerTrainer
from envs.simpy_env.CyberWithChannelEnvSB_123 import CyberEnv

# bigger case
G = create_network2()
env = CyberEnv(provided_graph=G,channelModel=True,envDebug=False, R2_qlimit=100, ch_bw = 2500,with_threat=True)

# smaller case
#env = CyberEnv(channelModel=True,envDebug=False, R2_qlimit=70, ch_bw = 500)

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

venv = DummyVecEnv([lambda: CyberEnv(provided_graph=G,channelModel=True,envDebug=False, R2_qlimit=100, ch_bw = 2500,with_threat=True)])


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