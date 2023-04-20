import sys
import os
directory = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0,directory+'\imitation')
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from airl import AIRL
from reward_nets import BasicShapedRewardNet
from networks import RunningNorm
import rollout
from wrappers import RolloutInfoWrapper
from powergym.env_register import make_env
import random
import itertools

env = make_env('13Bus', worker_idx=None)
env.seed(123456)
venv = DummyVecEnv([lambda: env])

# get obs, act
obs_dim = env.observation_space.shape[0]
train_profiles = list(range(env.num_profiles))
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("a2c_13Bus_airl_v2")

del model # remove to demonstrate saving and loading

model = A2C.load("a2c_13Bus_airl_v2")

rollouts = rollout.rollout(
    model,
    DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=1000),
)

a2c_policy = A2C(
    env=venv,
    policy=MlpPolicy
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
    gen_algo=a2c_policy,
    reward_net=reward_net,
)

airl_trainer.train(30000)

total_numsteps = 0

for i_episode in itertools.count(start=1):

    episode_reward = 0
    episode_steps = 0
    done = False
    load_profile_idx = random.choice(train_profiles)
    obs = env.reset(load_profile_idx=load_profile_idx)

    while not done:
        action, _states = airl_trainer.gen_algo.predict(obs)
        next_obs, reward, done, info = env.step(action)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        mask = 1 if episode_steps == env.horizon else float(not done)
        obs = next_obs

    print("episode: {}, profile: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode,
                                                                                               load_profile_idx,
                                                                                               total_numsteps,
                                                                                               episode_steps,
                                                                                               round(episode_reward,
                                                                                                     2)))

    total_numsteps += 24

    if total_numsteps >= 1000: break

