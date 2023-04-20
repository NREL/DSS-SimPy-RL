from powergym.env_register import make_env
import random
from stable_baselines3 import A2C

import itertools

env = make_env('8500Node', worker_idx=None)
env.seed(123456)

# get obs, act
obs_dim = env.observation_space.shape[0]
train_profiles = list(range(env.num_profiles))
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("a2c_8500Node_v2")

del model # remove to demonstrate saving and loading

model = A2C.load("a2c_8500Node_v2")

total_numsteps = 0

for i_episode in itertools.count(start=1):

    episode_reward = 0
    episode_steps = 0
    done = False
    load_profile_idx = random.choice(train_profiles)
    obs = env.reset(load_profile_idx=load_profile_idx)

    while not done:
        action, _states = model.predict(obs)
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

