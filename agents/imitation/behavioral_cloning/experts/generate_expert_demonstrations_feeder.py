import sys
from numpy import average
import os
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(directory)))))
sys.path.insert(0,desktop_path+'\ARM_IRL')
sys.path.insert(1,os.path.dirname(directory))
from rollout import TrajectoryAccumulator
from wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import random
from envs.openDSSenvSB_DiscreteSpace import openDSSenv

#['L55', 'L58', 'L77','L68','L45', 'L101','L41']
restoration_order ={
    'L41':[5,3,1,4],
    'L45':[5,3,1,4],
    'L55':[4,6],
    'L58':[4,6],
    'L68':[1,2,6],
    'L77':[1,2,6],
    'L101':[2,1,3]
}

restoration_order_with_ss ={
    'L41':[1,3,4,6,2],
    'L45':[0,3,4,6,2],
    'L55':[0,2,6,4],
    'L58':[2,6,4,7],
    'L68':[0,1,2,6,4,7],
    'L77':[0,1,2,6,4,7],
    'L101':[1,2,3,6,7]
}

def expert_rollouts(env,episodes,_expert=True):
    """This function performs roll-outs on the Open-DSS environment to obtain trajectory samples

        :param env: The OpenDSS RL environment
        :type env: gym.Env
        :param episodes: number of episodes the rollout to be performed.
        :type episodes: int
        :param _expert: if set true then generate expert trajectories or random trajectories
        :type _expert: bool
        :return: average episode length and trajectories/rollouts
        :rtype: int, Trajectories
    """
    # Collect rollout tuples.
    trajectories = []
    # accumulator for incomplete trajectories
    trajectories_accum = TrajectoryAccumulator()

    test_episode = episodes
    max_t=100
    acc_len = 0
    acc_reward = 0
    expert = _expert
    
    succ_episode = 0
    succ_len=0
    for i in range(test_episode):
        obs = env.reset()
        #line_faults = env.current_line_faults
        line_faults = env.current_line_faults
        #print('Current Fault : {0}'.format(line_faults))
        done = False
        episode_length = 0
        episodic_reward = 0
        expert_order = []
        action=[]
        trajectories_accum.add_step(dict(obs=obs),0)
        while not done and episode_length < max_t:
            
            if len(env.current_line_faults) > 0 and expert:
                # based on the line fault follow the expert sequence
                expert_order = restoration_order_with_ss[line_faults[0]] # we take the first line as we consider single line outage for now
                #print(expert_order)
                action = 'Sw'+str(expert_order[episode_length]+1)
            else:
                # follow random sequence
                action = random.choice(env.switch_names[0:])
            
            #print(action)
            obs, reward, done, info = env.step(action,result={})
            """ if expert:
                print('Current Fault : {0}'.format(line_faults))
                print(obs) """

            episodic_reward+=reward
            episode_length+=1
            if done:
                succ_episode+=1
                succ_len+=episode_length
            if expert:
                #print('Step {0}'.format(episode_length))
                new_trajs = trajectories_accum.add_steps_and_auto_finish(
                        [int(action[-1])-1],
                        [obs],
                        [reward],
                        [done],
                        [info],
                        True
                    )
            
                trajectories.extend(new_trajs)
            
        #print('Episode Length >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {0}'.format(episode_length))
        acc_len+=episode_length
        acc_reward +=episodic_reward
    print(succ_episode)
    print('Average Episode Length Before Training : {0}, Avg Reward : {1}, Succ Avg Episode {2}'.format(acc_len/test_episode, acc_reward/test_episode, succ_len/succ_episode))
    return acc_len/test_episode, succ_len/succ_episode, trajectories
