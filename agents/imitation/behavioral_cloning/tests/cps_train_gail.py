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

from gail import GAIL
from reward_nets import BasicShapedRewardNet
from networks import RunningNorm
from generate_expert_demonstrations_cps import expert_rollouts
from envs.openDSSenvSB_DiscreteSpace import openDSSenv
from envs.simpy_env.CyberWithChannelEnvSB_123_Experimentation import CyberEnv
from envs.simpy_env.generate_network import create_network2
from envs.simpy_dss.CPEnv_DiscreteDSS_RtrDropRate import CyberPhysicalEnvMT,CyberPhysicalMapping
import torch
# bigger case
#G = create_network()
#env = CyberEnv(provided_graph=G)


def evaluate_policy(cpenv,model):
    """Evaluating the learned policy by running experiments in the environment
    
        :param cpenv: The cyber-physical RL environment
        :type cpenv: Gym.Env
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
        state = obs = cpenv.reset()
        #print('Episode ===========================> {0}'.format(i+1))
        done = False
        episode_length = 0
        episodic_reward = 0
        while not done and episode_length < max_t:
            #print(obs)
            action, _states = model.predict(obs)
            #action = 'Sw'+str(action+1)
            obs, reward, done, info = cpenv.step(action)
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

bi, li = CyberPhysicalMapping()
comp_zones = bi
for i,(k,v) in enumerate(li.items()):
    comp_zones[k] = bi[v[0]]

comp_zones['C83'] = bi['83']
comp_zones['C88a'] = bi['88']
comp_zones['C90b'] = bi['90']
comp_zones['C92c'] = bi['92']

def train_and_evaluate(comp_zones, exp_tajectories, channel_bws, router_qlimits, policy_net_train_len,gail_train_len):
    """For different combination of channel bandwidths, router queue limits and expert demonstrations, train and test the policy and saves the results, reward and policy network.
    
        :param comp_zones: Cyber Physical mapping information.
        :type comp_zones: dict
        :param exp_tajectories: List of the number of expert demonstrations steps considered for GAIL training
        :type exp_tajectories: list
        :param channel_bws: List of the channel bandwidths value considered in the communication network
        :type channel_bws: list
        :param router_qlimits: List of the router queue upper bound considered in the network
        :type router_qlimits: list
        :param policy_net_train_len: Samples considered for training the policy network fed in the generator network as initial policy
        :type policy_net_train_len: int
        :param gail_train_len: Samples considered for training GAIL network
        :type gail_train_len: int
        :return: Nothing
        :rtype: None
    """
    for exp_traj in exp_tajectories:
        for c in channel_bws:
            print('Channel BW : {0}'.format(c))
            for r in router_qlimits:
                results = {}
                G = create_network2()
                cenv = CyberEnv(provided_graph=G, channelModel=False,envDebug=False, R2_qlimit=r, ch_bw = c,with_threat=True,comp_zones=comp_zones)

                # Create the Physical Network
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
                #line_faults = ['L55']

                penv = openDSSenv(_dss = dss, _critical_loads=critical_loads_bus, _line_faults =line_faults, _switch_names = switch_names, _capacitor_banks = capacitor_banks)

                # This is the creation of mixed environment
                #cyber_phy_env = CyberPhysicalEnvDummy(cenv, penv, comp_zones)

                cyber_phy_env= CyberPhysicalEnvMT(cenv, penv,comp_zones)

                # generate expert roll-outs
                avg_epi_len_random, succ_avg_epi_len_random, rollouts_random = expert_rollouts(cyber_phy_env.envs[0],cyber_phy_env.envs[1],episodes=100,_expert = False)
                results['avg_episode_length_random']= avg_epi_len_random
                results['succ_episode_length_random']= succ_avg_epi_len_random
                avg_epi_len_expert, succ_avg_epi_len_expert, rollouts_expert = expert_rollouts(cyber_phy_env.envs[0],cyber_phy_env.envs[1],episodes=exp_traj,_expert = True)
                results['avg_episode_length_expert']= avg_epi_len_expert 
                results['succ_episode_length_expert']=succ_avg_epi_len_expert 

                venv = DummyVecEnv([lambda: CyberPhysicalEnvMT(cenv, penv,comp_zones)])

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

                learner.learn(policy_net_train_len)

                # this is the discriminator Network
                # normalize_input_layer defines how the input feature of the reward network/discriminator is performed
                reward_net = BasicShapedRewardNet(
                    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
                )
                gail_trainer = GAIL(
                    demonstrations=rollouts_expert,
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

                episode_len_before_training,succ_epi_len_before_training,learner_rewards_before_training =  evaluate_policy(cyber_phy_env,learner)

                print(f"Reward before training: {learner_rewards_before_training}, avg : {average(learner_rewards_before_training)}")
                print(f"Episode length before training: {episode_len_before_training}, avg : {average(episode_len_before_training)}")
                results['rw_before_training'] = learner_rewards_before_training
                results['succ_epi_len_before_training'] = succ_epi_len_before_training
                results['episode_length_before_training'] = episode_len_before_training
                gail_trainer.train(gail_train_len)  # Note: set to 300000 for better results

                episode_len_after_training,succ_epi_len_after_training,learner_rewards_after_training = evaluate_policy(cyber_phy_env,gail_trainer.gen_algo)

                print(f"Reward after training: {learner_rewards_after_training}, avg : {average(learner_rewards_after_training)}")
                print(f"Episode length after training: {episode_len_after_training} , avg : {average(episode_len_after_training)}")
                results['rw_after_training'] = learner_rewards_after_training
                results['succ_epi_len_after_training']=succ_epi_len_after_training
                results['episode_length_after_training'] = episode_len_after_training
                scipy.io.savemat('CPS_GAIL_trlen_'+str(gail_train_len)+'_exp_traj_'+str(exp_traj)+'_qlimit_'+str(r)+'_ch_BW_'+str(c)+'goal__random_rtr_one_threat.mat',results)
                discrim ='gail_trlen_'+str(gail_train_len)+'_exp_traj_'+str(exp_traj)+'_reward_qlimit_'+str(r)+'_ch_bw_'+str(c)+'after_training_one_threat.pt'
                generat='gail_trlen_'+str(gail_train_len)+'_exp_traj_'+str(exp_traj)+'_policy_qlimit_'+str(r)+'_ch_bw_'+str(c)+'after_training_one_threat.pt'


if __name__ == "__main__":
    bi, li = CyberPhysicalMapping()
    comp_zones = bi
    for i,(k,v) in enumerate(li.items()):
        comp_zones[k] = bi[v[0]]
    comp_zones['C83'] = bi['83']
    comp_zones['C88a'] = bi['88']
    comp_zones['C90b'] = bi['90']
    comp_zones['C92c'] = bi['92']
    exp_tajectories = [500,1000,1500,2000,2500]
    channel_bws = [1000,1500,2000,2500]
    router_qlimits = [80,100,120]
    policy_network_train_len = 20000
    gail_train_len = 30000
    train_and_evaluate(comp_zones, exp_tajectories,channel_bws, router_qlimits,policy_network_train_len,gail_train_len)