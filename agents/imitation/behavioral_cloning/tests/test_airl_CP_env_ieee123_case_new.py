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

from airl import AIRL
from reward_nets import BasicShapedRewardNet
from networks import RunningNorm

from envs.openDSSenvSB_DiscreteSpace import openDSSenv
from envs.simpy_env.CyberWithChannelEnvSB_123 import CyberEnv
from envs.simpy_env.generate_network import create_network2
from envs.simpy_dss.CPEnv_DiscreteDSS_RtrDropRate import CyberPhysicalEnvMT,CyberPhysicalMapping
import torch
# bigger case
#G = create_network()
#env = CyberEnv(provided_graph=G)

bi, li = CyberPhysicalMapping()
comp_zones = bi
for i,(k,v) in enumerate(li.items()):
    comp_zones[k] = bi[v[0]]

comp_zones['C83'] = bi['83']
comp_zones['C88a'] = bi['88']
comp_zones['C90b'] = bi['90']
comp_zones['C92c'] = bi['92']

for c in range(1500,2600,500):
    print('Channel BW : {0}'.format(c))
    for r in range(120,140,20):
        results = {}
        G = create_network2()
        cenv = CyberEnv(provided_graph=G, channelModel=False,envDebug=False, R2_qlimit=r, ch_bw = c,with_threat=False,comp_zones=comp_zones)

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

        expert = PPO(
            policy=MlpPolicy,
            env=cyber_phy_env,
            seed=0,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.0003,
            n_epochs=10,
            n_steps=64,
        )
        expert.learn(1000)  # Note: set to 100000 to train a proficient expert

        print(' ppo expert learned')
        rollouts = rollout.rollout(
            expert,
            DummyVecEnv([lambda: RolloutInfoWrapper(CyberPhysicalEnvMT(cenv, penv,comp_zones))]),
            rollout.make_sample_until(min_timesteps=None, min_episodes=100),
        )
        print('executing rollout')

        for ro in rollouts:
            print(ro)

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

        # this is the discriminator Network
        # normalize_input_layer defines how the input feature of the reward network/discriminator is performed
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

        #print("Reward Network/Discriminator Parameter before training")
        #print(list(reward_net.parameters()))

        #print("Policy Network/Generator Parameter before training")
        #print(learner.get_parameters())

        learner_rewards_before_training, episode_len_before_training = evaluate_policy(
            learner, venv, 100, return_episode_rewards=True
        )

        print(f"Reward before training: {learner_rewards_before_training}, avg : {average(learner_rewards_before_training)}")
        print(f"Episode length before training: {episode_len_before_training}, avg : {average(episode_len_before_training)}")
        results['rw_before_training'] = learner_rewards_before_training
        results['episode_length_before_training'] = episode_len_before_training
        airl_trainer.train(300000)  # Note: set to 300000 for better results

        #print("Reward Network/Discriminator Parameter after training")
        #print(list(airl_trainer._reward_net.parameters()))

        #print("Policy Network/Generator Parameter before training")
        #print(airl_trainer.gen_algo.get_parameters())

        """ learner_rewards_after_training, _ = evaluate_policy(
            learner, venv, 100, return_episode_rewards=True
        ) """

        learner_rewards_after_training, episode_len_after_training = evaluate_policy(
            airl_trainer.gen_algo, venv, 100, return_episode_rewards=True
        )

        print(f"Reward after training: {learner_rewards_after_training}, avg : {average(learner_rewards_after_training)}")
        print(f"Episode length after training: {episode_len_after_training} , avg : {average(episode_len_after_training)}")
        results['rw_after_training'] = learner_rewards_after_training
        results['episode_length_after_training'] = episode_len_after_training
        scipy.io.savemat('CPS_AIRL_qlimit_'+str(r)+'_ch_BW_'+str(c)+'goal__random_rtr_no_threat.mat',results)
        discrim ='reward_qlimit_'+str(r)+'_ch_bw_'+str(c)+'after_training_no_threat.pt'
        generat='policy_qlimit_'+str(r)+'_ch_bw_'+str(c)+'after_training_no_threat.pt'
        torch.save(airl_trainer._reward_net,desktop_path+'\\Results\\models\\cps_'+discrim)
        torch.save(airl_trainer.gen_algo.policy,desktop_path+'\\Results\\models\\cps_'+generat)