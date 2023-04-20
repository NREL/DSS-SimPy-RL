import sys
from numpy import average
import os
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(directory)))))
sys.path.insert(0,desktop_path+'\ARM_IRL')
sys.path.insert(1,os.path.dirname(directory))
sys.path.insert(2,os.path.dirname(directory)+'\experts')
import bc
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import gym
import tempfile
import scipy.io
import opendssdirect as dss
import rollout
from wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from torch import jit,tensor
from dagger import SimpleDAggerTrainer
from reward_nets import BasicShapedRewardNet
from networks import RunningNorm
from generate_expert_demonstrations_feeder import expert_rollouts
from envs.openDSSenvSB_DiscreteSpace import openDSSenv
#from envs.simpy_env.generate_network import create_network2
#from CPEnv_DiscreteDSS_RtrDropRate import CyberPhysicalEnvMT,CyberPhysicalMapping
import torch
# bigger case
#G = create_network()
#env = CyberEnv(provided_graph=G)

def evaluate_policy(env,model):
    """Evaluating the learned policy by running experiments in the environment
    
        :param env: The OpenDSS RL environment
        :type env: Gym.Env
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
        state = obs = env.reset()
        #print('Episode ===========================> {0}'.format(i+1))
        done = False
        episode_length = 0
        episodic_reward = 0
        while not done and episode_length < max_t:
            #print(obs)
            action, _states = model.predict(obs)
            #print(action)
            action = 'Sw'+str(action+1)
            obs, reward, done, info = env.step(action,result={})
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



def train_and_evaluate(env, exp_trajectory_len, policy_net_train_len, dagger_train_lens):
    """For different combination of expert demonstrations, train and test the policy and saves the results.
    
        :param env: The OpenDSS RL environment
        :type env: gym.Env
        :param exp_tajectory_len: The number of expert demonstrations steps considered for AIRL training
        :type exp_tajectory_len: int
        :param policy_net_train_len: Samples considered for training the policy network fed in the generator network as initial policy
        :type policy_net_train_len: int
        :param dagger_train_lens: Samples considered for training DAgger 
        :type dagger_train_lens: int
        :return: Nothing
        :rtype: None
    """
    results={}
    avg_epi_len_random, succ_avg_epi_len_random, rollouts_random = expert_rollouts(env,episodes=500,_expert = False)
    results['avg_episode_length_random']= avg_epi_len_random
    results['succ_episode_length_random']= succ_avg_epi_len_random
    avg_epi_len_expert, succ_avg_epi_len_expert, rollouts_expert = expert_rollouts(env,episodes=exp_trajectory_len,_expert = True)
    results['avg_episode_length_expert']= avg_epi_len_expert
    results['succ_episode_length_expert']=succ_avg_epi_len_expert 
    
    venv = DummyVecEnv([lambda: env])
    bc_trainer = bc.BC(observation_space=env.observation_space,action_space=env.action_space,)
    # this is the generator or the policy optimizer 
    ppo_model = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
    )

    ppo_model.learn(policy_net_train_len)

    print('Defined the expert policy')

    episode_len_before_training,succ_epi_len_before_training,learner_rewards_before_training = evaluate_policy(env,ppo_model)

    print(f"Reward before training: {learner_rewards_before_training}, avg : {average(learner_rewards_before_training)}")
    print(f"Episode length before training: {episode_len_before_training}, avg : {average(episode_len_before_training)}")
    results['rw_before_training'] = learner_rewards_before_training
    results['succ_epi_len_before_training'] = succ_epi_len_before_training
    results['episode_length_before_training'] = episode_len_before_training

    # this is the discriminator Network
    # normalize_input_layer defines how the input feature of the reward network/discriminator is performed
    reward_net = BasicShapedRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
    )
    print('Defined the reward model')

    for tr_len in dagger_train_lens:
        with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
                #print(tmpdir)
                dagger_trainer = SimpleDAggerTrainer(venv=venv, scratch_dir=tmpdir, expert_policy=ppo_model, bc_trainer=bc_trainer,expert_trajs=rollouts_expert)

        print('Starting Dagger Training')
        dagger_trainer.train(tr_len)

        episode_len_after_training,succ_epi_len_after_training,learner_rewards_after_training = evaluate_policy(env,dagger_trainer.policy)
        print(f"Reward after training: {learner_rewards_after_training}, avg : {average(learner_rewards_after_training)}")
        print(f"Episode length after training: {episode_len_after_training} , avg : {average(episode_len_after_training)}")
        results['rw_after_training'] = learner_rewards_after_training
        results['succ_epi_len_after_training'] = succ_epi_len_after_training
        results['episode_length_after_training'] = episode_len_after_training
        scipy.io.savemat('Dagger_ieee123_feeder_tr_len_'+str(tr_len)+'.mat',results)

if __name__ == "__main__":
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

    env = openDSSenv(_dss = dss, _critical_loads=critical_loads_bus, _line_faults =line_faults, _switch_names = switch_names, _capacitor_banks = capacitor_banks)
    env.contingency = 1 # causes only one line fault
    exp_trajectory_len = 1000
    ppo_train_len=2000
    dagger_train_lens =[5000,10000,15000,20000,25000,30000]
    train_and_evaluate(env,exp_trajectory_len,ppo_train_len,dagger_train_lens)



