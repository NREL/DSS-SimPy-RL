# testing open dss env
import os
import sys
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(directory))
sys.path.insert(0,desktop_path+'\ARM_IRL')
from distutils.dir_util import create_tree
from envs.openDSSenv import openDSSenv
import opendssdirect as dss
import random
import scipy.io
import statistics

from agents.dqn import *
from collections import namedtuple, deque 
import torch

der = True

dss_data_dir = desktop_path+'\\ARM_IRL\\cases\\123Bus_Simple\\'
dss_master_file_dir = 'Redirect ' + dss_data_dir + 'IEEE123Master.dss'

if der:
    dss_data_dir = desktop_path+'\\ARM_IRL\\cases\\123Bus_SimpleMod\\'
    dss_master_file_dir = 'Redirect ' + dss_data_dir + 'IEEE123Master.dss'

dss.run_command(dss_master_file_dir)

circuit = dss.Circuit
#critical_loads_bus = ['58','59','99','100','88','93','94','78','48','50', '111','114', '37','39']
critical_loads_bus = ['58','59','99','100','88','93','94','78','48','50', '111','114', '37','39']
capacitor_banks =['C83', 'C88a', 'C90b','C92c']
# switch from and two buses, with the first 6 are normally closed and the last two are normally open
switches = { 0: ['150r','149'], 1: ['13','152'], 2: ['18','135'], 3: ['60','160'], 4: ['97','197'], 5: ['61','61s'], 6: ['151','300'], 7: ['54','94'] }


switch_names =[]
for k,sw in enumerate(switches):
    switch_names.append('Sw'+str(k+1))

#line_faults = ['L55','L68', 'L58', 'L77', 'L45', 'L101', 'L41']
line_faults = ['L55', 'L58', 'L77','L68','L45', 'L101','L41']


episodes = 100
max_episode_len = 100



for c in range(1,5,1): # select contingency
    for load_ub in range(11,12,1):
        der = True

        dss_data_dir = desktop_path+'\\ARM_IRL\\cases\\123Bus_Simple\\'
        dss_master_file_dir = 'Redirect ' + dss_data_dir + 'IEEE123Master.dss'

        if der:
            dss_data_dir = desktop_path+'\\ARM_IRL\\cases\\123Bus_SimpleMod\\'
            dss_master_file_dir = 'Redirect ' + dss_data_dir + 'IEEE123Master.dss'
        
        dss.run_command(dss_master_file_dir)
        circuit = dss.Circuit
        #critical_loads_bus = ['58','59','99','100','88','93','94','78','48','50', '111','114', '37','39']
        critical_loads_bus = ['58','59','99','100','88','93','94','78','48','50', '111','114', '37','39']
        capacitor_banks =['C83', 'C88a', 'C90b','C92c']
        # switch from and two buses, with the first 6 are normally closed and the last two are normally open
        switches = { 0: ['150r','149'], 1: ['13','152'], 2: ['18','135'], 3: ['60','160'], 4: ['97','197'], 5: ['61','61s'], 6: ['151','300'], 7: ['54','94'] }

        ders = ['35','48','64','78','95','108']
        switch_names =[]
        for k,sw in enumerate(switches):
            switch_names.append('Sw'+str(k+1))

        #line_faults = ['L55','L68', 'L58', 'L77', 'L45', 'L101', 'L41']
        line_faults = ['L55', 'L58', 'L77','L68','L45', 'L101','L41']
        #agent = Agent(state_size=env.observation_spaces.shape[0],action_size=len(switch_names) - 1,seed=0)
        env = openDSSenv(_dss = dss, _critical_loads=critical_loads_bus, _line_faults =line_faults, _switch_names = switch_names, _capacitor_banks = capacitor_banks,load_ub=load_ub,_ders=ders)
        eps_start = 1.0
        eps_end = 0.01
        eps_decay =0.996
        scores = [] # list containing score from each episode
        scores_window = deque(maxlen=100) # last 100 scores
        eps = eps_start
        results ={}
        #env.load_lower_bound = load_lb/10
        env.load_upper_bound = load_ub/10
        env.contingency = c
        success= 0
        agg_episode_len = []
        agg_res_1 = []
        agg_res_2 = []
        agg_res_3 = []
        agg_sens_res_1 = []
        agg_sens_res_2 = []
        agg_sens_res_3 = []
        agg_top_1 = []
        agg_top_2 = []
        agg_top_3 = []
        agg_top_4 = []
        agg_sens_top_1 = []
        agg_sens_top_2 = []
        agg_sens_top_3 = []
        agg_sens_top_4 = []
        
        for i in range(episodes):
            #print('Episode {}'.format(i+1))
            #state = np.array(env.reset())
            state = env.reset()
            #print('observation : {}'.format(state))

            done = False
            ctr = 0
            episodic_reward = 0
            ep_len = 0
            res1=0.0
            diff_res1=0.0
            prev_res1=0.0
            res2=0.0
            diff_res2=0.0
            prev_res2=0.0
            res3=0.0
            diff_res3=0.0
            prev_res3=0.0
            top1=0.0
            diff_top1=0.0
            prev_top1=0.0
            top2=0.0
            diff_top2=0.0
            prev_top2=0.0
            top3=0.0
            diff_top3=0.0
            prev_top3=0.0
            top4=0.0
            diff_top4=0.0
            prev_top4=0.0

            switch_selected = []
            while not done and ctr < max_episode_len:
                ctr+=1
                # randomly select an action for time-being until we train an agent
                action = random.choice(switch_names[0:])
                #action = agent.act(state,eps)
                if action in switch_selected:
                    continue
                else:
                    switch_selected.append(action)
                next_state, reward, done,info,_ = env.step(action, result={})
                #agent.step(state,action,reward[0],np.array(next_state),done)
                #state = np.array(next_state)
                if ep_len == 0:
                    diff_res1 = reward[1]
                    diff_res2 = reward[2]
                    diff_res3 = reward[3]
                    diff_top1 = reward[4]
                    diff_top2 = reward[5]
                    diff_top3 = reward[6]
                    diff_top4 = reward[7]
                else:
                    diff_res1+=(reward[1] - prev_res1)
                    diff_res2+=(reward[2] - prev_res2)
                    diff_res3+=(reward[3] - prev_res3)
                    diff_top1+=(reward[4] - prev_top1)
                    diff_top2+=(reward[5] - prev_top2)
                    diff_top3+=(reward[6] - prev_top3)
                    diff_top1+=(reward[7] - prev_top4)
                state=next_state
                score=reward[0]
                episodic_reward += reward[0]
                ep_len+=1
                res1+=reward[1]
                res2+=reward[2]
                res3+=reward[3]
                top1+=reward[4]
                top2+=reward[5]
                top3+=reward[6]
                top4+=reward[7]
                prev_res1 = reward[1]
                prev_res2 = reward[2]
                prev_res3 = reward[3]
                prev_top1 = reward[4]
                prev_top2 = reward[5]
                prev_top3 = reward[6]
                prev_top4 = reward[7]
                scores_window.append(score) ## save the most recent score
                scores.append(score) ## sae the most recent score
                eps = max(eps*eps_decay,eps_end)## decrease the epsilon
                #print('Step {0}: Avg BC: {1} Avg CL : {2} Avg EBC : {3}'.format(ep_len,reward[1],reward[2],reward[3]))
            if ctr < max_episode_len:
                success+=1
                agg_episode_len.append(ctr)
            #print('Sensitivities {0} {1} {2}'.format(diff_res1/ep_len, diff_res2/ep_len,diff_res3/ep_len))
            #agg_episode_len.append(ctr)
            agg_res_1.append(res1/ep_len)
            agg_res_2.append(res2/ep_len)
            agg_res_3.append(res3/ep_len)
            agg_top_1.append(top1/ep_len)
            agg_top_2.append(top2/ep_len)
            agg_top_3.append(top3/ep_len)
            agg_top_4.append(top4/ep_len)
            agg_sens_res_1.append(diff_res1/ep_len)
            agg_sens_res_2.append( diff_res2/ep_len)
            agg_sens_res_3.append(diff_res3/ep_len)
            agg_sens_top_1.append(diff_top1/ep_len)
            agg_sens_top_2.append( diff_top2/ep_len)
            agg_sens_top_3.append(diff_top3/ep_len)
            agg_sens_top_4.append(diff_top4/ep_len)
        print('Case: Contingency: {0}, Load Upper Bnd: {1}, avg episode len: {2}, Success rate: {3}'.format(c,load_ub/10, statistics.mean(agg_episode_len), success/episodes))
        print('Score: BC: {0}, CL: {1}, EBC: {2}'.format(statistics.mean(agg_res_1),statistics.mean(agg_res_2),statistics.mean(agg_res_3)))
        print('Topo Score: : Avg. Clustering {0}, Graph Diameter: {1}, Path Length: {2} Algebraic Connectivity: {3}'.format(statistics.mean(agg_top_1),statistics.mean(agg_top_2),statistics.mean(agg_top_3),statistics.mean(agg_top_4)))
        print('***************************************************************************************************')
        results['succ_rate'] = success/episodes
        results['avg_episode_len'] = statistics.mean(agg_episode_len)
        results['var_episode_len'] = statistics.variance(agg_episode_len)
        results['avg_bc'] = statistics.mean(agg_res_1)
        results['var_bc'] = statistics.variance(agg_res_1)
        results['avg_cl'] = statistics.mean(agg_res_2)
        results['var_cl'] = statistics.variance(agg_res_2)
        results['avg_ebc'] = statistics.mean(agg_res_3)
        results['var_cl'] = statistics.variance(agg_res_3)
       
        results['avg_bc_sens'] = statistics.mean(agg_sens_res_1)
        results['var_bc_sens'] = statistics.variance(agg_sens_res_1)
        results['avg_cl_sens'] = statistics.mean(agg_sens_res_2)
        results['var_cl_sens'] = statistics.variance(agg_sens_res_2)
        results['avg_ebc_sens'] = statistics.mean(agg_sens_res_3)
        results['var_cl_sens'] = statistics.variance(agg_sens_res_3)

        results['avg_clus'] = statistics.mean(agg_top_1)
        results['var_clus'] = statistics.variance(agg_top_1)
        results['avg_gd'] = statistics.mean(agg_top_2)
        results['var_gd'] = statistics.variance(agg_top_2)
        results['avg_pl'] = statistics.mean(agg_top_3)
        results['var_pl'] = statistics.variance(agg_top_3)
        results['avg_ac'] = statistics.mean(agg_top_4)
        results['var_ac'] = statistics.variance(agg_top_4)
        results['avg_clus_sens'] = statistics.mean(agg_sens_top_1)
        results['var_clus_sens'] = statistics.variance(agg_sens_top_1)
        results['avg_gd_sens'] = statistics.mean(agg_sens_top_2)
        results['var_gd_sens'] = statistics.variance(agg_sens_top_2)
        results['avg_pl_sens'] = statistics.mean(agg_sens_top_3)
        results['var_pl_sens'] = statistics.variance(agg_sens_top_4)
        results['avg_ac_sens'] = statistics.mean(agg_sens_top_4)
        results['var_ac_sens'] = statistics.variance(agg_sens_top_4)

        scipy.io.savemat('Contingency_Type_'+str(c)+'_goal_volt_satisfy_Topo_Resilience_Info_with_sensitivity.mat',results)




