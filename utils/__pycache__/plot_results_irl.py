from cProfile import label
import statistics
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(os.path.dirname(directory)))

def plot_graphs(res):
    labels = ['1000','1500', '2000', '2500']
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars
    print(x)
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, res[0], width, label='Random',color='white', hatch=".",edgecolor='black')
    rects2 = ax.bar(x + width/2, res[1], width, label='PPO',color='white',hatch='//',edgecolor='black')
    rects11 = ax.bar(x + (3*width)/2, res[2], width, label='Expert',color='white',hatch='*',edgecolor='black')
    rects21 = ax.bar(x + (5*width)/2, res[3], width, label='GAIL',color='white',hatch='-',edgecolor='black')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Avg. Episode Length',fontsize=14)
    #ax.set_ylim((0,40))
    ax.set_ylim((0,120))
    ax.set_xlabel('Channel Bandwidth (in Bits/sec)',fontsize=14)
    #ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels,fontsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.legend(fontsize=14)

    fig.tight_layout()
    fig.savefig('evaluation_gail_ieee123_a2c_medium_trainining.png',format='png',dpi=600)
    #fig.savefig('evaluation_bc_ieee123_improved_trainining.png',format='png',dpi=600)
    plt.show()

def plot_graphs_phy(res):
    labels = ['N-1','N-1, DER outage','N-2','N-3']
    x = np.array(labels)  # the label locations

    fig, ax = plt.subplots(1, 1)
    twin_ax = ax.twinx()
    ax.plot(x,res[0],'bs-',label='Success Rate')
    ax.set_ylabel('Success Rate (in %)',fontsize=14)
    twin_ax.plot(x,res[1],'rs-',label='Avg. Episode Len')
    twin_ax.set_ylabel('Avg. Episode Len',fontsize=14)
    ax.set_xlabel('Type of Contingency',fontsize=14)
    twin_ax.tick_params(axis='y',colors='red',labelsize=13)
    ax.tick_params(axis='y',colors='blue',labelsize=13)
    ax.tick_params(axis='x',colors='black',labelsize=13)
    ax.legend(fontsize=12)
    twin_ax.legend(fontsize=12)
    ax.set_xticks(x)
    ax.legend(loc='center right')
    twin_ax.legend(loc='lower center')
    plt.savefig('impact_of_contingency.png',dpi=300,bbox_inches='tight')

def plot_graphs_compare_mininet(res):
    labels = ['500','1000','1500','2000','2500']
    x = np.array(labels)  # the label locations

    fig, ax = plt.subplots(1, 1)
    #twin_ax = ax.twinx()
    ax.plot(x,res[0],'bs-',label='Mininet')
    ax.set_ylabel('Latency (in milliseconds)',fontsize=14)
    ax.plot(x,res[1],'rx-',label='Simpy')

    ax.set_xlabel('Channel Bandwidth (in Bits/sec)',fontsize=14)

    ax.tick_params(axis='y',colors='blue',labelsize=13)
    ax.tick_params(axis='x',colors='black',labelsize=13)
    ax.legend(fontsize=12)

    ax.set_xticks(x)
    ax.legend(loc='center right')

    plt.savefig('comparison_with_mininet.png',dpi=300,bbox_inches='tight')

def plot_graphs_res_metric(res):
    labels = ['BC','CL','EBC','BC_stvy','CL_stvy','EBC_stvy']
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars
    print(x)
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, res[0], width, label='N-1',color='white', hatch=".",edgecolor='black')
    rects2 = ax.bar(x + width/2, res[1], width, label='N-2',color='white',hatch='//',edgecolor='black')
    rects11 = ax.bar(x + (3*width)/2, res[2], width, label='N-1, CB Outage',color='white',hatch='*',edgecolor='black')
    #rects21 = ax.bar(x + (5*width)/2, w, width, label='u=5',color='white',hatch='-',edgecolor='black')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores',fontsize=14)
    #ax.set_ylim((0,110))
    ax.set_xlabel('Resilience Metric and its Sensitivity',fontsize=14)
    #ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels,fontsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.legend(fontsize=14)

    fig.tight_layout()
    fig.savefig('resilience_score.png',format='png',dpi=600)
    plt.show()

def plot_twin_axis(data):
    x = ['500', '1000', '1500', '2000', '2500']
    sr = data[1]
    avg_len = data[2]
    latency = data[0]

    fig, ax = plt.subplots(1, 1)
    twin_ax = ax.twinx()
    ax.plot(x,sr,'bx-',label='Success Rate')
    ax.plot(x,avg_len,'gx-',label='Avg Episode Length')
    ax.set_ylabel('Success Rate & Avg. Episode Length',fontsize=14)
    ax.set_xlabel('Channel Bandwidth',fontsize=14)
    twin_ax.plot(x,latency,'r*-',label='Latency')
    twin_ax.set_ylabel('Latency',fontsize=14)
    twin_ax.tick_params(axis='y',colors='red',labelsize=13)
    ax.tick_params(axis='y',colors='blue',labelsize=13)
    ax.tick_params(axis='x',colors='black',labelsize=13)
    ax.legend(fontsize=12)
    twin_ax.legend(fontsize=12)
    ax.set_xticks(x)
    ax.legend(loc='center right')
    twin_ax.legend(loc='upper right')
    plt.savefig('channel_bw_impact.png',dpi=300,bbox_inches='tight')


folder_path=desktop_path+'\\Results\\Aug4'

results=[]
for ch in range(1000,2600,500):
    results_random = []
    results_expert=[]
    results_ppo=[]
    results_airl=[]
    # change the contingency type
    for rq in range(80,130,20):
        try:
            #content = scipy.io.loadmat(folder_path+'\AIRL_Actual_Expert_C_ieee123_qlimit_'+str(rq)+'_ch_BW_'+str(ch)+'goal_5pkt_each_single_threat.mat')
            #content = scipy.io.loadmat(folder_path+'\Dagger_Actual_Expert_CM_2_medium_training_qlimit_'+str(rq)+'_ch_BW_'+str(ch)+'goal_5pkt_each_single_threat.mat')
            #content = scipy.io.loadmat(folder_path+'\GAIL_Actual_Expert_C_ieee123_long_training_qlimit_'+str(rq)+'_ch_BW_'+str(ch)+'goal_5pkt_each_single_threat.mat')
            #content = scipy.io.loadmat(folder_path+'\BC_Actual_Expert_C_ieee123_long_training_qlimit_'+str(rq)+'_ch_BW_'+str(ch)+'goal_5pkt_each_single_threat.mat')
            #content = scipy.io.loadmat(folder_path+'\BC_Actual_Expert_CM_2_long_training_qlimit_'+str(rq)+'_ch_BW_'+str(ch)+'goal_5pkt_each_single_threat.mat')
            #content = scipy.io.loadmat(folder_path+'\GAIL_Actual_Expert_CM_2_medium_training_qlimit_'+str(rq)+'_ch_BW_'+str(ch)+'goal_5pkt_each_single_threat.mat')
            content = scipy.io.loadmat(folder_path+'\AIRL_Actual_Expert_C_ieee123_long_training_qlimit_'+str(rq)+'_ch_BW_'+str(ch)+'goal_5pkt_each_single_threat.mat')
            #content = scipy.io.loadmat(folder_path+'\AIRL_Actual_Expert_CM_2_qlimit_'+str(rq)+'_ch_BW_'+str(ch)+'goal_5pkt_each_single_threat.mat')
            #print(content)
            results_random.append(content['avg_episode_length_random'][0][0])
            #results_random.append(content['succ_avg_episode_length_random'][0][0])
            results_ppo.append(content['succ_epi_len_before_training'][0][0])
            #results_ppo.append(content['episode_length_before_training'][0][0])
            #episode_length_before_training
            #results_expert.append(content['succ_avg_episode_length_expert'][0][0])
            results_expert.append(content['avg_episode_length_expert'][0][0])
            #episode_length_after_training
            results_airl.append(content['succ_epi_len_after_training'][0][0])
            #results_airl.append(content['episode_length_after_training'][0][0])
        except:
            #print()
            pass
    results.append([statistics.mean(results_random),statistics.mean(results_ppo),statistics.mean(results_expert),statistics.mean(results_airl)])
#print(results)
results=[list(i) for i in zip(*results)]
print(results)
plot_graphs(results)

        
