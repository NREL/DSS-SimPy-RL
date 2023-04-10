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
    #labels = ['500', '1000', '1500', '2000', '2500']
    #labels = ['50','70','90','110','130','150']
    labels = ['0.2','0.4','0.6','0.8','1.0']
    #labels = ['0.7','0.9','1.1','1.3']
    #labels = ['6', '8', '10', '12', '20']
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars
    print(x)
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, res[0], width, label='Rql=50',color='white', hatch=".",edgecolor='black')
    rects2 = ax.bar(x + width/2, res[1], width, label='Rql=70',color='white',hatch='//',edgecolor='black')
    rects11 = ax.bar(x + (3*width)/2, res[2], width, label='Rql=90',color='white',hatch='*',edgecolor='black')
    rects21 = ax.bar(x + (5*width)/2, res[3], width, label='Rql=110',color='white',hatch='-',edgecolor='black')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Avg. Episode Length',fontsize=14)
    ax.set_ylim((3,4))
    ax.set_xlabel('Probability of selection of R3 over R2',fontsize=14)
    #ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels,fontsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.legend(fontsize=14)

    fig.tight_layout()
    fig.savefig('impact_ideal_router_selection.png',format='png',dpi=600)
    plt.show()

def plot_graphs_mixed_domain_res(res):
    labels = ['500', '1000', '1500', '2000', '2500']
    #labels = ['50','70','90','110','130','150']
    #labels = ['0.2','0.4','0.6','0.8','1.0']
    #labels = ['0.7','0.9','1.1','1.3']
    #labels = ['6', '8', '10', '12', '20']
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars
    print(x)
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, res[0], width, label='N-1',color='white', hatch=".",edgecolor='black')
    rects2 = ax.bar(x + width/2, res[1], width, label='N-2',color='white',hatch='//',edgecolor='black')
    rects11 = ax.bar(x + (3*width)/2, res[2], width, label='N-1, DER outages',color='white',hatch='*',edgecolor='black')
    rects21 = ax.bar(x + (5*width)/2, res[3], width, label='N-3',color='white',hatch='-',edgecolor='black')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Avg. Episode Length',fontsize=14)
    ax.set_ylim((0,55))
    ax.set_xlabel('Channel Bandwidth (Bits/sec)',fontsize=14)
    #ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels,fontsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.legend(fontsize=14)

    fig.tight_layout()
    fig.savefig('cyber_physical_combined_evaluation.png',format='png',dpi=600)
    plt.show()

def plot_graphs_mixed_domain_res2(res):
    labels = ['0','1','2','3']
    #labels = ['50','70','90','110','130','150']
    #labels = ['0.2','0.4','0.6','0.8','1.0']
    #labels = ['0.7','0.9','1.1','1.3']
    #labels = ['6', '8', '10', '12', '20']
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars
    print(x)
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, res[0], width, label='N-1',color='white', hatch=".",edgecolor='black')
    rects2 = ax.bar(x + width/2, res[1], width, label='N-2',color='white',hatch='//',edgecolor='black')
    rects11 = ax.bar(x + (3*width)/2, res[2], width, label='N-1, DER outages',color='white',hatch='*',edgecolor='black')
    rects21 = ax.bar(x + (5*width)/2, res[3], width, label='N-3',color='white',hatch='-',edgecolor='black')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Avg. Episode Length',fontsize=14)
    ax.set_ylim((0,120))
    ax.set_xlabel('Number of Core Routers compromised',fontsize=14)
    #ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels,fontsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.legend(fontsize=14)

    fig.tight_layout()
    fig.savefig('cyber_physical_combined_evaluation_threat_impact.png',format='png',dpi=600)
    plt.show()


def plot_graphs_phy(res):
    labels = ['N-1','N-1, DER outage','N-2','N-3']
    x = np.array(labels)  # the label locations
    """
    width = 0.15  # the width of the bars
    print(x)
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, res[0], width, label='Success Rate',color='white', hatch=".",edgecolor='black')
    rects2 = ax.bar(x + width/2, res[1], width, label='Average Episode Length',color='white',hatch='//',edgecolor='black')
    #rects11 = ax.bar(x + (3*width)/2, res[2], width, label='Rql=90',color='white',hatch='*',edgecolor='black')
    #rects21 = ax.bar(x + (5*width)/2, res[3], width, label='Rql=110',color='white',hatch='-',edgecolor='black')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Avg. Episode Len & Success %',fontsize=14)
    #ax.set_ylim((3,4))
    ax.set_xlabel('Type of Contingency',fontsize=14)
    #ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels,fontsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.legend(fontsize=14) """

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
    #labels = ['500', '1000', '1500', '2000', '2500']
    #labels = ['50','70','90','110','130','150']
    labels = ['BC','CC','EBC','BC_stvy','CC_stvy','EBC_stvy']
    #labels = ['0.7','0.9','1.1','1.3']
    #labels = ['6', '8', '10', '12', '20']
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars
    print(x)
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, res[0], width, label='N-1',color='white', hatch=".",edgecolor='black')
    rects2 = ax.bar(x + width/2, res[2], width, label='N-1, DER Outages',color='white',hatch='//',edgecolor='black')
    rects11 = ax.bar(x + (3*width)/2, res[1], width, label='N-2',color='white',hatch='*',edgecolor='black')
    rects21 = ax.bar(x + (5*width)/2, res[3], width, label='N-3',color='white',hatch='-',edgecolor='black')

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
    fig.savefig('resilience_score_new.png',format='png',dpi=600)
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
    # plt.figure(figsize=(5,4))
    # plt.scatter(x,sol,marker="o");
    # plt.figure(figsize=(5,4))
    # plt.scatter(x,comp,marker="*");
    # ax.set_xticklabels(fontsize=15)
    # ax.set_yticklabels(fontsize=15)
    #ax.legend(font_s)
    ax.set_xticks(x)
    ax.legend(loc='center right')
    twin_ax.legend(loc='upper right')
    plt.savefig('channel_bw_impact.png',dpi=300,bbox_inches='tight')


folder_path=desktop_path+'\\Results\\Aug5'
results=[]
for c in range(1,5,1):
    res_contingency=[]
    for threat in range(0,4,1):
        content = scipy.io.loadmat(folder_path+'\Contingency_Type_'+str(c)+'_Rtr_Comp_'+str(threat)+'_goal_cp_Rql_100_remove_unsucessful.mat')
        res_contingency.append(content['avg_episode_len_all'][0][0])
    results.append(res_contingency)
plot_graphs_mixed_domain_res2(results)
