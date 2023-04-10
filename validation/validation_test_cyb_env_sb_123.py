# testing cyber simpy based env with varying parameters such as channel Bandwidth, R3/R2 selection ration, Q limit of router etc

#from envs.simpy_env.CyberEnv import CyberEnv

#from envs.simpy_env.CyberWithChannelEnv import CyberEnv
import os
import sys
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(directory))
sys.path.insert(0,desktop_path+'\ARM_IRL')
from torch import dtype
from envs.simpy_env.CyberWithChannelEnvSB_123 import CyberEnv

import random
import networkx as nx
import numpy as np
from envs.simpy_env.generate_network import create_network,draw_cyber_network,create_network2
import scipy.io

smallCase = False


# Change the Channel Bandwidth from 500 to 2500 bits/ 25 sec
for c in range(500,2600,500):
    print('Channel BW : {0}'.format(c))
    for r in range(50,160,20):
        results ={}
        if smallCase:
            env = CyberEnv(channelModel=True,envDebug=False, R2_qlimit=r, ch_bw = c)

        else:
            G = create_network2()
            env = CyberEnv(provided_graph=G,channelModel=True,envDebug=False, R2_qlimit=r, ch_bw = c,with_threat=True)
            #draw_cyber_network(env.G)

        episodes = 100
        max_episode_len = 100

        """ for router in env.routers:
            print(router.id)
            for ch in router.out_channel:
                print(ch.cid) """
        
        #break
        success= 0
        agg_episode_len = 0
        agg_rtr2_drop_rate = 0
        agg_wait_rate = 0
        agg_ur ={}
        for i in range(episodes):
            
            #print('****************')
            state = env.reset()
            #print('observation : {}'.format(state))
            """ print('Episode {}'.format(i+1))
            for router in env.routers:
                print(router.qlimit) """

            done = False
            ctr = 0
            episodic_reward = 0
            selected_ideal = 0
            episodic_ur={}
            while not done and ctr < max_episode_len:
                ctr+=1
                if smallCase:
                    action_index = random.randint(0,1)
                    # randomly select an action for time-being until we train an agent      
                    action = {'device':0, 'next_hop':action_index}
                    action = [0, action_index]
                    if action_index ==1:
                        selected_ideal+=1

                else:
                    # randomly pick a router to modify the next_hop
                    router_id = random.randint(0,env.deviceCount-1)
                    
                    # currently random:  to implement  get the next hop from the shortest path algorithm
                    rnd_action_index = random.randint(0, len(env.routers[router_id].out)-1)

                    #shortest_path_action_index = nx.single_source_shortest_path(env.G, router_id)['PS'][1]
                    path_to_receiver =nx.single_source_shortest_path(env.G, router_id)['PS']

                    shortest_path_action_index = path_to_receiver[1]

                    #rnd_action = {'device':router_id, 'next_hop':rnd_action_index}
                    rnd_action = [router_id,rnd_action_index]

                    if shortest_path_action_index != 'PS':
                        rtr_id = 'R'+str(shortest_path_action_index)
                        rtr_ix = [ix for (ix,item) in enumerate(env.routers[router_id].out) if item.id == rtr_id][0]
                        #action = {'device':router_id, 'next_hop':rtr_ix}
                        action = [router_id,rtr_ix]
                    else:
                        action = rnd_action
                        #action = np.array([router_id,rnd_action_index], dtype=np.int32)
                    """ if rnd_action["next_hop"] != action["next_hop"]:
                        print('Selected Action : {}'.format(rnd_action))
                        print('Selected optimal action :{}'.format(action)) """


                next_state, reward, done, info = env.step(action, result={})
                #print('State : {0}, Next-State : {1}, Reward : {2}, Done : {3}, Info :{4}'.format(state, next_state, reward, done, info))

                # based on the channel utilization factor, update the weights of the graph of the environment
                #utilization_rates = next_state[env.deviceCount:]
                utilization_rates = next_state
                #episodic_ur+=np.average(utilization_rates)
                #print(env.channel_map)
                G_to_update = env.G
                for i,data in enumerate(G_to_update.edges(data=True)):
                    #print(data)
                    # get the index from the channel map
                    src_dest = []
                    if 'PS' in str(data[1]):
                        src_dest.append('R'+str(data[0]))
                        src_dest.append(str(data[1]))
                    elif 'PG' in str(data[1]):
                        src_dest.append(data[1])
                        src_dest.append('R'+str(data[0]))
                    else:
                        src_dest.append('R'+str(data[1]))
                        src_dest.append('R'+str(data[0]))
                    #print(src_dest)
                    ix = [k for (k,v) in env.channel_map.items() if v == src_dest][0]
                    #print(ix)
                    G_to_update[data[0]][data[1]].update({'weight':utilization_rates[ix]})
                    if str(data[0])+'_'+str(data[1]) in episodic_ur.keys():
                        episodic_ur[str(data[0])+'_'+str(data[1])]+=utilization_rates[ix]
                    else:
                        episodic_ur[str(data[0])+'_'+str(data[1])] =utilization_rates[ix]
                env.G = G_to_update
                state = next_state
                episodic_reward += reward

            if smallCase:
                #print('Episode Length : '+str(ctr))
                agg_episode_len+=ctr
                if ctr < max_episode_len:
                    success +=1
                #print('selected ideal '+str(selected_ideal))
                #print("Last 10 waits: "  + ", ".join(["{:.3f}".format(x) for x in env.interpreter.waits[-10:]]))
                """ 
                try:
                    print("Router : ",str(env.routers[0].id))
                    print("received: {}, dropped {}, sent {}".format(env.routers[0].packets_rec, env.routers[0].packets_drop, env.senders[0].packets_sent))
                    print("loss rate: {}".format(float(env.routers[0].packets_drop)/env.routers[0].packets_rec))
                except:
                    pass
                """
                try:
                    #print("Router : ",str(env.routers[1].id))
                    #print("received: {}, dropped {}, sent {}".format(env.routers[1].packets_rec, env.routers[1].packets_drop, env.senders[1].packets_sent))
                    #print("loss rate: {}".format(float(env.routers[1].packets_drop)/env.routers[1].packets_rec))
                    agg_rtr2_drop_rate+=float(env.routers[1].packets_drop)/env.routers[1].packets_rec
                except:
                    pass
                """
                try:
                    print("Router : ",str(env.routers[2].id))
                    print("received: {}, dropped {}".format(env.routers[2].packets_rec, env.routers[2].packets_drop))
                    print("loss rate: {}".format(float(env.routers[2].packets_drop)/env.routers[2].packets_rec))
                except:
                    pass
                
                try:
                    print("Router : ",str(env.routers[3].id))
                    print("received: {}, dropped {}".format(env.routers[3].packets_rec, env.routers[3].packets_drop))
                    print("loss rate: {}".format(float(env.routers[3].packets_drop)/env.routers[3].packets_rec))
                except:
                    pass
                try:
                    print("Last 10 sink arrival times: " + ", ".join(["{:.3f}".format(x) for x in env.interpreter.arrivals[-10:]]))
                    print("average wait = {:.3f}".format(sum(env.interpreter.waits)/len(env.interpreter.waits)))
                except:
                    pass """

            else:
                #break
                agg_wait_rate += sum(env.interpreter.waits)/len(env.interpreter.waits)
                agg_episode_len+=ctr
                if ctr < max_episode_len:
                    success +=1
                #print('Episode Length : '+str(ctr))
                dr=0
                for rtr in env.routers:
                    #print("Router : ",str(rtr.id))
                    #print("received: {}, dropped {}".format(rtr.packets_rec, rtr.packets_drop))
                    try:
                        #print("loss rate: {}".format(float(rtr.packets_drop)/rtr.packets_rec))
                        dr+=float(rtr.packets_drop)/rtr.packets_rec
                    except:
                        pass
                agg_rtr2_drop_rate+=dr/len(env.routers)
                for k,v in episodic_ur.items():
                    if k in agg_ur.keys():
                        agg_ur[k]+=v/ctr
                    else:
                        agg_ur[k] = v/ctr

        for k,v in agg_ur.items():
            agg_ur[k] = v/episodes

        print('Case: Qlimit: {0}, success rate: {1}, avg episode len: {2}, avg drop rate: {3}, avg wait: {4}, avg utilization rate: {5}'.format(r,success/episodes, agg_episode_len/episodes, agg_rtr2_drop_rate/episodes,agg_wait_rate/episodes, agg_ur))
        print('***************************************************************************************************')
        results['succ_rate'] = success/episodes
        results['avg_episode_len'] = agg_episode_len/episodes
        results['avg_rtr_drop_rate'] = agg_rtr2_drop_rate/episodes
        results['avg_wait'] = agg_wait_rate/episodes
        results['agg_ur'] = agg_ur
        scipy.io.savemat('qlimit_'+str(r)+'_ch_BW_'+str(c)+'goal_total_70_pkt_exp_size_mean_25_no_threat_no_delay.mat',results)




