# testing cyber simpy based env
import os
import sys
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(directory))
sys.path.insert(0,desktop_path+'\ARM_IRL')
from envs.simpy_env.CyberEnv import CyberEnv
import random
import networkx as nx
import numpy as np
from envs.simpy_env.generate_network import create_network,draw_cyber_network


smallCase = False

if smallCase:
    env = CyberEnv()

else:
    G = create_network()
    env = CyberEnv(provided_graph=G)
    #draw_cyber_network(env.G)

episodes = 20
max_episode_len = 50



for i in range(episodes):
    print('Episode {}'.format(i+1))
    print('****************')
    state = env.reset()
    print('observation : {}'.format(state))
    action_index = random.randint(0,1)
    done = False
    ctr = 0
    episodic_reward = 0
    while not done and ctr < max_episode_len:
        ctr+=1
        if smallCase:
            # randomly select an action for time-being until we train an agent      
            action = {'device':0, 'next_hop':action_index}
        else:
            # randomly pick a router to modify the next_hop
            router_id = random.randint(0,env.deviceCount-1)
            
            # currently random:  to implement  get the next hop from the shortest path algorithm
            rnd_action_index = random.randint(0, len(env.routers[router_id].out)-1)

            path_to_receiver =nx.single_source_shortest_path(env.G, router_id)['PS']

            shortest_path_action_index = path_to_receiver[1]

            rnd_action = {'device':router_id, 'next_hop':rnd_action_index}

            if shortest_path_action_index != 'PS':
                # get the index of the next hop router 
                rtr_id = 'R'+str(shortest_path_action_index)
                rtr_ix = [ix for (ix,item) in enumerate(env.routers[router_id].out) if item.id == rtr_id][0]
                action = {'device':router_id, 'next_hop':rtr_ix}
            else:
                action = rnd_action
            if rnd_action["next_hop"] != action["next_hop"]:
                print('Selected Action : {}'.format(rnd_action))
                print('Selected optimal action :{}'.format(action))

        next_state, reward, done, info = env.step(action, result={})[0]
        print('State : {0}, Next-State : {1}, Reward : {2}, Done : {3}, Info :{4}'.format(state, next_state, reward, done, info))
        state = next_state
        episodic_reward += reward

    if smallCase:
        print('Episode Length : '+str(ctr))

        print("Last 10 waits: "  + ", ".join(["{:.3f}".format(x) for x in env.interpreter.waits[-10:]]))
        print("Router : ",str(env.routers[0].id))
        print("received: {}, dropped {}, sent {}".format(env.routers[0].packets_rec, env.routers[0].packets_drop, env.senders[0].packets_sent))
        print("loss rate: {}".format(float(env.routers[0].packets_drop)/env.routers[0].packets_rec))
        print("Router : ",str(env.routers[1].id))
        print("received: {}, dropped {}, sent {}".format(env.routers[1].packets_rec, env.routers[1].packets_drop, env.senders[1].packets_sent))
        print("loss rate: {}".format(float(env.routers[1].packets_drop)/env.routers[1].packets_rec))
        try:
            print("Router : ",str(env.routers[2].id))
            print("received: {}, dropped {}".format(env.routers[2].packets_rec, env.routers[2].packets_drop))
            print("loss rate: {}".format(float(env.routers[2].packets_drop)/env.routers[2].packets_rec))
        except:
            pass
        print("Router : ",str(env.routers[3].id))
        print("received: {}, dropped {}".format(env.routers[3].packets_rec, env.routers[3].packets_drop))
        print("loss rate: {}".format(float(env.routers[3].packets_drop)/env.routers[3].packets_rec))

        print("Last 10 sink arrival times: " + ", ".join(["{:.3f}".format(x) for x in env.interpreter.arrivals[-10:]]))
        print("average wait = {:.3f}".format(sum(env.interpreter.waits)/len(env.interpreter.waits)))

    else:
        for r in env.routers:
            print("Router : ",str(r.id))
            print("received: {}, dropped {}".format(r.packets_rec, r.packets_drop))
            try:
                print("loss rate: {}".format(float(r.packets_drop)/r.packets_rec))
            except:
                pass

    

    




