# in this code we will try to dynamically change routes based on the packet drop rate caused due to network congestion

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:08:05 2022

@author: abhijeetsahu

A sample network with scenario

    PG2---> SW2
           /   \
          /     \
PG--->SW1      SW4 --> PS
          \     /
           \   /
            SW3

Scenario description : There are two packet sender : PG and PG2, packet receiver : PS
And there are 4 routers/switch : SW1, SW2, SW3, and SW4

The simple testing : Whenever the packet loss count at SW 2 raise beyond a threshold all the traffic originated from PG to go to PS
will be redirected from PG (SW1--> SW2 --> SW4 ---> PS) , to (SW1--> SW3 --> SW4 ---> PS)

Further the task would be to update the SimComponentsDynamic class to add the RL env
"""

import random
import functools
import simpy
import matplotlib.pyplot as plt
from SimComponentsDynamic import PacketGenerator, PacketSink, SwitchPort, PortMonitor
import networkx as nx

def constArrival():
    return 1.5    # time interval

def constSize():
    return 100.0  # bytes

adist = functools.partial(random.expovariate, 0.5)
sdist = functools.partial(random.expovariate, 0.01)  # mean size 100 bytes
samp_dist = functools.partial(random.expovariate, 1.0)
port_rate = 1000.0

env = simpy.Environment()  # Create the SimPy environment
# Create the packet generators and sink

G = nx.Graph()
nodes = []

# A simple mesh network is created 
ps = PacketSink(env, "PS",debug=False, rec_arrivals=True)

pg = PacketGenerator(env, "PG1", adist, sdist)
pg2 = PacketGenerator(env, "PG2", constArrival, constSize)
#pg3 = PacketGenerator(env, "PG3", constArrival, constSize)


sw1 = SwitchPort(env, "R1", rate=400.0, qlimit=300)
sw2 = SwitchPort(env, "R2",rate=400.0, qlimit=300)
sw3 = SwitchPort(env, "R3",rate=300.0, qlimit=300)
sw4 = SwitchPort(env, "R4", rate=300.0, qlimit=300)

nodes += [ps,pg,pg2,sw1,sw2,sw3,sw4]
for node in nodes:
    G.add_node(node.id)
#G.add_nodes_from(nodes)

# Wire packet generators, switch ports, and sinks together
pg.out = sw1
pg2.out = sw2
#pg3.out = sw3
sw1.out.append(sw2)
sw1.out.append(sw3)
sw2.out.append(sw4)
sw3.out.append(sw4)
sw4.out.append(ps)

edges = [(pg,sw1), (pg2,sw2), (sw1,sw2), (sw1,sw3), (sw2,sw4), (sw3,sw4), (sw4,ps)]
for edge in edges:
    G.add_edge(edge[0].id,edge[1].id)

#G.add_edges_from(edges)
nx.draw_networkx(G, with_labels=True, pos=nx.spring_layout(G))
plt.savefig('cyber_network.png', dpi=150)

# Using a PortMonitor to track queue sizes over time
pm1 = PortMonitor(env, sw1, samp_dist)
pm2 = PortMonitor(env, sw2, samp_dist)
pm3 = PortMonitor(env, sw3, samp_dist)
pm4 = PortMonitor(env, sw4, samp_dist)

# Run it
env.run(until=8000)
print("Last 10 waits: "  + ", ".join(["{:.3f}".format(x) for x in ps.waits[-10:]]))
print("Last 10 queue sizes: {}".format(pm1.sizes[-10:]))
print("received: {}, dropped {}, sent {}".format(sw1.packets_rec, sw1.packets_drop, pg.packets_sent))
print("loss rate: {}".format(float(sw1.packets_drop)/sw1.packets_rec))
print("average system occupancy: {:.3f}".format(float(sum(pm1.sizes))/len(pm1.sizes)))


print("Last 10 queue sizes: {}".format(pm2.sizes[-10:]))
print("received: {}, dropped {}, sent {}".format(sw2.packets_rec, sw2.packets_drop, pg2.packets_sent))
print("loss rate: {}".format(float(sw2.packets_drop)/sw2.packets_rec))
print("average system occupancy: {:.3f}".format(float(sum(pm2.sizes))/len(pm2.sizes)))


print("Last 10 queue sizes: {}".format(pm3.sizes[-10:]))
#print("received: {}, dropped {}, sent {}".format(sw3.packets_rec, sw3.packets_drop, pg3.packets_sent))
print("received: {}, dropped {}".format(sw3.packets_rec, sw3.packets_drop))
print("loss rate: {}".format(float(sw3.packets_drop)/sw3.packets_rec))
print("average system occupancy: {:.3f}".format(float(sum(pm3.sizes))/len(pm3.sizes)))

print("Last 10 queue sizes: {}".format(pm4.sizes[-10:]))
print("received: {}, dropped {}".format(sw4.packets_rec, sw4.packets_drop))
print("loss rate: {}".format(float(sw4.packets_drop)/sw4.packets_rec))
print("average system occupancy: {:.3f}".format(float(sum(pm4.sizes))/len(pm4.sizes)))

print("Last 10 sink arrival times: " + ", ".join(["{:.3f}".format(x) for x in ps.arrivals[-10:]]))
print("average wait = {:.3f}".format(sum(ps.waits)/len(ps.waits)))
