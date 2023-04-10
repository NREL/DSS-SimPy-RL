# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 9:00:05 2022

@author: abhijeetsahu

A sample code to generate cisco based static routing policy (using shortest path) for a mesh network with 16 LANs
"""

import networkx as nx
import numpy as np

# edges network ids
edge_net_id ={}
node_net_id={}

A = np.array([[1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
             [0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
             [0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0],
             [0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0],
             [0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0],
             [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1],
             [0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0],
             [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0],
             [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
             [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])

A = np.array([[1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
             [1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0],
             [1,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0],
             [1,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0],
             [0,0,1,1,0,1,0,0,0,0,0,1,0,0,0,1],
             [0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0],
             [0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0],
             [0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0],
             [0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0],
             [0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,0],
             [0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0],
             [0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0],
             [0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0],
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])

counter=0
ncounter=0
for ix, w in enumerate(A):
    for iy, v in enumerate(w):
        #if ix is not iy and v==1:
        if iy > ix and v == 1:
            edge_net_id[counter] = '10.0.'+str(counter+1)+'.0'+'#'+str(ix)+','+str(iy)
            counter += 1
    node_net_id[ncounter]='192.168.'+str(ncounter+1)+'.0'
    ncounter += 1
print(edge_net_id)
print(node_net_id)

edge_pair= {}
for id in edge_net_id:
    val = edge_net_id[id]
    edge_pair[str(val).split('#')[1]] = str(val).split('#')[0] # forward link
    # exchange src, dest to dest , src
    src_dest = str(val).split('#')[1]
    x = src_dest.split(',')
    x.reverse()
    dest_src = ','.join(x)
    edge_pair[dest_src] = str(val).split('#')[0] # backward link

print(edge_pair)
src=0
G = nx.from_numpy_matrix(A)

for src in range(0,16):
    paths = nx.single_source_shortest_path(G, src)
    f = open("Router"+str(src)+".txt", "w+")
    print(paths)
    route_list=[]
    for key in paths:
        #print(str(key) +' and '+str(paths[key]))
        try:
            if len(paths[key]) > 1:
                match_string = str(src)+','+str(paths[key][1])
                #print(match_string)
                if match_string in edge_pair.keys():
                    route_list.append('route add '+str(node_net_id[key])+ '/24 gw '+str(edge_pair[match_string]))
        except KeyError:
            print(node_net_id)
            print(key)
    print('route list for Node'+str(src))
    print(route_list)
    for routes in route_list:
        f.write(routes+'\n')