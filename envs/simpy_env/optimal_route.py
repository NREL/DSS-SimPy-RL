# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 9:31:05 2022

@author: abhijeetsahu

These includes codes for the optimal routing from the expert to be considered in training IRL agents
"""

from CyberEnv import CyberEnv
import random
import networkx as nx
import numpy as np

env = CyberEnv()

nw = env.G

# how to allocate edge weigths to the problem

# currently simpy provides the node attributes of packet losss rates and all

for node in nw.nodes:
    paths = nx.single_source_shortest_path(nw, node)
    for k,v in paths.items():
        if k == 'PS':
            print(v)
    #print(paths)