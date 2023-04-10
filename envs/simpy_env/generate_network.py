# this code will convert the simpy network to networkx and vice-verse

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def create_network():
    """code for creating a complex graph, here we assume there are 7 different zones and the main control center. so take there are 8 LANs
       Lets say there are 16 Routers, 8 are directly connected to the LANs and 8 are for the backbone network
       Lets say the adjacency matrix
    """

    A = np.array([[0,1,1,0,1,0,1,1,1,0,0,0,0,0,0,0],
                [1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0],
                [0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                [1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0],
                [0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
                [1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]]) 

    G = nx.from_numpy_matrix(A)
    nx.set_node_attributes(G, 'router', 'nodetype')

    # add all the pgen and psink nodes and their respective edges into the G
    sources = 7
    for i in range(sources):
        G.add_node('PG'+str(i+9), nodetype='sender')
        G.add_edge('PG'+str(i+9), i+9, weight=0.0) # this is 9 because 9 + 7 = 16
    G.add_node('PS',nodetype='sink')
    G.add_edge('PS',0, weight=0.0)
    return G


def create_network2():
    A = np.array([[0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0],
                [0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0],
                [0,0,1,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0],
                [0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1],
                [0,0,0,0,0,1,0,0,1,0,0,1,0,1,1,0,0,0],
                [0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,1],
                [0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,1,0],
                [0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,1,0,0],
                [0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0]]) 
    G = nx.from_numpy_matrix(A)
    nx.set_node_attributes(G, 'router', 'nodetype')

    # add all the pgen and psink nodes and their respective edges into the G
    sources = 7
    for i in range(sources):
        G.add_node('PG'+str(i+11), nodetype='sender')
        G.add_edge('PG'+str(i+11), i+11, weight=0.0) # this is 11 because 11 + 7 = 18
    G.add_node('PS',nodetype='sink')
    G.add_edge('PS',0, weight=0.0)
    return G

def draw_cyber_network(G):
    """This function draw the cyber network

    """
    node_color = []
    for node in G.nodes(data=True):
        if node[1]['nodetype'] == 'router':
            node_color.append('blue')
        elif node[1]['nodetype'] == 'sender':
            node_color.append('red')
        elif node[1]['nodetype'] == 'sink':
            node_color.append('green')
    nx.draw(G, with_labels=True, node_color = node_color, pos=nx.kamada_kawai_layout(G))    
    plt.savefig('cyber_new_network.png')
    

