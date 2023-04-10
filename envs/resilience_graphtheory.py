
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(directory))
sys.path.insert(0,desktop_path+'\ARM_IRL')
import networkx as nx
import opendssdirect as dss
import re
from collections import defaultdict
from statistics import mean
#from generate_scenario import get_Vbus


# get the voltages by bus names
def get_Vbus(dss, circuit, busname):  # busname doesn't has .1, .2, or .3
    """
        Calculate the bus voltages of a bus in the distribution feeder

        :param dss: opendss network 
        :type dss: dss python object
        :param circuit: DSS circuit
        :type circuit: circuit object
        :param busname: bus name as defined in the dss case
        :type busname: str
        :return: bus voltage
        :rtype: float
    """
    circuit.SetActiveBus(busname)
    voltage = dss.Bus.VMagAngle()
    Vmag = [ii / dss.Bus.kVBase() / 1000 for ii in voltage[0:len(voltage):2]]
    # Vmag = [ii/1 for ii in voltage[0:len(voltage):2]]
    return Vmag

class GraphResilienceMetric:
    """This is the class for computing graph based resilience metric

    :param _dss: The opendss object
    :type _dss: opendssdirect python object
    """
    def __init__(self, _dss):
        self.dss = _dss

    def create_nxgraph(self, line_faults, switches_on):
        """Create the Networkx graph

        :param line_faults: List of line fault
        :type line_faults: list
        :param switches_on: List of sectionalizing switch that are operating
        :type switches_on: list
        :return: The networkx graph model along with the node position dictionary
        :rtype: graph, dictionary
        """
        N = len(self.dss.Circuit.AllBusNames())
        bus_names = self.dss.Circuit.AllBusNames()
        G = nx.Graph()
        pos = {}
        node_type = {}
        elements = self.dss.Circuit.AllElementNames()
        for i in range(N):
            busname = self.dss.Circuit.AllBusNames()[i]
            self.dss.Circuit.SetActiveBus(busname) 
            pos[busname] = (self.dss.Bus.X(), self.dss.Bus.Y())
            G.add_node(busname, pos = pos[busname])

        """ for i in range(len(elements)):
            busname = dss.Circuit.AllBusNames()[i]
            dss.Circuit.SetActiveBus(busname) 
            el = elements[i]
            if not re.search('^Line', el) and not re.search('^Transformer', el) and not re.search(
                        '^SwtControl', el) and not re.search('^RegControl', el) and not re.search('^Reactor',el):
                if 'Load' in el:
                    node_type[busname] = 'Load'
                elif 'Generator' in el:
                    node_type[busname] = 'Generator'
                elif 'Vsource' in el:
                    node_type[busname] = 'Source'
                else:
                    node_type[busname] = 'Node'
                
                pos[busname] = (dss.Bus.X(), dss.Bus.Y())
                G.add_node(busname, pos = pos[busname],node_type = node_type[busname]) """

        # work on the edges
        lines = self.dss.Lines.First()

        # dictionary that stores the from and to bus and the status
        edges = {}
        while lines:
            line = self.dss.Lines
            lname = line.Name()
            from_bus = line.Bus1().split('.')[0] if '.' in line.Bus1() else line.Bus1()
            to_bus = line.Bus2().split('.')[0] if '.' in line.Bus2() else line.Bus2()
            status = 0
            if 'sw' in lname:
                if lname in switches_on:
                    status = 1
            elif lname not in line_faults:
                status = 1
            edges[lname] = (from_bus,to_bus,status)
            lines = self.dss.Lines.Next()

        for i,(k,v) in enumerate(edges.items()):
            status = v[2]
            if status == 1:
                G.add_edge(v[0],v[1])

        return G,pos

        """ nx.draw_networkx(G, pos=pos,node_size=12, with_labels=True)
        # plot and save in file
        plt.savefig('dummy.png')
    """
        # add edge depending on  the line, transformer and switch status

    def compute_bc(self, line_faults, switches_on):
        """
        Calculate Betweenness Centrality (BC) of network graph nodes.This function calculates the BC for the network topology given.

        :param dss: opendss network 
        :type dss: dss python object
        :param line_faults: List of line fault
        :type line_faults: list
        :param switches_on: List of sectionalizing switch that are operating
        :type switches_on: list
        :return: numpy array with BC values for each node/bus.
        :rtype: numpy_array (double)
        """

        # get number of buses in the system
        #N = len(dss.Circuit.AllBusNames())
        #bcBuses = np.arange(N)
        bcBuses = self.dss.Circuit.AllBusNames()

        # create networkx network
        networkTopology,_ = self.create_nxgraph(line_faults, switches_on)
        # convert to digraph
        networkTopologyDigraph = nx.DiGraph(networkTopology)
        # calculate closeness centrality
        bc_dict = nx.betweenness_centrality(networkTopologyDigraph)

        # vectorize results (from dict to numpy array)
        bcBuses = np.vectorize(bc_dict.get)(bcBuses)

        return bcBuses



    def compute_cl(self, line_faults, switches_on):
        """
        Calculate Closeness Centrality (CL) of network graph nodes.This function calculates the CL for the network topology given.

        :param dss: opendss network 
        :type dss: dss python object
        :param line_faults: List of line fault
        :type line_faults: list
        :param switches_on: List of sectionalizing switch that are operating
        :type switches_on: list
        :return: numpy array with CL values for each node/bus.
        :rtype: numpy_array (double)
        """

        # get number of buses in the system
        clBuses = self.dss.Circuit.AllBusNames()

        # create networkx network
        networkTopology,_ = self.create_nxgraph(line_faults, switches_on)
        # calculate closeness centrality
        cl_dict = nx.closeness_centrality(networkTopology)
        # vectorize results (from dict to numpy array)
        clBuses = np.vectorize(cl_dict.get)(clBuses)

        return clBuses




    def compute_ebc(self, line_faults, switches_on):
        """
        Calculate Edge Betweenness Centrality (EBC) of network graph. This function calculates the EBC for the network topology  given.

        :param dss: opendss network 
        :type dss: dss python object
        :param line_faults: List of line fault
        :type line_faults: list
        :param switches_on: List of sectionalizing switch that are operating
        :type switches_on: list
        :return: numpy array with BC values for each node/bus. The max. EBC value for an edge connected to the respective node/bus.
        :rtype: numpy_array (double)
        """

        # get number of buses in the system
        ebcList = self.dss.Circuit.AllBusNames()
        ebcBuses = {}
        for i in ebcList:
            ebcBuses[i] = 0.0

        # create networkx network
        networkTopology,_ = self.create_nxgraph(line_faults, switches_on)
        # calculate edge betweenness centrality
        ebc_dict = nx.edge_betweenness_centrality(networkTopology)

        # Go through sorted dataframe and assign the max PI value to corresponding connected buses
        for key in ebc_dict:
            # print(key, '->', ebc_dict[key])
            # Go through dict and assign the max ebc value to corresponding connected buses
            try:
                if(ebc_dict[key] > ebcBuses[key[0]]):
                    ebcBuses[key[0]] = ebc_dict[key]
                if(ebc_dict[key] > ebcBuses[key[1]]):
                    ebcBuses[key[1]] = ebc_dict[key]
            except:
                continue
        
        BusesEbc = np.vectorize(ebcBuses.get)(ebcList)
        return BusesEbc

    ################################# Venkatesh addition for graphical topological resilience metrics #####################################################     
    def graph_topology_metrics(self, line_faults, switches_on):
        # create networkx network
        networkTopology,_ = self.create_nxgraph(line_faults, switches_on)
        # Average clustering for all nodes 
        avg_clustering = nx.average_clustering(networkTopology) 
        try:
            # Graph diameter 
            graph_diameter =nx.diameter(networkTopology)
        except:
            graph_diameter = 0

        try:
            # Characteristic path length/Avg shortest path length 
            characteristic_path_length=nx.average_shortest_path_length(networkTopology)
        except:
            characteristic_path_length=0
        # Algebraic connectivity 
        algebraic_connectivity=nx.algebraic_connectivity(networkTopology)

        #storing in a single list to return 
        topology_metrics = {'avg_clustering': avg_clustering, 'graph_dia':graph_diameter, 'path_length':characteristic_path_length, 'alg_conn':algebraic_connectivity}

        return topology_metrics



    def draw_network(self, line_faults, switches_on, fileName='dummy.png'):
        """
        Draws the network graph in a png file. This function draws the network topology  of the network given.

        :param dss: opendss network 
        :type dss: dss python object
        :param line_faults: List of line fault
        :type line_faults: list
        :param switches_on: List of sectionalizing switch that are operating
        :type switches_on: list
        :param fileName: Name of the file where the graph is going to be drawn. 
        :type fileName: str
        :return: Nothing
        :rtype: None
        """

        # create networkx network
        networkTopology,pos = self.create_nxgraph(line_faults, switches_on)

        puVolt ={}
        Bus_name_vec = self.dss.Circuit.AllBusNames()
        v_val = []
        v_key = []
        for bus_name in enumerate(Bus_name_vec):
            puVolt[bus_name[1]] = get_Vbus(self.dss, self.dss.Circuit, bus_name[1])
            v_val.append(np.mean(puVolt[bus_name[1]]))
            v_key.append(bus_name[1])

        nx.set_node_attributes(networkTopology, values = 1.0, name="voltage")

        for i,node in enumerate(list(networkTopology.nodes(data = True))):
            networkTopology.nodes[node[0]]["voltage"] = mean(puVolt[node[0]])
        plt.clf()
        nx.draw_networkx_edges(networkTopology, pos)
        mcp = nx.draw_networkx_nodes(networkTopology, pos, node_color=v_val, vmin=0.9, vmax=max(v_val), cmap='Blues')
        #mcp = nx.draw_networkx_nodes(networkTopology, pos, cmap='Blues')
        nx.draw_networkx_labels(networkTopology, pos, {n: n for n in v_key}, font_size=10)

        #plt.clf()
        plt.colorbar(mcp)
        # plt.savefig(fig, dpi=150)
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
        pass

        """ # draw the network
        nx.draw_networkx(networkTopology, arrows=True, with_labels=True)
        # plot and save in file
        plt.savefig(fileName) """

    
if __name__ == "__main__":
    dss_data_dir = desktop_path+'\\AdaptiveResilienceMetric\\123Bus_Simple\\'
    dss_master_file_dir = 'Redirect ' + dss_data_dir + 'IEEE123Master.dss'

    dss.run_command(dss_master_file_dir)
    path_coord = desktop_path+'\\AdaptiveResilienceMetric\\123Bus_Simple\\BusCoords.dat'
    bus_coord = r"Buscoords "+path_coord
    dss.run_command(bus_coord)
    circuit = dss.Circuit

    line_faults =['l111','l117']
    switches_on=['sw4','sw5']

    grm = GraphResilienceMetric(_dss = dss)
    bcs = grm.compute_bc(line_faults,switches_on)
    cls = grm.compute_cl(line_faults,switches_on)
    ebcs = grm.compute_ebc(line_faults,switches_on)
    print('Computed topological metrics')
    #create_nxgraph(dss,line_faults,switches_on)
    grm.draw_network(line_faults,switches_on)
    

