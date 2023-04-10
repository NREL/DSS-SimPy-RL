SimPy Cyber RL Env
==================================================

Components
----------

Forwarder Device 
----------------
A router is a networking device used for forwarding data packets between two networks. It functions in the network layer of the OSI layer. Conventionally, the router software has two functional processing units: a) Control Plane b) Forwarding Plane. In the control plane, the router maintains a routing table that maintains which path would be used to forward a packet and along which interface. This routing table is statically configured or updated dynamically using dynamic routing protocols such as OSPF, EIGRP, RIP, etc. The time taken to forward packets depends on the processing time of parsing the packet, and searching the next hop information from the routing table. Hence, the service rate, as well as, the queue limit for every router is modeled. The queue limit indicate the amount of byte the router can forward at a given instance. In the forwarding plane, the router simply forwards the packet to the desired interface based on the routing rule. In the MDP model, modelling the router drop rate as state, would play a crucial role in enforcing new routes. The router packet drop rate is defined as ratio of the number of dropped packets and the received packets.

Channel 
-------
The channel models the latency and bandwidth between each node in the network model. Based on the traffic, the channels update it utilization rate and computes the available channel capacity with the update frequency in the Simpy DES. A packet injected into the channel is dropped if the total bytes in channel crosses the available channel bandwith. For the evaluation of the environment, the experiments are performed under varying channel bandwidths.

Data Concentrator (DC)
-----------------
It is acting as data collector and forwarder for all the smart meters in the zones. When a fault occurs in the distribution feeder, the relay captures it and forward it to the Data Concentrator (DC). The DC within the zone forwards the state to the Distribution System Operator (DSO) which acts as the Data Aggregator. In the simulation model, it is assumed all the sensor data are accumulated at the DC, hence the payload size of the packet sent to the DA depends on the number of components i.e. transmission lines, buses, switches, etc within a zone. Conventionally, the DC communicates with sensors using power line carriers, while they communicate to DSO through the Wide Area Network (WAN). 

Distribution System Operator (DSO) / Data Aggregator (DA)
----------------------------------------------
Packets are received from the DC from different zones in the system. This node acts as the DSO collecting data from each DC. In the MDP model, the goal state to be the state when the data aggregator accept atleast **Ng** traffic within a timeframe from the DCs of each zone. 


Classes and Functions
---------------------

.. automodule:: CyberWithChannelEnvSB_123_Experimentation
   :members:
   :undoc-members:
   :show-inheritance:
