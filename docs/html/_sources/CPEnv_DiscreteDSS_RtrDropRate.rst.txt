SimPyDSS RL Env Test
======================================

.. image:: Figures/Episode_Steps.PNG
   :width: 400
   :alt: Alternate text

Steps in an episode created in the cyber-physical RL Environment. The interconnection in blue indicates the communication between the Simpy and Open DSS simulators. 
**1** indicates the passage of physical-side information to cyber network for determining the packet size. 
**2** indicates the passage of cyber and physical contingency to each others environment. Currently a physical fault adds an event in cyber emulator to generate a fault information to send to the aggregator. 
**3** indicates the merge of the cyber and physical state information to feed to the RL algorithm or the Agent. 
**4** Based on the policy, implement the action by segregating respective action of routing policy and control of sectionalizing switch. 
**5** Evaluating the goal $G_P$ and $G_C$ for terminating the episode when both goals are reached. 

Classes and Functions
---------------------

.. automodule:: CPEnv_DiscreteDSS_RtrDropRate
   :members:
   :undoc-members:
   :show-inheritance:
