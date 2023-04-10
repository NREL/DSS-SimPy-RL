OpenDSS RL Env
==================================

Open-DSS is an open source electric power distribution system simulator developed for grid modernization with the integration of DERs. This work focuses on developing an RL environment leveraging the *OpenDSSDirect.py* to interface with the OpenDSS modeled distribution feeder, for executing the contingencies and restoring them using network re-configuration using sectionalizing switch, an automated switching device that is intended to isolate faults and restore loads. The optimal network-reconfiguration is modeled as a MDP, where the variability is introduced at the beginning of each episode through random selection of different load profile along with a contingency.

Classes and Functions
---------------------

.. automodule:: openDSSenvSB_DiscreteSpace
   :members:
   :undoc-members:
   :show-inheritance:
