.. Adaptive Resilience Metric IRL documentation master file, created by
   sphinx-quickstart on Sun Aug 14 17:47:33 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ARM-IRL's documentation!
==========================================================

This project develops a communication discrete-event simulation environment for reinforcement learning using SimPy. Further the environment is extended for cyber physical simulation with integration of the OpenDSS environment, that provides a playground for cyber resilient distribution grid control. This co-simulation RL environment is light-weight and can assist in performing faster simulations and generating large-scale datasets. In the current work two Markov Decision Process (MDP) models are developed for re-routing and network reconfiguration based restoration in the communication and feeder network respectively.

This cyber-physical RL environment is further utilized to learn an **Adaptive Resilience Metric** using the concept of **Inverse Reinforcement Learning**.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   openDSSenvSB_DiscreteSpace
   generate_scenario
   CyberWithChannelEnvSB_123_Experimentation
   SimpyCyberEnvTest
   OpenDssEnvTest
   CPEnv_DiscreteDSS_RtrDropRate
   OpenDssEnvVisual
   CyberEnvVisualNewNW
   resilience_graphtheory
   generate_expert_demonstrations_feeder
   generate_expert_demonstrations_ieee123
   generate_expert_demonstrations_cps
   cyber_train_bc_ieee123
   phy_train_bc
   cyber_train_airl_ieee123
   phy_train_airl
   cps_train_airl
   phy_train_gail
   cps_train_gail
   phy_train_dagger
   visualize_reward_net_cyber
   visualize_reward_net_physical
   birl
   linear_func_approx
   bc
   dagger
   gail
   airl
   dqn
   per_dqn


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
