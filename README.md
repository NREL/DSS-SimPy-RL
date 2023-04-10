# DSS-SimPy-RL
Open-DSS and SimPy based Cyber-Physical RL environment

Use the `environment.yaml` file to create the conda environment for installing the python packages required in this repository.
Commad: `conda env create --file environment.yaml`

**Description of the files within `agents` folder**
- `dqn.py` : Given the Reward model, training the RL agent using Deep Q Learning.
- `per_dqn.py` : Implementation of the Prioritized Experience Replay based-Deep Q Learning

**Description of the files within `cases` folder**
- `123Bus_DER` : This folder consists of the Open DSS files for the 123-bus feeder with DER integrated.
- `123Bus_Simple` : The folder consists of the Open DSS 123-bus files for the 123-bus feeder base model.
- `123Bus_SimpleMod` : Some minor modification from the Simple model.

**Description of the files within `powergym` folder**
- It contains the OpenDSS based RL environment for volt var control. Adopted from [PowerGym](https://github.com/siemens/powergym)

**Description of the files within `docs` folder**
- `Figures` : Contains all the figures used in user documentation.
- `html` : Contains all the html file for the user documentation.
- All the .rst files are the files generated using the sphinx python package.

**Description of the files within `envs` folder**
- `simpy_env` : All the SimPy-based communication RL environments.
- `simpy_dss` : All the fused Cyber-Physical RL environments for network reconfiguration and also volt var control.
- `openDSSenv.py`: This environment is created for episodic interaction of the OpenDSS simulator for training RL agent, similar to an Open AI Gym environment. Implements the N-1 and N-2 contingencies.
- `openDSSenvSB_DiscreteSpace.py`: Have a discrete space model for the OpenDSS environment.
- `generate_scenario.py`: This generates unique scenarios for the distribution feeder contingencies.
- `resilience_graphtheory.py`: Contains the topological resilience functions.

**Description of the files within `Figures` folder**
- Contains all the figure used in the environment and ARM-IRL paper.

**Description of the files `gurobi` folder**
- MATLAB implementation of network reconfiguration
- Examples of OPF using gurobi solver in python (with QC relaxation).

**Description of the files `mininet` folder**
- `generate_mininet_ieee123_network.py`: Linux router based implementation of the communication network of IEEE-123 bus feeder using static routes.
- `mininet_backend_HC_new_network.py`: An open vswitch based router implementation of the communication network of IEEE-123 bus feeder case.
- `generate_network.py`: Creates the adjacency matrix of the communication network


**Description of the files `pyomo_test` folder**
- folder consist some simple examples for running ACOPF using Pyomo

**Description of the files `rewards` folder**
- It containing some trained discriminator network.

**Description of the files `tests` folder**
- This includes test of forward RL agents (from stable-baselines and some developed from scratch: refer to `agents` folder) for the proposed MDP models incorporated in the RL environment.

**Description of the files `utils` folder**
- Contains codes for plotting the graphs for the purpose of research papers.

**Description of the files `validation` folder**
- Contains codes for the RL environment validation work.

- **visuals**: This folder is a visualization app for visualizing the simulation and observing the resilience metric in real-time as the episodes and their steps are executed. The `OpenDssEnvVisual.py` is the application implementation for the Open DSS physical RL environment.

![Sample Visualization for the open-DSS RL environment](https://github.com/Abhijeet1990/Dss_SimPy_RL/blob/main/visualization.PNG?raw=true)

`CyberEnvVisual.py` is the application implementation for the Simpy based Cyber RL environment. The IEEE 123 bus case is divided into 7 zones. Within each zones there is a `data concentrator` shown in red node, the green node is the `DSO`, while all the blue nodes are the `Routers` represent the hybrid topology. 

![Sample Visualization for the Simpy-based RL environment](https://github.com/Abhijeet1990/Dss_SimPy_RL/blob/main/cyber_visualization_new.PNG?raw=true)

`CyberPhysicalEnvVisual.py` is the application implementation for the Simpy based and OpenDSS based CyberPhysical combined RL environment.

[User Documentation](https://abhijeet1990.github.io/Dss_SimPy_RL/html/index.html) for the Dss_SimPy_RL alongwith the utilization of this RL environment for Adaptive resilience Metric learning using Inverse Reinforcement Learning (ARM_IRL).












