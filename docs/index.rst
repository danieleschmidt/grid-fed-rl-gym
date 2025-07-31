Grid-Fed-RL-Gym Documentation
=============================

Welcome to Grid-Fed-RL-Gym, a comprehensive framework for training and deploying 
reinforcement learning agents on power distribution networks with federated learning capabilities.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   architecture
   api_reference
   examples
   contributing

Overview
--------

Grid-Fed-RL-Gym provides:

* **Digital Twin Simulation**: High-fidelity power flow modeling with real-time dynamics
* **Federated Learning**: Privacy-preserving distributed training across utilities  
* **Offline RL**: Learn from historical data without online exploration
* **Safety Constraints**: Hard constraints on voltage, frequency, and equipment limits
* **Multi-Agent Support**: Coordinate multiple grid controllers and DERs
* **Industry Standards**: IEEE test feeders and CIM-compliant data models

Quick Start
-----------

.. code-block:: bash

   pip install grid-fed-rl-gym
   
.. code-block:: python

   from grid_fed_rl import GridEnvironment
   from grid_fed_rl.feeders import IEEE13Bus
   
   # Create grid environment
   env = GridEnvironment(feeder=IEEE13Bus())
   
   # Run simulation
   obs = env.reset()
   action = env.action_space.sample()
   next_obs, reward, done, info = env.step(action)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`