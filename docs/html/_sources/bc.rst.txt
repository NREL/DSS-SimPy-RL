Behavioral Cloning
=========
In Behavioral Cloning agent receives as training data both the encountered states and actions of the demonstrator, and then uses a classifier or regressor to replicate the expert’s policy. The steps involved in BC, involves: a) Collect demonstrations from expert. b) Assuming the expert trajectories as an i.i.d state-action pairs, learn a policy using supervised learning by minimizing the loss function. 

Classes and Functions
---------------------

.. automodule:: bc
   :members:
   :undoc-members:
   :show-inheritance:
