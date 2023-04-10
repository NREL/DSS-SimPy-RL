DAgger Imitation Learning
=============
 Due to the i.i.d. assumption in the behavior cloning, if a classifier makes a mistake under the distribution of states faced by the demonstrator, then results following it faces compounded errors. DAgger proposes a new meta-algorithm which learns a stationary deterministic policy guaranteed to perform efficiently with the induced distribution of states. starts by extracting dataset at each iteration under the current policy and trains the next policy under the aggregate of all the collected datasets. The intuition behind this algorithm is that over the iterations, it is building up the set of inputs that the learned policy is likely to encounter during its execution based on previous experience (training iterations).

Classes and Functions
---------------------

.. automodule:: dagger
   :members:
   :undoc-members:
   :show-inheritance:
