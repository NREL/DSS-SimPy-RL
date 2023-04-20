# This is the MAX MARGIN based linear function approximation using Linear Programming (2nd method adopted in NG (2000) paper)

import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
import random

import os
os.environ['PATH'] += r';C:\Users\asahu\.conda\envs\generic_gym_env\Library\mingw-w64\bin'
from cvxopt import matrix, solvers

def value(policy, n_states, transition_probabilities, reward, discount,
                    threshold=1e-2):
    """
    Find the value function associated with a policy.

    policy: List of action ints for each state.
    n_states: Number of states. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """
    v = np.zeros(n_states)

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            vs = v[s]
            a = policy[s]
            v[s] = sum(transition_probabilities[s, a, k] *
                       (reward[k] + discount * v[k])
                       for k in range(n_states))
            diff = max(diff, abs(vs - v[s]))

    return v

def optimal_value(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-2):
    """
    Find the optimal value function.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """

    v = np.zeros(n_states)

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            max_v = float("-inf")
            for a in range(n_actions):
                tp = transition_probabilities[s, a, :]
                max_v = max(max_v, np.dot(tp, reward + discount*v))

            new_diff = abs(v[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = max_v

    return v

def find_policy(n_states, n_actions, transition_probabilities, reward, discount,
                threshold=1e-2, v=None, stochastic=True):
    """
    Find the optimal policy.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    v: Value function (if known). Default None.
    stochastic: Whether the policy should be stochastic. Default True.
    -> Action probabilities for each state or action int for each state
        (depending on stochasticity).
    """

    if v is None:
        v = optimal_value(n_states, n_actions, transition_probabilities, reward,
                          discount, threshold)

    if stochastic:
        # Get Q using equation 9.2 from Ziebart's thesis.
        Q = np.zeros((n_states, n_actions))
        for i in range(n_states):
            for j in range(n_actions):
                p = transition_probabilities[i, j, :]
                Q[i, j] = p.dot(reward + discount*v)
        Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
        Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
        return Q

    def _policy(s):
        return max(range(n_actions),
                   key=lambda a: sum(transition_probabilities[s, a, k] *
                                     (reward[k] + discount * v[k])
                                     for k in range(n_states)))
    policy = np.array([_policy(s) for s in range(n_states)])
    return policy

def irl(n_states, n_actions, transition_probability, policy, discount, Rmax,
        l1):
    """
    Find a reward function with inverse RL as described in Ng & Russell, 2000.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    policy: Vector mapping state ints to action ints. Shape (N,).
    discount: Discount factor. float.
    Rmax: Maximum reward. float.
    l1: l1 regularisation. float.
    -> Reward vector
    """

    A = set(range(n_actions))  # Set of actions to help manage reordering
                               # actions.
    # The transition policy convention is different here to the rest of the code
    # for legacy reasons; here, we reorder axes to fix this. We expect the
    # new probabilities to be of the shape (A, N, N).
    transition_probability = np.transpose(transition_probability, (1, 0, 2))

    def T(a, s):
        """
        Shorthand for a dot product used a lot in the LP formulation.
        """

        return np.dot(transition_probability[policy[s], s] -
                      transition_probability[a, s],
                      np.linalg.inv(np.eye(n_states) -
                        discount*transition_probability[policy[s]]))

    # This entire function just computes the block matrices used for the LP
    # formulation of IRL.

    # Minimise c . x.
    c = -np.hstack([np.zeros(n_states), np.ones(n_states),
                    -l1*np.ones(n_states)])

    #print(' c shape ', c.shape)
    zero_stack1 = np.zeros((n_states*(n_actions-1), n_states))
    #print(' zero_stack1 shape ', zero_stack1.shape)
    T_stack = np.vstack([
        -T(a, s)
        for s in range(n_states)
        for a in A - {policy[s]}
    ])
    #print(' T_stack shape ', T_stack.shape)
    I_stack1 = np.vstack([
        np.eye(1, n_states, s)
        for s in range(n_states)
        for a in A - {policy[s]}
    ])
    #print(' I_stack1 shape ', I_stack1.shape)
    I_stack2 = np.eye(n_states)
    #print(' I_stack2 shape ', I_stack2.shape)
    zero_stack2 = np.zeros((n_states, n_states))
    #print(' zero_stack2 shape ', zero_stack2.shape)
    D_left = np.vstack([T_stack, T_stack, -I_stack2, I_stack2])
    D_middle = np.vstack([I_stack1, zero_stack1, zero_stack2, zero_stack2])
    D_right = np.vstack([zero_stack1, zero_stack1, -I_stack2, -I_stack2])

    D = np.hstack([D_left, D_middle, D_right])
    #print(' D shape ', D.shape)
    b = np.zeros((n_states*(n_actions-1)*2 + 2*n_states, 1))
    bounds = np.array([(None, None)]*2*n_states + [(-Rmax, Rmax)]*n_states)

    # We still need to bound R. To do this, we just add
    # -I R <= Rmax 1
    # I R <= Rmax 1
    # So to D we need to add -I and I, and to b we need to add Rmax 1 and Rmax 1
    D_bounds = np.hstack([
        np.vstack([
            -np.eye(n_states),
            np.eye(n_states)]),
        np.vstack([
            np.zeros((n_states, n_states)),
            np.zeros((n_states, n_states))]),
        np.vstack([
            np.zeros((n_states, n_states)),
            np.zeros((n_states, n_states))])])
    b_bounds = np.vstack([Rmax*np.ones((n_states, 1))]*2)
    D = np.vstack((D, D_bounds))
    b = np.vstack((b, b_bounds))
    A_ub = matrix(D)
    b = matrix(b)
    c = matrix(c)
    results = solvers.lp(c, A_ub, b)
    r = np.asarray(results["x"][:n_states], dtype=np.double)
    #print('Reward ',r)
    return r.reshape((n_states,))

def v_tensor(value, transition_probability, feature_dimension, n_states,
             n_actions, policy):
    """
    Finds the v tensor used in large linear IRL.

    value: NumPy matrix for the value function. The (i, j)th component
        represents the value of the jth state under the ith basis function.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    feature_dimension: Dimension of the feature matrix. int.
    n_states: Number of states sampled. int.
    n_actions: Number of actions. int.
    policy: NumPy array mapping state ints to action ints.
    -> v helper tensor.
    """

    v = np.zeros((n_states, n_actions-1, feature_dimension))
    for i in range(n_states):
        a1 = policy[i]
        exp_on_policy = np.dot(transition_probability[i, a1], value.T)
        seen_policy_action = False
        for j in range(n_actions):
            # Skip this if it's the on-policy action.
            if a1 == j:
                seen_policy_action = True
                continue

            exp_off_policy = np.dot(transition_probability[i, j], value.T)
            if seen_policy_action:
                v[i, j-1] = exp_on_policy - exp_off_policy
            else:
                v[i, j] = exp_on_policy - exp_off_policy
    return v

def large_irl(value, transition_probability, feature_matrix, n_states,
              n_actions, policy):
    """
    Find the reward in a large state space.

    value: NumPy matrix for the value function. The (i, j)th component
        represents the value of the jth state under the ith basis function.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state. 
    n_states: Number of states sampled. int.
    n_actions: Number of actions. int.
    policy: NumPy array mapping state ints to action ints.
    -> Reward for each state in states.
    """

    D = feature_matrix.shape[1]

    # First, calculate v, which is just a helper tensor.
    v = v_tensor(value, transition_probability, D, n_states, n_actions, policy)

    # Now we can calculate c, G, h, A, and b.

    # x = [z y_i^+ y_i^- a], which is a [N (K-1)*N (K-1)*N D] vector.
    x_size = n_states + (n_actions-1)*n_states*2 + D

    # c is a big stack of ones and zeros; there's N ones and the rest is zero.
    c = -np.hstack([np.ones(n_states), np.zeros(x_size - n_states)])
    assert c.shape[0] == x_size

    # A is [0 I_j -I_j -v^T_{ij}] and j NOT EQUAL TO policy(i).
    # I believe this is accounted for by the structure of v.
    A = np.hstack([
        np.zeros((n_states*(n_actions-1), n_states)),
        np.eye(n_states*(n_actions-1)),
        -np.eye(n_states*(n_actions-1)),
        np.vstack([v[i, j].T for i in range(n_states)
                             for j in range(n_actions-1)])])
    assert A.shape[1] == x_size

    # b is just zeros!
    b = np.zeros(A.shape[0])

    # Break G up into the bottom row and other rows to construct it.
    bottom_row = np.vstack([
                    np.hstack([
                        np.ones((n_actions-1, 1)).dot(np.eye(1, n_states, l)),
                        np.hstack([-np.eye(n_actions-1) if i == l
                                   else np.zeros((n_actions-1, n_actions-1))
                         for i in range(n_states)]),
                        np.hstack([2*np.eye(n_actions-1) if i == l
                                   else np.zeros((n_actions-1, n_actions-1))
                         for i in range(n_states)]),
                        np.zeros((n_actions-1, D))])
                    for l in range(n_states)])
    assert bottom_row.shape[1] == x_size
    G = np.vstack([
            np.hstack([
                np.zeros((D, n_states)),
                np.zeros((D, n_states*(n_actions-1))),
                np.zeros((D, n_states*(n_actions-1))),
                np.eye(D)]),
            np.hstack([
                np.zeros((D, n_states)),
                np.zeros((D, n_states*(n_actions-1))),
                np.zeros((D, n_states*(n_actions-1))),
                -np.eye(D)]),
            np.hstack([
                np.zeros((n_states*(n_actions-1), n_states)),
                -np.eye(n_states*(n_actions-1)),
                np.zeros((n_states*(n_actions-1), n_states*(n_actions-1))),
                np.zeros((n_states*(n_actions-1), D))]),
            np.hstack([
                np.zeros((n_states*(n_actions-1), n_states)),
                np.zeros((n_states*(n_actions-1), n_states*(n_actions-1))),
                -np.eye(n_states*(n_actions-1)),
                np.zeros((n_states*(n_actions-1), D))]),
            bottom_row])
    assert G.shape[1] == x_size

    h = np.vstack([np.ones((D*2, 1)),
                   np.zeros((n_states*(n_actions-1)*2+bottom_row.shape[0], 1))])

    from cvxopt import matrix, solvers
    c = matrix(c)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)
    results = solvers.lp(c, G, h, A, b)
    alpha = np.asarray(results["x"][-D:], dtype=np.double)
    return np.dot(feature_matrix, -alpha)



class Gridworld(object):
    """
    Gridworld MDP.
    """

    def __init__(self, grid_size, wind, discount):
        """
        grid_size: Grid size. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        -> Gridworld
        """

        self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        self.n_actions = len(self.actions)
        self.n_states = grid_size**2
        self.grid_size = grid_size
        self.wind = wind
        self.discount = discount

        # Preconstruct the transition probability array.
        self.transition_probability = np.array(
            [[[self._transition_probability(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])

    def __str__(self):
        return "Gridworld({}, {}, {})".format(self.grid_size, self.wind,
                                              self.discount)

    def feature_vector(self, i, feature_map="ident"):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> Feature vector.
        """

        if feature_map == "coord":
            f = np.zeros(self.grid_size)
            x, y = i % self.grid_size, i // self.grid_size
            f[x] += 1
            f[y] += 1
            return f
        if feature_map == "proxi":
            f = np.zeros(self.n_states)
            x, y = i % self.grid_size, i // self.grid_size
            for b in range(self.grid_size):
                for a in range(self.grid_size):
                    dist = abs(x - a) + abs(y - b)
                    f[self.point_to_int((a, b))] = dist
            return f
        # Assume identity map.
        f = np.zeros(self.n_states)
        f[i] = 1
        return f

    def feature_matrix(self, feature_map="ident"):
        """
        Get the feature matrix for this gridworld.

        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> NumPy array with shape (n_states, d_states).
        """

        features = []
        for n in range(self.n_states):
            f = self.feature_vector(n, feature_map)
            features.append(f)
        return np.array(features)

    def int_to_point(self, i):
        """
        Convert a state int into the corresponding coordinate.

        i: State int.
        -> (x, y) int tuple.
        """

        return (i % self.grid_size, i // self.grid_size)

    def point_to_int(self, p):
        """
        Convert a coordinate into the corresponding state int.

        p: (x, y) tuple.
        -> State int.
        """

        return p[0] + p[1]*self.grid_size

    def neighbouring(self, i, k):
        """
        Get whether two points neighbour each other. Also returns true if they
        are the same point.

        i: (x, y) int tuple.
        k: (x, y) int tuple.
        -> bool.
        """

        return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1

    def _transition_probability(self, i, j, k):
        """
        Get the probability of transitioning from state i to state k given
        action j.

        i: State int.
        j: Action int.
        k: State int.
        -> p(s_k | s_i, a_j)
        """

        xi, yi = self.int_to_point(i)
        xj, yj = self.actions[j]
        xk, yk = self.int_to_point(k)

        if not self.neighbouring((xi, yi), (xk, yk)):
            return 0.0

        # Is k the intended state to move to?
        if (xi + xj, yi + yj) == (xk, yk):
            return 1 - self.wind + self.wind/self.n_actions

        # If these are not the same point, then we can move there by wind.
        if (xi, yi) != (xk, yk):
            return self.wind/self.n_actions

        # If these are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (xi, yi) in {(0, 0), (self.grid_size-1, self.grid_size-1),
                        (0, self.grid_size-1), (self.grid_size-1, 0)}:
            # Corner.
            # Can move off the edge in two directions.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                return 1 - self.wind + 2*self.wind/self.n_actions
            else:
                # We can blow off the grid in either direction only by wind.
                return 2*self.wind/self.n_actions
        else:
            # Not a corner. Is it an edge?
            if (xi not in {0, self.grid_size-1} and
                yi not in {0, self.grid_size-1}):
                # Not an edge.
                return 0.0

            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1 - self.wind + self.wind/self.n_actions
            else:
                # We can blow off the grid only by wind.
                return self.wind/self.n_actions

    def reward(self, state_int):
        """
        Reward for being in state state_int.

        state_int: State integer. int.
        -> Reward.
        """

        if state_int == self.n_states - 1:
            return 1
        return 0

    def average_reward(self, n_trajectories, trajectory_length, policy):
        """
        Calculate the average total reward obtained by following a given policy
        over n_paths paths.

        policy: Map from state integers to action integers.
        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        -> Average reward, standard deviation.
        """

        trajectories = self.generate_trajectories(n_trajectories,
                                             trajectory_length, policy)
        rewards = [[r for _, _, r in trajectory] for trajectory in trajectories]
        rewards = np.array(rewards)

        # Add up all the rewards to find the total reward.
        total_reward = rewards.sum(axis=1)

        # Return the average reward and standard deviation.
        return total_reward.mean(), total_reward.std()

    def optimal_policy(self, state_int):
        """
        The optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy = self.int_to_point(state_int)

        if sx < self.grid_size and sy < self.grid_size:
            return rn.randint(0, 2)
        if sx < self.grid_size-1:
            return 0
        if sy < self.grid_size-1:
            return 1
        raise ValueError("Unexpected state.")

    def optimal_policy_deterministic(self, state_int):
        """
        Deterministic version of the optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy = self.int_to_point(state_int)
        if sx < sy:
            return 0
        return 1

    def generate_trajectories(self, n_trajectories, trajectory_length, policy,
                                    random_start=False):
        """
        Generate n_trajectories trajectories with length trajectory_length,
        following the given policy.

        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        policy: Map from state integers to action integers.
        random_start: Whether to start randomly (default False). bool.
        -> [[(state int, action int, reward float)]]
        """

        trajectories = []
        for _ in range(n_trajectories):
            if random_start:
                sx, sy = rn.randint(self.grid_size), rn.randint(self.grid_size)
            else:
                sx, sy = 0, 0

            trajectory = []
            for _ in range(trajectory_length):
                if rn.random() < self.wind:
                    action = self.actions[rn.randint(0, 4)]
                else:
                    # Follow the given policy.
                    action = self.actions[policy(self.point_to_int((sx, sy)))]

                if (0 <= sx + action[0] < self.grid_size and
                        0 <= sy + action[1] < self.grid_size):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                else:
                    next_sx = sx
                    next_sy = sy

                state_int = self.point_to_int((sx, sy))
                action_int = self.actions.index(action)
                next_state_int = self.point_to_int((next_sx, next_sy))
                reward = self.reward(next_state_int)
                trajectory.append((state_int, action_int, reward))

                sx = next_sx
                sy = next_sy

            trajectories.append(trajectory)

        return np.array(trajectories)

def large_network_test(grid_size, discount):
    """
    Run large state space linear programming inverse reinforcement learning on
    the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    """

    wind = 0.3
    trajectory_length = 3*grid_size

    gw = Gridworld(grid_size, wind, discount)

    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])

    # kind of the expert policy
    policy = [gw.optimal_policy_deterministic(s) for s in range(gw.n_states)]

    # Need a value function for each basis function.
    feature_matrix = gw.feature_matrix(feature_map='coord')

    print('coord features ',feature_matrix.shape)
    values = []
    for dim in range(feature_matrix.shape[1]):
        reward = feature_matrix[:, dim]
        values.append(value(policy, gw.n_states, gw.transition_probability,
                            reward, gw.discount))
    values = np.array(values)

    r = large_irl(values, gw.transition_probability,
                        feature_matrix, gw.n_states, gw.n_actions, policy)

    plt.subplot(1, 2, 1)
    plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()


def small_network_test(grid_size, discount):
    """
    Run linear programming inverse reinforcement learning on the gridworld MDP.
    Plots the reward function.
    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    """

    wind = 0.3
    trajectory_length = 3*grid_size

    gw = Gridworld(grid_size, wind, discount)

    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
    policy = [gw.optimal_policy_deterministic(s) for s in range(gw.n_states)]
    r = irl(gw.n_states, gw.n_actions, gw.transition_probability,
            policy, gw.discount, 1, 5)

    plt.subplot(1, 2, 1)
    plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()

if __name__ == '__main__':
    small_network_test(5,0.2)
    #large_network_test(10, 0.9)
