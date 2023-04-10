import heapq
import numpy as np
from itertools import count
from collections import deque
import random
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras import optimizers
from keras.callbacks import TensorBoard

tiebreaker = count()

class Replay_Memory:
    """ Standard replay memory sampled uniformly """
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, transition):
        self.memory.append(transition)

    def batch(self, n):
        return random.sample(self.memory, n)

    def size(self):
        return len(self.memory)
    
    def __len__(self):
        return self.size()

    def is_full(self):
        return True if self.size() >= self.max_size else False
    
class PER:
    """ Prioritized replay memory using binary heap """
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = []

    def add(self, transition, TDerror):
        heapq.heappush(self.memory, (-TDerror, next(tiebreaker), transition))
        if self.size() > self.max_size:
            self.memory = self.memory[:-1]
        heapq.heapify(self.memory)
    
    def batch(self, n):
        batch = heapq.nsmallest(n, self.memory)
        batch = [e for (_, _, e) in batch]
        self.memory = self.memory[n:]
        return batch

    def size(self):
        return len(self.memory)
    
    def __len__(self):
        return self.size()

    def is_full(self):
        return True if self.size() >= self.max_size else False


class Agent:
    def __init__(self, state_size, action_size, per=True):
        self.state_size = state_size
        self.action_size = action_size
#         self.memory = deque(maxlen=300)
        self.per = per
        if per:
            self.memory = PER(1000)
        else:
            self.memory = Replay_Memory(300)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9992
        self.learning_rate = 0.001
        self.tau = .50 #0.125
        self.model = self._create_model()
        self.target_model = self._create_model()
        self.epsilon_decay_counter = 1000

    def _create_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(300, input_dim=self.state_size, activation='relu'))
        model.add(Dense(150, activation='relu'))
#         model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
#         model.add(Dense(self.action_size))
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss='mse',
                      optimizer=adam)
        return model

    def remember(self, state, action, reward, next_state, done):
        if self.per:
            td_error = self.compute_TDerror((state, action, reward, next_state, done))
            self.memory.add((state, action, reward, next_state, done), td_error)
        else:
            self.memory.add((state, action, reward, next_state, done))
        
    def _randmax(self, collection):
        return np.random.choice(np.argwhere(collection == np.amax(collection)).flatten(), 1)[0]

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.array([state])
        q_action_pairs = self.model.predict(state)[0]
        return self._randmax(q_action_pairs)
    
    def compute_TDerror(self, transition):
        S, A, R, S_p, terminal = transition
        S = np.array([S])
        S_p = np.array([S_p])
        y = self.model.predict(S)[0]
        y_old = np.array(y)
        A_p = self._randmax(self.model.predict(S_p)[0])
        y_target = self.target_model.predict(S_p)[0]
        y[A] = R if terminal else R + self.gamma * y_target[A_p]
        TDerror = np.abs(y_old[A]-y[A])
        return TDerror  

    def replay(self, batch_size):
        minibatch = self.memory.batch(32)
        current_states = np.squeeze(np.array([transition[0] for transition in minibatch]))
        next_states = np.squeeze(np.array([transition[3] for transition in minibatch]))
        y = self.model.predict(current_states)
        q = self.target_model.predict(next_states)

        for i, (_, action, reward, next_state, done) in enumerate(minibatch):
            new_q = reward
            if not done:
                #print(next_state.shape)
                next_state = np.expand_dims(next_state, axis=0)
                j = self.model.predict(next_state)
                max_target_a = self._randmax(j[0])
                new_q += self.gamma * q[i][max_target_a]
            y[i][action] = new_q
        loss = self.model.train_on_batch(current_states, y)
        
        if self.per:
            for item in minibatch:
                self.memory.add(item, self.compute_TDerror(item))
            
    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)