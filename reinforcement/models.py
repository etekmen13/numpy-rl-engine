from neuralnet.models import MLP
from neuralnet.layers import Layer
from neuralnet.activations import ReLU
from reinforcement.replay import ReplayBuffer
import numpy as np
from copy import deepcopy


class DQN:
    def __init__(self, state_dim, action_dim, hidden_units, loss, optimizer):
        layers = [Layer(state_dim, hidden_units[0], activation=ReLU())]

        for i in range(len(hidden_units) - 1):
            layers.append(
                Layer(hidden_units[i], hidden_units[i + 1], activation=ReLU())
            )

        layers.append(Layer(hidden_units[-1], action_dim))

        self.q = MLP(*layers, loss=loss)
        self.target_q = deepcopy(self.q)
        self.loss = loss
        self.replay_buffer = ReplayBuffer(10000)
        self.optimizer = optimizer
        self.optimizer(self.q)
        self.optimizer.zero_grad()

    def get_params(self):
        return self.q.params

    def update_single(self, state, action, reward, next_state, done):
        q_current = self.q.forward(state)
        q_next = self.target_q.forward(next_state)
        q_target = q_current.copy()
        gamma = 0.99
        q_target[action] = reward + (1 - done) * gamma * np.max(q_next)
        loss = self.loss(q_current, q_target)
        grad = self.loss.gradient(q_current, q_target)
        self.q.backward(grad)
        return loss

    def get_action(self, state, eps):
        if np.random.rand() < eps:
            return np.random.randint(self.q.layers[-1].weights.get().shape[1])
        else:
            return self.get_action_greedy(state)

    def get_action_greedy(self, state):
        q_values = self.q.forward(state)
        action = np.argmax(q_values)
        return action.astype(np.int32)

    def update_batch(self, batch_size):
        self.optimizer.zero_grad()
        batch = self.replay_buffer.sample(batch_size)
        results = []
        for batch_item in batch:
            results.append(self.update_single(*batch_item))
        self.optimizer.step()

        for k in range(len(self.q.params)):
            self.target_q.params[k].data = (
                0.01 * self.q.params[k].data + (1 - 0.01) * self.target_q.params[k].data
            )

        return np.mean(results)

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))

    def clear_replay_buffer(self):
        self.replay_buffer.clear()

    def __len__(self):
        return len(self.replay_buffer)

    def full(self):
        return len(self.replay_buffer) == self.replay_buffer.capacity
