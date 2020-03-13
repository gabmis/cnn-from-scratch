from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    def __init__(self):
        self.input_shape = None
        self.weights = None
        self.activation = None
        self.optimizer = None
        self.input_memory = None
        pass

    @abstractmethod
    def initialize_weights(self):
        pass

    @abstractmethod
    def forward(self, inpt):
        pass

    @abstractmethod
    def backward(self, deltas):
        pass

    @abstractmethod
    def compute_grads(self, deltas):
        pass


class Activation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, inpt):
        pass

    @abstractmethod
    def derivative(self, inpt):
        pass


class Loss(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, inpt, targets):
        pass

    @abstractmethod
    def gradient(self, inpt, targets):
        pass


class Optimizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def step(self, weights, grads):
        pass


class Batcher:
    def __init__(self, data, targets, batch_size):
        self.data = data
        self.n_samples = data.shape[0]
        self.elapsed_samples = 0
        self.targets = targets
        self.batch_size = batch_size

    def next(self):
        done = False
        if self.elapsed_samples > self.n_samples:
            done = True
            self.elapsed_samples = 0
        batch_index = np.random.choice(self.n_samples, self.batch_size)
        self.elapsed_samples += self.batch_size
        return done, self.data[batch_index], self.targets[batch_index]
