import logging

import numpy as np
from tqdm import trange

from base import Activation, Loss, Optimizer, Layer, Batcher
from conv import convolution, rot180, grad_convolution, convolve2d, deltas_convolution

logger = logging.getLogger(__name__)


class Identity(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, inpt):
        return inpt

    def derivative(self, inpt):
        return 1


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, inpt):
        return 1 / (1 + np.exp(-inpt))

    def derivative(self, inpt):
        return self.forward(inpt) * self.forward(1 - inpt)


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, inpt):
        return np.maximum(0, inpt)

    def derivative(self, inpt):
        return (inpt > 0).astype(float)


class Softmax(Activation):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, inpt):
        pass

    def derivative(self, inpt):
        pass


class MSE(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, inpt, targets):
        return np.mean((inpt - targets) ** 2, axis=1)

    def gradient(self, inpt, targets):
        return 2 / inpt.shape[1] * (inpt - targets)


class SGD(Optimizer):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr

    def step(self, weights, grads):
        return weights - self.lr * grads.mean(axis=0)


class FC(Layer):
    def __init__(
        self, input_size, batch_size, n_neurons, activation: Activation = Sigmoid()
    ):
        super().__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.n_neurons = n_neurons
        self.activation = activation
        self.initialize_weights()

    def initialize_weights(self):
        # TODO: use standard initialization
        self.weights = np.random.rand(self.n_neurons, self.input_size + 1)

    def forward(self, inpt):
        self.input_memory = np.column_stack([inpt, np.ones(self.batch_size)])
        self.weighted_input_memory = np.einsum(
            "ij,bj->bi", self.weights, self.input_memory
        )
        return self.activation.forward(self.weighted_input_memory)

    def backward(self, deltas):
        grads, deltas = self.compute_grads(deltas)
        self.weights = self.optimizer.step(self.weights, grads)
        return deltas

    def compute_grads(self, deltas, return_deltas=True):
        grads = np.einsum("bj,bi->bji", deltas, self.input_memory)
        # backprop recursion
        deltas = np.einsum("ij,bj->bi", self.weights.T, deltas)
        if return_deltas:
            return grads, deltas
        return grads


class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inpt):
        self.input_shape = inpt.shape[1:]
        return inpt.reshape(inpt.shape[0], -1)

    def backward(self, deltas):
        return deltas.reshape((-1, *self.input_shape))


class Conv2D(Layer):
    def __init__(
        self,
        input_shape,
        kernel_size,
        n_filters,
        full=True,
        activation: Activation = ReLU(),
    ):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.full = full
        self.activation = activation
        self.initialize_weights()

    def initialize_weights(self):
        # TODO: use standard initialization
        self.weights = np.random.rand(
            self.n_filters, self.kernel_size, self.kernel_size, self.input_shape[-1]
        )

    def forward(self, inpt):
        self.input_memory = inpt
        self.weighted_input_memory = convolution(inpt, self.weights, self.full)
        return self.activation.forward(self.weighted_input_memory)

    def backward(self, deltas):
        grads, deltas = self.compute_grads(deltas)
        self.optimizer.step(self.weights, grads)
        return deltas

    def compute_grads(self, deltas, return_deltas=True):
        grads = grad_convolution(deltas, self.input_memory, self.weights, self.full)
        deltas = deltas_convolution(deltas, self.weights)
        if return_deltas:
            return grads, deltas
        return grads


class CNN:
    def __init__(self, layers, loss: Loss, optimizer: Optimizer, batcher: Batcher):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.batcher = batcher
        self.check_shape_compatibility()

    def check_shape_compatibility(self):
        try:
            self.forward(self.batcher.next())
        except Exception:
            logger.exception("Exception in forward pass, check your shapes.")
            raise

    def train(self, n_epochs):
        loss = [0]
        t = trange(n_epochs, desc="Loss = {}".format(np.mean(loss)), leave=True)
        for _ in t:
            done = False
            while not done:
                done, batch, targets = self.batcher.next()
                loss.append(self.backpropagate(batch, targets))
            t.set_description("Loss = {}".format(np.mean(loss)))

    def forward(self, inpt, targets=None):
        for layer in self.layers:
            inpt = layer.forward(inpt)
        out = self.loss.forward(inpt, targets) if targets is not None else inpt
        return out

    def backpropagate(self, batch, targets):
        # perform forward pass and store activations
        losses = self.forward(batch, targets)
        # initial delta
        deltas = self.loss.gradient(batch, targets)
        # backpropagate deltas
        for layer in self.layers[::-1]:
            deltas = layer.activation.derivative(layer.weighted_input_memory) * deltas
            deltas = layer.backward(deltas)
        return losses.mean()
