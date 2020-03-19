import logging

import numpy as np
from tqdm import trange

from base import Activation, Loss, Optimizer, Layer, Batcher
from conv import convolution, grad_convolution, deltas_convolution

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


def softmax(inpt, epsilon=1e-6):
    # substract max component for each vector to prevent overlflow
    exp_x = np.exp(inpt - inpt.max(axis=1)[:, None])
    return exp_x / exp_x.sum(axis=1)[:, None]


class SoftmaxCrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, inpt, targets):
        return -(targets * np.log(softmax(inpt) + 1e-6)).sum(axis=1)

    def gradient(self, inpt, targets):
        softmax_inpt = softmax(inpt)
        return softmax_inpt - targets


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
    def __init__(self, input_shape, n_neurons, activation: Activation = Identity()):
        super().__init__()
        self.input_shape = input_shape
        self.input_size = input_shape[0]
        self.n_neurons = n_neurons
        self.activation = activation
        self.initialize_weights()

    def initialize_weights(self):
        self.weights = (
            2
            / self.input_size
            * np.random.normal(0, 1, (self.n_neurons, self.input_size + 1))
        )

    def forward(self, inpt):
        self.input_memory = np.column_stack([inpt, np.ones(inpt.shape[0])])
        self.weighted_input_memory = np.einsum(
            "ij,bj->bi", self.weights, self.input_memory
        )
        return self.activation.forward(self.weighted_input_memory)

    def backward(self, deltas, optimizer):
        grads, deltas = self.compute_grads(deltas)
        self.weights = optimizer.step(self.weights, grads)
        return deltas

    def compute_grads(self, deltas, return_deltas=True):
        grads = np.einsum("bj,bi->bji", deltas, self.input_memory)
        # backprop recursion
        deltas = np.einsum("ij,bj->bi", self.weights.T, deltas)
        if return_deltas:
            # remove bias dimension
            return grads, deltas[:, :-1]
        return grads


class Flatten(Layer):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def initialize_weights(self):
        pass

    def forward(self, inpt):
        self.input_shape = inpt.shape[1:]
        return inpt.reshape(inpt.shape[0], -1)

    def compute_grads(self, deltas):
        pass

    def backward(self, deltas, optimizer):
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
        self.weights = np.random.normal(
            0,
            1,
            (self.n_filters, self.kernel_size, self.kernel_size, self.input_shape[-1]),
        )
        self.weights = (
            np.sqrt(2)
            / np.sqrt(self.kernel_size * self.kernel_size * self.input_shape[-1])
        ) * self.weights

    def forward(self, inpt):
        self.input_memory = inpt
        self.weighted_input_memory = convolution(inpt, self.weights, self.full)
        return self.activation.forward(self.weighted_input_memory)

    def backward(self, deltas, optimizer):
        grads, deltas = self.compute_grads(deltas)
        self.weights = optimizer.step(self.weights, grads)
        return deltas

    def compute_grads(self, deltas, return_deltas=True):
        grads = grad_convolution(deltas, self.input_memory, self.weights, self.full)
        deltas = deltas_convolution(deltas, self.weights)
        if return_deltas:
            return grads, deltas
        return grads


class CNN:
    def __init__(self, layers, loss: Loss, optimizer: Optimizer, pred_func=None):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.pred_func = pred_func

    def check_shape_compatibility(self, batcher):
        try:
            done, inpt, targets = batcher.next()
            self.forward(inpt)
        except Exception:
            logger.exception("Exception in forward pass, check your shapes.")
            raise

    def train(self, batcher, n_epochs):
        history = []
        loss = np.inf
        t = trange(n_epochs, desc="Loss = {}".format(np.mean(loss)), leave=True)
        for _ in t:
            done = False
            while not done:
                done, batch, targets = batcher.next()
                loss = self.backpropagate(batch, targets)
                history.append(loss)
                t.set_description("Loss = {}".format(np.mean(loss)))
        return history

    def forward(self, inpt, targets=None):
        for i, layer in enumerate(self.layers):
            inpt = layer.forward(inpt)
            if np.isnan(inpt).sum() > 0 or np.isinf(inpt).sum() > 0:
                logger.warning("Output of layer {} has NaNs.".format(i))
        if targets is not None:
            loss_value = self.loss.forward(inpt, targets)
            return inpt, loss_value
        else:
            return inpt

    def backpropagate(self, batch, targets):
        # perform forward pass and store activations
        output, loss_values = self.forward(batch, targets)
        # initial delta
        deltas = self.loss.gradient(output, targets)
        # backpropagate deltas
        for layer in self.layers[::-1]:
            if layer.activation is not None:
                deltas = (
                    layer.activation.derivative(layer.weighted_input_memory) * deltas
                )
            deltas = layer.backward(deltas, self.optimizer)
        return loss_values.mean()

    def predict(self, inpt):
        pred = self.forward(inpt)
        pred = self.pred_func(pred) if self.pred_func is not None else pred
        return pred
