from unittest import TestCase

import numpy as np
from tqdm import trange

from base import Batcher
from cnn import FC, Identity, SGD, MSE, Conv2D


class TestFC(TestCase):
    def test_forward(self):
        input_size = 5
        batch_size = 3
        inpt = np.random.rand(batch_size, input_size)
        fc = FC(input_size, n_neurons=2, activation=Identity())
        out = fc.forward(inpt)
        expected_shape = (batch_size, 2)
        if not out.shape == expected_shape:
            self.fail(
                "Wrong output shape {} for expected {}".format(
                    out.shape, expected_shape
                )
            )

        def iterative_forward(inpt_):
            res = np.zeros(expected_shape)
            for i, elem in enumerate(inpt_):
                res[i] = fc.weights @ np.concatenate([elem, [1]])
            return res

        expected_res = iterative_forward(inpt)
        if not np.allclose(out, expected_res):
            self.fail("Output {} different from expected {}".format(out, expected_res))

    def test_backward(self):
        data = np.random.rand(100, 3)
        targets = np.zeros((100, 2))
        input_size = 3
        batcher = Batcher(data, targets, batch_size=20)
        fc = FC(input_size, n_neurons=2, activation=Identity())
        fc.optimizer = SGD(lr=0.1)
        loss = MSE()
        for _ in range(10000):
            done, batch, targets = batcher.next()
            pred = fc.forward(batch)
            deltas = loss.gradient(pred, targets)
            fc.backward(deltas)
        if not np.allclose(fc.weights, np.zeros_like(fc.weights)):
            self.fail("FC layer weights {} did not converge to 0.".format(fc.weights))

    def test_compute_grad(self):
        # tagets are assumed to be [0, 0]
        inpt = [[1, 0], [0, 1]]
        input_size = 2
        fc = FC(input_size, n_neurons=1, activation=Identity())
        # override with test weights
        fc.weights = np.ones(2).reshape(-1, 1)
        # test error gradient
        fc.forward(inpt)
        deltas = 2 * np.ones(2).reshape(-1, 1)
        grads = fc.compute_grads(deltas, return_deltas=False)
        expected = [[2, 0, 2], [0, 2, 2]]
        if not np.allclose(np.squeeze(grads), expected):
            self.fail(
                "Output grads {} different than expected grads {}".format(
                    grads, expected
                )
            )


class TestConv2D(TestCase):
    def test_compute_grad_shape(self):
        """Test shapes."""
        batch_size = 4
        input_shape = (5, 5, 1)
        batch = np.random.rand(batch_size, *input_shape)
        kernel_size = 3
        n_filters = 2
        conv = Conv2D(input_shape, kernel_size, n_filters, activation=Identity())
        out = conv.forward(batch)
        # fake deltas
        deltas = np.random.rand(*out.shape)
        grads, deltas = conv.compute_grads(deltas)
        grad = grads.mean(axis=0)
        if not grad.shape == conv.weights.shape:
            self.fail(
                "grads shape mismatch. Out {} vs. Expected {}".format(
                    grad.shape, conv.weights.shape
                )
            )
        if not deltas.shape == batch.shape:
            self.fail(
                "deltas shape mismatch. Out {} vs. Expected {}".format(
                    deltas.shape, batch.shape
                )
            )

    def test_compute_grad(self):
        def test(full):
            batch_size = 20
            input_shape = (3, 3, 1)
            n_samples = 1000
            n_targets = 9 if full else 4
            images = np.random.rand(n_samples, *input_shape)
            kernel_size = 2
            n_filters = 1
            targets = np.zeros((n_samples, n_targets))
            batcher = Batcher(images, targets, batch_size)
            conv = Conv2D(
                input_shape, kernel_size, n_filters, activation=Identity(), full=full
            )
            conv.optimizer = SGD(lr=0.1)
            loss = MSE()
            loss_value = np.inf
            n_epochs = 10
            t = trange(n_epochs, desc="Loss = {}".format(np.mean(loss_value)), leave=True)
            for _ in t:
                loss_list = []
                done = False
                while not done:
                    done, images_batch, targets_batch = batcher.next()
                    pred = conv.forward(images_batch)
                    pred = pred.reshape(-1, n_targets)
                    loss_value = loss.forward(pred, targets_batch)
                    deltas = loss.gradient(pred, targets_batch)
                    deltas = deltas.reshape(
                        (-1, int(np.sqrt(n_targets)), int(np.sqrt(n_targets)), 1)
                    )
                    deltas = conv.activation.derivative(conv.weighted_input_memory) * deltas
                    conv.backward(deltas)
                    loss_list.append(loss_value.mean())
                t.set_description("Loss = {}".format(np.mean(loss_list)))
            if not np.allclose(conv.weights, np.zeros_like(conv.weights), atol=1e-2):
                self.fail(
                    "Conv layer weights {} did not converge to 0 with full={}.".format(conv.weights, full)
                )
        test(True)
        test(False)
