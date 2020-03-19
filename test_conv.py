from unittest import TestCase

import numpy as np

from conv import convolve2d, rot180


class Test(TestCase):
    def test_convolve2d(self):
        filtr = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        filtr = np.tile(filtr[:, :, None], [1, 1, 2])
        image = np.arange(25).reshape((5, 5))
        image = np.tile(image[None, :, :, None], [4, 1, 1, 2])
        out = convolve2d(image, filtr)
        expected = image.sum(axis=3)
        if not np.allclose(out, expected):
            self.fail("Output {} different from expected {}".format(out, expected))

        def iterative_conv(image, filtr):
            res = np.zeros((image.shape[0] - filtr.shape[0] + 1, image.shape[1] - filtr.shape[1] + 1))
            for i in range(image.shape[0] - filtr.shape[0] + 1):
                for j in range(image.shape[1] - filtr.shape[1] + 1):
                    for p in range(filtr.shape[0]):
                        for q in range(filtr.shape[1]):
                            res[i, j] += image[i + p, j + q] * filtr[p, q]
            return res

        image = np.random.rand(7, 7)
        filtr = np.random.rand(3, 3)
        out = convolve2d(image[None, ..., None], filtr[..., None], full=False)
        expected = iterative_conv(image, filtr)
        if not np.allclose(np.squeeze(out), expected):
            self.fail("Output {} different from expected {}".format(out, expected))

    def test_rot180(self):
        inpt = np.tile(np.arange(4).reshape(2, 2)[None, :, :], [3, 1, 1])
        expected = np.tile(np.array([[3, 2], [1, 0]])[None, :, :], [3, 1, 1])
        if not np.allclose(rot180(inpt), expected):
            self.fail()
