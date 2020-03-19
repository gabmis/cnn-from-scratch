import numpy as np
import pandas as pd

from base import Batcher
from cnn import Conv2D, Flatten, FC, ReLU, CNN, SoftmaxCrossEntropy, SGD, softmax


def parse_mnist_kaggle(fname):
    df = pd.read_csv(fname)
    column_offset = 0
    targets_onehot = None
    if "label" in df.columns:
        targets = df.loc[:, "label"].to_numpy()
        targets_onehot = np.zeros((len(targets),  10))
        targets_onehot[:, targets] = 1
        column_offset = 1
    images = df.iloc[:, column_offset:].to_numpy().reshape(-1, 28, 28, 1) / 255
    return images, targets_onehot


images_train, targets_train = parse_mnist_kaggle("mnist_kaggle/train.csv")
images_test, _ = parse_mnist_kaggle("mnist_kaggle/test.csv")

n_epochs = 10
batch_size = 20

batcher = Batcher(images_train, targets_train, batch_size)

conv = Conv2D(input_shape=images_train.shape[1:], kernel_size=3, n_filters=32)
flatten = Flatten(input_shape=conv.output_shape)
fc_1 = FC(input_shape=flatten.output_shape, n_neurons=128, activation=ReLU())
fc_2 = FC(input_shape=fc_1.output_shape, n_neurons=10)
layers = [
    conv,
    flatten,
    fc_1,
    fc_2,
]
cnn = CNN(layers=layers, loss=SoftmaxCrossEntropy(), optimizer=SGD(), pred_func=softmax)
cnn.check_shape_compatibility(batcher)
cnn.train(batcher, n_epochs)

predictions = cnn.predict(images_test)
submission = np.argmax(predictions, axis=1)
submission = pd.DataFrame(submission, columns=["Label"])
submission.index.name = "ImageId"
submission.to_csv("mnist_kaggle/submission.csv")
