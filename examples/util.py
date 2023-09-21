import numpy as np
import pandas as pd

# Compute accuracy of some predictions y_pred with respect to true labels
# y_true.
def accuracy(y_pred, y_true):
    num = np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)
    den = len(y_true)
    return np.sum(num) / den

# Compute accuracy of the trained model predict(p, -).
def model_accuracy(predict, p, x, y):
    y_hats = np.zeros_like(y)
    for i in range(0, len(x)):
        y_hats[i] = predict(*p, *x[i])
    return accuracy(y_hats, y)

# Load the iris data from a given path, and scale it by multiplying inputs and
# outputs by 'scale'.
def load_iris(path, scale: int):
    iris = pd.read_csv(path)

    # load training data
    train_input = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()

    # construct labels manually since data is ordered by class
    train_labels = np.repeat([0,1,2], 50)

    # one-hot encode 3 classes
    train_labels = np.identity(3)[train_labels]

    # NOTE: input data is only down to 1dp, so we multiply by 10 and cast to int.
    return (train_input * scale).astype(int), (train_labels * scale).astype(int)
