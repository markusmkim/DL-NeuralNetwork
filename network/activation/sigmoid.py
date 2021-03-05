import numpy as np


class Sigmoid:
    @staticmethod
    # apply the sigmoid function
    def apply(tensor):
        return 1 / (1 + np.exp(-tensor))


    @staticmethod
    def derivative(node_outputs, only_same_shape=False):
        flat_derivatives = node_outputs * (1 - node_outputs)
        if only_same_shape:
            return flat_derivatives
        diagonals = []
        for output in node_outputs:
            derivatives = output * (1 - output)
            diagonal_derivatives = np.diag(derivatives)
            diagonals.append(diagonal_derivatives)
        return flat_derivatives, np.array(diagonals)

    """
    @staticmethod
    def conv_derivative(node_outputs):
        return node_outputs * (1 - node_outputs)
    """