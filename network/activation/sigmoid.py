import numpy as np


class Sigmoid:
    @staticmethod
    # apply the sigmoid function
    def apply(tensor):
        return 1 / (1 + np.exp(-tensor))


    # if diagonal_matrix, return both
    @staticmethod
    def derivative(node_outputs):
        flat_derivatives = node_outputs * (1 - node_outputs)
        diagonals = []
        for output in node_outputs:
            derivatives = output * (1 - output)
            diagonal_derivatives = np.diag(derivatives)
            diagonals.append(diagonal_derivatives)
        return flat_derivatives, np.array(diagonals)
