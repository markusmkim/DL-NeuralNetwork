import numpy as np


class TanH:
    @staticmethod
    # apply the tanh function
    def apply(tensor):
        return (2 / (1 + np.exp(- 2 * tensor))) - 1


    @staticmethod
    def derivative(node_outputs):
        flat_derivatives = 1 - (node_outputs**2)
        diagonals = []
        for output in node_outputs:
            derivatives = 1 - (output**2)
            diagonal_derivatives = np.diag(derivatives)
            diagonals.append(diagonal_derivatives)
        return flat_derivatives, np.array(diagonals)
