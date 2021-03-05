import numpy as np


class Relu:
    @staticmethod
    # apply the relu function
    def apply(tensor):
        def relu_func(a, b):
            return a if a > b else 0.0
        vectorizer = np.vectorize(relu_func)
        return vectorizer(tensor, 0)


    @staticmethod
    def derivative(node_outputs, only_same_shape=False):
        def derivative_func(a, b):
            return 1.0 if a > b else 0.0
        vectorizer = np.vectorize(derivative_func)
        flat_derivatives = vectorizer(node_outputs, 0)
        if only_same_shape:
            return flat_derivatives
        diagonals = []
        for output in node_outputs:
            derivatives = vectorizer(output, 0)
            diagonal_derivatives = np.diag(derivatives)
            diagonals.append(diagonal_derivatives)
        return flat_derivatives, np.array(diagonals)

