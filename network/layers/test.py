from network.layers.convolution import ConvolutionalLayer
import numpy as np
from network.activation.sigmoid import Sigmoid

layer = ConvolutionalLayer((3, 3), 3, 1, 'same', 2, None, activation=Sigmoid)
layer_2 = ConvolutionalLayer((3, 3), 5, 1, 'valid', 3, None, activation=Sigmoid)

input_batch = np.array([
    [
        [
            [5, 0, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
        ],
        [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
        ]
    ],
    [
        [
            [6, 1, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
        ],
        [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
        ]
    ]
])

a = layer.forward_pass(input_batch)
layer.backward_pass(a)
