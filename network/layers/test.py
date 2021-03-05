from network.layers.convolution import ConvolutionalLayer
from network.layers.dense import DenseLayer
from network.layers.input import ConvInputLayer
import numpy as np
from network.activation.sigmoid import Sigmoid
from network.layers.helpers.visualizer import plot_kernel

conv_input = ConvInputLayer(2)
layer = ConvolutionalLayer((3, 3), 3, 1, 'same', conv_input, activation=Sigmoid)
layer_2 = ConvolutionalLayer((3, 3), 5, 1, 'valid', layer, activation=Sigmoid, flatten=True)
dense = DenseLayer(10, layer_2)

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

conv_output = conv_input.forward_pass(input_batch)
print('Input shape:', input_batch.shape)
output1 = layer.forward_pass(conv_output)
print('Output1 shape', output1.shape)
output2 = layer_2.forward_pass(output1)
print('Output2 shape', output2.shape)
output_dense = dense.forward_pass(output2)
print('Output_dense shape', output_dense.shape)
backpass_dense = dense.backward_pass(output_dense)
print('Backpass dense shape', backpass_dense.shape)
back_pass2 = layer_2.backward_pass(backpass_dense)
print('Backpass 2 shape', back_pass2.shape)
back_pass1 = layer.backward_pass(back_pass2)
print('Revieved back shape', back_pass1.shape)
# b = layer.backward_pass(a)
layer_kernels = layer.weights
plot_kernel(layer_kernels)
"""
for b in layer.input_impacts:
    print('\n', b)
    print('-'*300)
    for k in layer.input_impacts[b]:
        print('\n', k)
        print(len(layer.input_impacts[b][k]))
"""
# print('result', b.shape)
