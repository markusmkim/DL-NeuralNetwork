import numpy as np
import math


# input layer dimension 10
input_vector = np.arange(10).reshape(10, )
print(input_vector)


# weights between input layer and hidden layer of size = 2
weights = np.array([
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 2],
])

weights_T = np.transpose(weights)

print(weights_T)

values_hidden_layer = np.matmul(weights_T, input_vector)
print(values_hidden_layer)
