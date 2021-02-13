import numpy as np
import math

##### OLD #####

def sigmoid(tensor):
    return tensor * 2

# if diagonal_matrix, return both
def sigmoid_derivative(node_outputs, return_diagonal_matrix=False):
    if return_diagonal_matrix:
        diagonals = []
        for output in node_outputs:
            derivatives = output
            diagonal_derivatives = np.diag(derivatives)
            diagonals.append(diagonal_derivatives)
        return node_outputs * 2, np.array(diagonals)

    return node_outputs * 2  # node_output * (node_output - 1)


def loss_derivative(z_tensor):
    return z_tensor - 1


# 3 - 2 - 2 network from slides, with batch size = 2
# input layer dimension 10
x = np.array([
    [2, 1, 2],
    [1, 2, 3]
])  # input
print(x)

# weights between input layer and hidden layer of size = 2
weights_x_y = np.array([
    [0.5, 0.5],
    [1, 1],
    [0.5, 2],
])
weights_y_z = np.array([
    [0.5, 1],
    [0.5, 0.5]
])

# forward pass to y
y = sigmoid(np.dot(x, weights_x_y))  # or use np.matmul?
print('\ny')
print(y)

# forward pass to z
z = np.dot(y, weights_y_z)
print('\nZ')
print(z)


# now lets build som jacobians, denne man får inn fra next layer
J_L_Z = loss_derivative(z)
print('\nJ_L_Z')
print(J_L_Z)


flat_diagonal, J_Z_sum = sigmoid_derivative(z, return_diagonal_matrix=True)  # this needs to be built as diagonal matrix when len(z) > 1
#print('\nJ_Z_sum')
#print(J_Z_sum)
print('\nflat')
print(flat_diagonal)

# np.outer(y, flat_diagonal)
J_hat_w_output_Z = np.einsum('ij,ik->ijk', y, flat_diagonal)
print("\nJ_hat_w_output_Z")
print(J_hat_w_output_Z)

# gradients for weights
J_L_w = np.einsum('ij,ikj->ikj', J_L_Z, J_hat_w_output_Z)  # J_L_Z * J_hat_w_output_Z
print('\nJ_L_w')
print(J_L_w)

J_Z_Y = np.dot(J_Z_sum, np.transpose(weights_y_z))

print('\nJ_Z_Y')
print(J_Z_Y)

# dot product elementwise in batch axis (i = batch axis)
J_L_Y = np.einsum('ij,ijk->ik', J_L_Z, J_Z_Y)
print('\nJ_L_Y')
print(J_L_Y)


_, J_Y_sum = sigmoid_derivative(y, return_diagonal_matrix=True)


J_Y_X = np.dot(J_Y_sum, np.transpose(weights_x_y))
print('\nJ_Y_X')
print(J_Y_X)


J_L_X = np.einsum('ij,ijk->ik', J_L_Y, J_Y_X)
print('\nJ_L_X')
print(J_L_X)


