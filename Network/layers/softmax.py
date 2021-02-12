import numpy as np


class SoftmaxOutputLayer:
    def __init__(self, size):
        self.size = size
        self.present_outputs = None


    def forward_pass(self, input_batch):
        exponentials = np.exp(input_batch)  # dimension = (batch size, number of output nodes)
        sum_exponentials = exponentials.sum(axis=1)  # row vector of length = batch size
        sum_exponentials_column_vector = sum_exponentials[:, None]  # reshape to column vector w/dim = (batch size, 1)
        output_batch = exponentials / sum_exponentials_column_vector

        self.present_outputs = output_batch
        return output_batch


    def backward_pass(self, jacobian_L_S):
        jacobian_S_Z = self.jacobian_S_Z(self.present_outputs)  # Jsoft
        jacobian_L_Z = self.jacobian_L_Z(jacobian_L_S, jacobian_S_Z)
        return jacobian_L_Z


    def jacobian_L_Z(self, jacobian_L_S, jacobian_S_Z):
        # dot product elementwise in batch axis (i = batch axis)
        return np.einsum('ij,ijk->ik', jacobian_L_S, jacobian_S_Z)


    def jacobian_S_Z(self, s_output_batch):
        jacobian_S_Z_batch = []
        for output in s_output_batch:
            dim = len(output)
            jacobian_S_Z = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(dim):
                    if i == j:
                        jacobian_S_Z[i][j] = output[i] - (output[i]**2)
                    else:
                        jacobian_S_Z[i][j] = - output[i]*output[j]

            jacobian_S_Z_batch.append(jacobian_S_Z)

        return np.array(jacobian_S_Z_batch)

