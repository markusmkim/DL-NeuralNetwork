import numpy as np


class HiddenLayer:
    def __init__(self, size, prev_layer, activation=None, learning_rate=0.1):
        self.size = size
        self.prev_layer = prev_layer
        self.activation = activation
        self.learning_rate = learning_rate
        self.present_outputs = None
        self.weights = self.initialize_weights()


    def initialize_weights(self):
        return (np.random.rand(self.prev_layer.size, self.size) / 5) - 0.05


    def forward_pass(self, input_batch):
        # apply activation function if supplied, else just pass the incoming values on
        weighted_sum = np.dot(input_batch, self.weights)
        output_batch = self.activation.apply(weighted_sum) if self.activation else weighted_sum
        self.present_outputs = output_batch
        return output_batch


    def backward_pass(self, jacobian_L_Z):
        if self.activation:
            jacobian_Z_sum_diag_flattened, jacobian_Z_sum = self.activation.derivative(self.present_outputs)
        else:
            jacobian_Z_sum_diag_flattened, jacobian_Z_sum = self.jacobian_Z_sum(self.present_outputs)
        jacobian_Z_W = self.jacobian_Z_W(self.prev_layer.present_outputs, jacobian_Z_sum_diag_flattened)
        jacobian_L_W = self.jacobian_L_W(jacobian_L_Z, jacobian_Z_W)
        self.update_weights(jacobian_L_W)

        jacobian_Z_Y = self.jacobian_Z_Y(jacobian_Z_sum, self.weights)
        jacobian_L_Y = self.jacobian_L_Y(jacobian_L_Z, jacobian_Z_Y)
        return jacobian_L_Y

    def jacobian_Z_Y(self, jacobian_Z_sum, weights_z):
        return np.dot(jacobian_Z_sum, np.transpose(weights_z))


    def jacobian_L_Y(self, jacobian_L_Z, jacobian_Z_Y):
        # dot product elementwise in batch axis (i = batch axis)
        return np.einsum('ij,ijk->ik', jacobian_L_Z, jacobian_Z_Y)


    def update_weights(self, jacobian_L_W):
        # gradients summed over the entire batch
        summed_gradients = np.sum(jacobian_L_W, axis=0)
        # update rule: w = w - learningrate * gradient
        self.weights = self.weights - (self.learning_rate * summed_gradients)


    def jacobian_L_W(self, jacobian_L_Z, jacobian_Z_W):
        # gradients per weight, keep batch dimension i unchanged
        return np.einsum('ij,ikj->ikj', jacobian_L_Z, jacobian_Z_W)


    def jacobian_Z_W(self, y_outputs, J_Z_sum_diag_flattened):
        # outer product elementwise in batch axis (i = batch axis)
        return np.einsum('ij,ik->ijk', y_outputs, J_Z_sum_diag_flattened)


    # used if no activation function supplied, returns identity matrix + flattened identity matrix
    def jacobian_Z_sum(self, outputs):
        batch_size = len(outputs)
        number_of_outputs = len(outputs[0])
        flat_identity_matrices = np.ones((batch_size, number_of_outputs))
        identity_matrices = []
        for i in range(batch_size):
            identity_matrices.append(np.identity(number_of_outputs))

        return flat_identity_matrices, np.array(identity_matrices)




