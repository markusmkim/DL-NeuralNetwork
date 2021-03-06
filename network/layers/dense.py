import numpy as np
from network.layers.utils import reshape_to_4d


class DenseLayer:
    def __init__(self, size, prev_layer, activation=None, learning_rate=0.1, wreg=None, wrt=0.001, wrap_output=False):
        self.type = 'dense'
        self.size = size
        self.prev_layer = prev_layer
        self.activation = activation
        self.learning_rate = learning_rate
        self.wreg = wreg
        self.wrt = wrt
        self.wrap_output = wrap_output
        self.num_filters = 1  # used by next layer if it is a convolutional layer
        self.present_inputs = None
        self.present_outputs = None
        if prev_layer.type != 'conv':
            self.weights = self.initialize_weights(prev_layer.size)
        else:
            self.weights = None
        self.biases = self.initialize_biases()

    # initialize weights from a uniform distribution between -0.1 and 0.1
    def initialize_weights(self, prev_size):
        return (np.random.rand(prev_size, self.size) / 5) - 0.1

    # initialize biases from a uniform distribution between -0.1 and 0.1
    def initialize_biases(self):
        return (np.random.rand(1, self.size) / 5) - 0.1


    def forward_pass(self, input_batch):
        self.present_inputs = input_batch

        if self.prev_layer.type == 'conv' and self.weights is None:
            # input should be flat
            # print('kkk', input_batch.shape)
            prev_size = input_batch.shape[1]
            self.weights = self.initialize_weights(prev_size)

        # apply activation function if supplied, else just pass the incoming values on
        weighted_sum = np.dot(input_batch, self.weights) + self.biases
        output_batch = self.activation.apply(weighted_sum) if self.activation else weighted_sum

        self.present_outputs = output_batch

        if self.wrap_output:
            output_batch = reshape_to_4d(output_batch)

        return output_batch


    def backward_pass(self, jacobian_L_Z):
        if self.wrap_output:
            jacobian_L_Z = jacobian_L_Z.reshape(self.present_outputs.shape)

        # calculate the jacobian of Z with regard to the input sum
        if self.activation:
            jacobian_Z_sum_diag_flattened, jacobian_Z_sum = self.activation.derivative(self.present_outputs)
        else:
            jacobian_Z_sum_diag_flattened, jacobian_Z_sum = self.jacobian_Z_sum(self.present_outputs)

        # calculate the gradients of the loss with regards to weights and biases
        y_outputs = self.present_inputs
        jacobian_Z_W = self.jacobian_Z_W(y_outputs, jacobian_Z_sum_diag_flattened)
        jacobian_Z_B = self.jacobian_Z_B(np.full((len(y_outputs), 1), 1), jacobian_Z_sum_diag_flattened)

        jacobian_L_W = self.jacobian_L_W(jacobian_L_Z, jacobian_Z_W)
        jacobian_L_B = self.jacobian_L_B(jacobian_L_Z, jacobian_Z_B)

        # update weights and biases
        self.update_weights(jacobian_L_W)
        self.update_bias(jacobian_L_B)

        # calculate the gradients of the loss with regards to previous layer Y outputs and pass backwards
        jacobian_Z_Y = self.jacobian_Z_Y(jacobian_Z_sum, self.weights)
        jacobian_L_Y = self.jacobian_L_Y(jacobian_L_Z, jacobian_Z_Y)
        return jacobian_L_Y


    def jacobian_Z_Y(self, jacobian_Z_sum, weights_z):
        return np.dot(jacobian_Z_sum, np.transpose(weights_z))


    def jacobian_L_Y(self, jacobian_L_Z, jacobian_Z_Y):
        # dot product elementwise in batch axis (i = batch axis)
        return np.einsum('ij,ijk->ik', jacobian_L_Z, jacobian_Z_Y)


    def update_weights(self, jacobian_L_W):
        gradients = jacobian_L_W
        # apply regularization if specified
        gradients = self.regulate_gradients(gradients)

        # gradients summed over the entire batch
        summed_gradients = np.sum(gradients, axis=0)
        # update rule: w = w - learningrate * gradient
        self.weights = self.weights - (self.learning_rate * summed_gradients)


    def update_bias(self, jacobian_L_B):
        # gradients summed over the entire batch
        summed_gradients = np.sum(jacobian_L_B, axis=0)
        # update rule: b = b - learningrate * gradient
        self.biases = self.biases - (self.learning_rate * summed_gradients)


    def regulate_gradients(self, gradients):
        if self.wrt == 'L2':
            return gradients + (self.wreg * self.weights)

        if self.wrt == 'L1':
            return gradients + (self.wreg + np.sign(self.weights))

        # else do not regulate
        return gradients


    def jacobian_L_W(self, jacobian_L_Z, jacobian_Z_W):
        # gradients per weight, keep batch dimension i unchanged
        return np.einsum('ij,ikj->ikj', jacobian_L_Z, jacobian_Z_W)


    def jacobian_L_B(self, jacobian_L_Z, jacobian_Z_B):
        # gradients per weight, keep batch dimension i unchanged
        return np.einsum('ij,ikj->ikj', jacobian_L_Z, jacobian_Z_B)


    def jacobian_Z_W(self, y_outputs, J_Z_sum_diag_flattened):
        # outer product elementwise in batch axis (i = batch axis)
        return np.einsum('ij,ik->ijk', y_outputs, J_Z_sum_diag_flattened)


    def jacobian_Z_B(self, biases, J_Z_sum_diag_flattened):
        # outer product elementwise in batch axis (i = batch axis)
        return np.einsum('ij,ik->ijk', biases, J_Z_sum_diag_flattened)

    # used if no activation function supplied (=linear activation), returns identity matrix + flattened identity matrix
    def jacobian_Z_sum(self, outputs):
        batch_size = len(outputs)
        number_of_outputs = len(outputs[0])
        flat_identity_matrices = np.ones((batch_size, number_of_outputs))
        identity_matrices = []
        for i in range(batch_size):
            identity_matrices.append(np.identity(number_of_outputs))

        return flat_identity_matrices, np.array(identity_matrices)




