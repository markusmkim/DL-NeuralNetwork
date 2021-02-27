import numpy as np
from network.layers.helpers.generators import generate_maps


class ConvolutionalLayer:
    def __init__(self, filter_shape, num_filters, strides, modes, prev_filters, prev_layer, activation=None, learning_rate=0.1, wreg=None, wrt=0.001):
        self.filter_shape = filter_shape
        self.num_filters = num_filters
        self.strides = strides
        self.modes = modes
        self.size = 0  # need for next layer if dense
        self.prev_layer = prev_layer
        self.activation = activation
        self.learning_rate = learning_rate
        self.wreg = wreg
        self.wrt = wrt
        self.present_outputs = None
        self.prev_filters = prev_filters  # this one needs to be derived from prev layer
        self.weights = self.initialize_weights()

    # initialize weights from a uniform distribution between -0.1 and 0.1
    def initialize_weights(self):
        return np.ones((self.num_filters,
                        self.prev_filters,
                        self.filter_shape[0],
                        self.filter_shape[1]))
        #  return (np.random.rand(self.num_filters, self.prev_filters, self.filter_shape[0], self.filter_shape[1]) / 5) - 0.1


    def forward_pass(self, input_batch):
        output_batch = []
        for batch_case in input_batch:
            output_maps = []
            for weight_index in range(len(self.weights)):
                input_maps_kernalized = []
                for input_map_index in range(len(batch_case)):
                    input_map = batch_case[input_map_index]
                    kernel = self.weights[weight_index][input_map_index]
                    input_map_kernalized = self.apply_kernel(input_map, kernel)
                    input_maps_kernalized.append(input_map_kernalized)

                output_map = np.sum(input_maps_kernalized, axis=0)
                output_maps.append(output_map)

            output_maps = np.array(output_maps)
            # apply activation function if supplied, else just pass the incoming values on
            output_maps = self.activation.apply(output_maps) if self.activation else output_maps
            output_batch.append(output_maps)

        output_batch = np.array(output_batch)
        self.present_outputs = output_batch
        return output_batch


    def apply_kernel(self, input_map, kernel, strides=1, mode='same'):
        output_map, padded_map = generate_maps(input_map, kernel, 1, 'full')
        # print(padded_map)
        # print(output_map)

        for row in range(output_map.shape[0]):
            for col in range(output_map.shape[1]):
                weighted_sum = 0
                # Y(k) = sum (-d, d) { x(k + d) w(d) }
                for kernel_row_index in range(kernel.shape[0]):
                    for kernel_col_index in range(kernel.shape[1]):
                        weight = kernel[kernel_row_index][kernel_col_index]
                        map_entry = padded_map[row + kernel_row_index][col + kernel_col_index]

                        weighted_sum += weight * map_entry

                output_map[row][col] = weighted_sum

        return output_map


    def backward_pass(self, jacobian_L_Z):
        # calculate the jacobian of Z with regard to the input sum
        if self.activation:
            jacobian_Z_sum_diag_flattened, jacobian_Z_sum = self.activation.derivative(self.present_outputs)
        else:
            jacobian_Z_sum_diag_flattened, jacobian_Z_sum = self.jacobian_Z_sum(self.present_outputs)

        # calculate the gradients of the loss with regards to weights and biases
        y_outputs = self.prev_layer.present_outputs
        jacobian_Z_W = self.jacobian_Z_W(y_outputs, jacobian_Z_sum_diag_flattened)

        jacobian_L_W = self.jacobian_L_W(jacobian_L_Z, jacobian_Z_W)

        # update weights and biases
        self.update_weights(jacobian_L_W)

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


    def jacobian_Z_W(self, y_outputs, J_Z_sum_diag_flattened):
        # outer product elementwise in batch axis (i = batch axis)
        return np.einsum('ij,ik->ijk', y_outputs, J_Z_sum_diag_flattened)

    # used if no activation function supplied (=linear activation), returns identity matrix + flattened identity matrix
    def jacobian_Z_sum(self, outputs):
        batch_size = len(outputs)
        number_of_outputs = len(outputs[0])
        flat_identity_matrices = np.ones((batch_size, number_of_outputs))
        identity_matrices = []
        for i in range(batch_size):
            identity_matrices.append(np.identity(number_of_outputs))

        return flat_identity_matrices, np.array(identity_matrices)


