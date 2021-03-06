import numpy as np
from network.layers.helpers.generators import generate_maps


class ConvolutionalLayer:
    def __init__(self, filter_shape, num_filters, stride, mode, prev_layer,
                 flatten_data=False, flatten_channels=False,  activation=None, learning_rate=0.1, wreg=None, wrt=0.001):
        self.type = 'conv'
        self.flatten_data = flatten_data
        self.flatten_channels = flatten_channels
        self.filter_shape = filter_shape
        self.num_filters = num_filters
        self.stride = stride
        self.mode = mode
        self.size = 0  # need for next layer if dense
        self.prev_layer = prev_layer
        self.activation = activation
        self.learning_rate = learning_rate
        self.wreg = wreg
        self.wrt = wrt
        self.present_inputs = None
        self.present_outputs = None
        self.prev_filters = prev_layer.num_filters  # this one needs to be derived from prev layer
        self.weights = self.initialize_weights()
        self.weight_impacts = None
        self.input_impacts = None

    # initialize weights from a uniform distribution between -0.1 and 0.1
    def initialize_weights(self):
        """
        return np.ones((self.num_filters,
                        self.prev_filters,
                        self.filter_shape[0],
                        self.filter_shape[1]))
        """
        return (np.random.rand(self.num_filters,
                               self.prev_filters,
                               self.filter_shape[0],
                               self.filter_shape[1]) / 5) - 0.1


    def initialize_impacts(self, shape):
        #print('ssss', self.num_filters, shape)
        impacts = {}
        for index_in_batch in range(shape[0]):
            impacts[index_in_batch] = {}
            for i in range(shape[1]):
                for ii in range(shape[2]):
                    for iii in range(shape[3]):
                        for iiii in range(shape[4]):
                            impacts[index_in_batch][(i, ii, iii, iiii)] = {}
        return impacts


    def forward_pass(self, input_batch):
        self.weight_impacts = self.initialize_impacts((input_batch.shape[0], ) + self.weights.shape)
        self.input_impacts = self.initialize_impacts((input_batch.shape[0], ) + (self.num_filters, ) + input_batch.shape[1:])
        self.present_inputs = input_batch
        # print('Getting in', input_batch.shape)
        output_batch = []
        for batch_case_index, batch_case in enumerate(input_batch):
            output_maps = []
            for weight_index in range(len(self.weights)):
                input_maps_kernalized = []
                for input_map_index in range(len(batch_case)):
                    input_map = batch_case[input_map_index]
                    input_map_kernalized = self.apply_kernel(batch_case_index, input_map, weight_index, input_map_index)
                    input_maps_kernalized.append(input_map_kernalized)

                output_map = np.sum(input_maps_kernalized, axis=0)
                output_maps.append(output_map)

            output_maps = np.array(output_maps)
            # apply activation function if supplied, else just pass the incoming values on
            output_maps = self.activation.apply(output_maps) if self.activation else output_maps
            output_batch.append(output_maps)

        output_batch = np.array(output_batch)
        self.present_outputs = output_batch

        if self.flatten_data:
            if self.flatten_channels:
                # assume 2d image dimensions are equal (image-ratio = 1:1)
                output_batch = output_batch.reshape((output_batch.shape[0],
                                                     output_batch.shape[1] * output_batch.shape[2]**2))
            else:
                output_batch = output_batch.reshape((output_batch.shape[0],
                                                     output_batch.shape[1], 1,
                                                     output_batch.shape[2]**2))
        elif self.flatten_channels:
            output_batch = output_batch.reshape((output_batch.shape[0],
                                                 output_batch.shape[1] * output_batch.shape[2] * output_batch.shape[3]))

        return output_batch


    def apply_kernel(self, batch_case_index, input_map, weight_index, input_map_index):
        kernel = self.weights[weight_index][input_map_index]
        output_map, padded_map, start_pad_row, start_pad_col = generate_maps(input_map, kernel, self.stride, self.mode)
        # counter = 0
        input_map_row_A = -start_pad_row
        pad_row_index = 0
        for row in range(output_map.shape[0]):
            input_map_col_A = -start_pad_col
            pad_col_index = 0
            for col in range(output_map.shape[1]):
                weighted_sum = 0
                # Y(k) = sum (-d, d) { x(k + d) w(d) }
                # weights_affecting_input = []
                for kernel_row_index in range(kernel.shape[0]):
                    for kernel_col_index in range(kernel.shape[1]):
                        weight = kernel[kernel_row_index][kernel_col_index]
                        map_entry = 0
                        map_entry_row = pad_row_index + kernel_row_index
                        map_entry_col = pad_col_index + kernel_col_index
                        if map_entry_row < padded_map.shape[0] and map_entry_col < padded_map.shape[1]:
                            map_entry = padded_map[map_entry_row][map_entry_col]
                        weighted_sum += weight * map_entry

                        # if map_entry != 0: DENNE er farlig
                        # for hver vekt, hvor mye påvirker vekten hver output
                        # weight_index = output_map_index
                        output_map_impact_key = (row, col)
                        weight_impact_key = (weight_index, input_map_index, kernel_row_index, kernel_col_index)
                        self.weight_impacts[batch_case_index][weight_impact_key][output_map_impact_key] = map_entry
                        # weight_impacts.append((output_map_impact_key, map_entry))
                        # self.weight_impacts[weight_impact_key] = weight_impacts

                        input_map_row = input_map_row_A + kernel_row_index
                        input_map_col = input_map_col_A + kernel_col_index
                        # må vite index for input_map her
                        # input_map_row = 0
                        # input_map_col = 0
                        if 0 <= input_map_row < input_map.shape[0] and 0 <= input_map_col < input_map.shape[1]:
                            # counter += 1
                            input_impact_key = (weight_index, input_map_index, input_map_row, input_map_col)
                            self.input_impacts[batch_case_index][input_impact_key][output_map_impact_key] = weight


                output_map[row][col] = weighted_sum

                pad_col_index += self.stride
                input_map_col_A += self.stride
            pad_row_index += self.stride
            input_map_row_A += self.stride
        # print('counter: ', counter)
        return output_map


    def backward_pass(self, jacobian_L_Z):
        if self.flatten_data or self.flatten_channels:
            # assume 2d image dimensions are equal (image-ratio = 1:1)
            jacobian_L_Z = jacobian_L_Z.reshape(self.present_outputs.shape)

        # calculate the jacobian of Z with regard to the input sum
        if self.activation:
            jacobian_Z_sum = self.activation.derivative(self.present_outputs, only_same_shape=True)
        else:
            jacobian_Z_sum = self.jacobian_Z_sum(self.present_outputs)

        # calculate the gradients of the loss with regards to weights
        # y_outputs = self.prev_layer.present_outputs
        # jacobian_Z_W = 0  # manuell  -  self.jacobian_Z_W(y_outputs, jacobian_Z_sum_diag_flattened)

        jacobian_L_W = self.jacobian_L_W(jacobian_L_Z, jacobian_Z_sum)

        # update weights
        self.update_weights(jacobian_L_W)

        #self.initialize_weight_impacts()

        # calculate the gradients of the loss with regards to previous layer Y outputs and pass backwards
        # jacobian_Z_Y = self.jacobian_Z_Y(jacobian_Z_sum, self.weights)
        # jacobian_L_Y = self.jacobian_L_Y(jacobian_L_Z, jacobian_Z_Y)
        jacobian_L_Y = self.jacobian_L_Y(jacobian_L_Z, jacobian_Z_sum)
        return jacobian_L_Y


    def jacobian_L_Y(self, jacobian_L_Z, jacobian_Z_sum):
        # print('L-Z')
        # print(jacobian_L_Z.shape)
        # print('Z-sum')
        # print(jacobian_Z_sum.shape)
        gradients = []
        # for every case in batch
        for batch_case_index in range(len(jacobian_L_Z)):
            gradients_batch_case = np.zeros(self.present_inputs.shape[1:])
            for input_map_index in range(gradients_batch_case.shape[0]):
                for row in range(gradients_batch_case.shape[1]):
                    for col in range(gradients_batch_case.shape[2]):
                        total_input_impact = 0
                        for output_map_index in range(self.num_filters):
                            input_impact_key = (output_map_index, input_map_index, row, col)
                            output_map_keys = self.input_impacts[batch_case_index][input_impact_key].keys()

                            for key in output_map_keys:
                                impact = self.input_impacts[batch_case_index][input_impact_key][key] * (
                                    jacobian_L_Z[batch_case_index][output_map_index][key[0]][key[1]]) * (
                                             jacobian_Z_sum[batch_case_index][output_map_index][key[0]][key[1]])
                                total_input_impact += impact

                        gradients_batch_case[input_map_index][row][col] = total_input_impact

            gradients.append(gradients_batch_case)

        gradients = np.array(gradients)
        # print('Passing back', gradients.shape)
        return gradients


    def update_weights(self, jacobian_L_W):
        gradients = jacobian_L_W
        # apply regularization if specified
        # gradients = self.regulate_gradients(gradients)

        # gradients summed over the entire batch
        summed_gradients = np.sum(gradients, axis=0)

        # update rule: w = w - learningrate * gradient
        self.weights = self.weights - (self.learning_rate * summed_gradients)


    def jacobian_L_W(self, jacobian_L_Z, jacobian_Z_sum):
        # print('L-Z')
        # print(jacobian_L_Z.shape)
        # print('Z-SUM')
        # print(jacobian_Z_sum.shape)
        gradients = []
        # for every case in batch
        for batch_case_index in range(len(jacobian_L_Z)):
            gradients_batch_case = np.zeros(self.weights.shape)
            for output_map_index in range(gradients_batch_case.shape[0]):
                for input_map_index in range(gradients_batch_case.shape[1]):
                    for weight_row in range(gradients_batch_case.shape[2]):
                        for weight_col in range(gradients_batch_case.shape[3]):
                            weight_impact_key = (output_map_index, input_map_index, weight_row, weight_col)
                            output_map_keys = self.weight_impacts[batch_case_index][weight_impact_key].keys()
                            total_weight_impact = 0
                            for key in output_map_keys:
                                impact = self.weight_impacts[batch_case_index][weight_impact_key][key] * (
                                    jacobian_L_Z[batch_case_index][output_map_index][key[0]][key[1]]) * (
                                    jacobian_Z_sum[batch_case_index][output_map_index][key[0]][key[1]])
                                total_weight_impact += impact

                            gradients_batch_case[output_map_index][input_map_index][weight_row][weight_col] = \
                                total_weight_impact

            gradients.append(gradients_batch_case)

        gradients = np.array(gradients)
        return gradients

    # used if no activation function supplied (=linear activation), returns identity matrix + flattened identity matrix
    def jacobian_Z_sum(self, outputs):
        return np.ones(outputs.shape)


    def regulate_gradients(self, gradients):
        if self.wrt == 'L2':
            return gradients + (self.wreg * self.weights)

        if self.wrt == 'L1':
            return gradients + (self.wreg + np.sign(self.weights))

        # else do not regulate
        return gradients



