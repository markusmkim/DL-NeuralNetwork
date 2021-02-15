import numpy as np


class InputLayer:
    def __init__(self, size):
        self.size = size
        self.present_outputs = None

    def forward_pass(self, input_batch):
        # add bias output node
        #biases = np.ones((len(input_batch), 1))
        #input_batch_with_biases = np.hstack((input_batch, biases))
        #self.present_outputs = input_batch_with_biases
        #return input_batch_with_biases
        self.present_outputs = input_batch
        return input_batch

    def backward_pass(self, jacobian_L_Z):
        pass