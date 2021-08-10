from network.layers.utils import reshape_to_4d


class InputLayer:
    def __init__(self, size):
        self.type = 'd-input'
        self.size = size
        self.present_outputs = None

    def forward_pass(self, input_batch):
        self.present_outputs = input_batch
        return input_batch

    def backward_pass(self, jacobian_L_Z):
        pass


class ConvInputLayer:
    def __init__(self, num_filters):
        self.type = 'c-input'
        self.num_filters = num_filters
        self.present_outputs = None

    def forward_pass(self, input_batch):
        input_batch = reshape_to_4d(input_batch)
        self.present_outputs = input_batch
        return input_batch

    def backward_pass(self, jacobian_L_Z):
        pass

