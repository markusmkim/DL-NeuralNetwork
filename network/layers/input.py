

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
        input_batch = wrap_channel(input_batch)
        self.present_outputs = input_batch
        return input_batch

    def backward_pass(self, jacobian_L_Z):
        pass


def wrap_channel(tensor):
    return tensor.reshape((tensor.shape[0], 1) + (tensor.shape[1:]))