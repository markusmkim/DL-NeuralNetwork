class InputLayer:
    def __init__(self, size):
        self.size = size
        self.present_outputs = None

    def forward_pass(self, input_batch):
        self.present_outputs = input_batch
        return input_batch

    def backward_pass(self, jacobian_L_Z):
        pass