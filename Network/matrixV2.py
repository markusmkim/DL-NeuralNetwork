import numpy as np
import math
from Network.layers.hidden import HiddenLayer
from Network.layers.input import InputLayer
from Network.layers.softmax import SoftmaxOutputLayer
from Network.activation.sigmoid import Sigmoid
from Network.loss.mse import MSE
from Network.loss.crossentropy import CrossEntropy


def batches(data, batch_size):
    number_of_batches = math.ceil(len(data) // batch_size)
    return np.split(data, number_of_batches)


class Network:
    def __init__(self, loss):
        self.loss = loss
        self.network = self.build_network()

    def build_network(self):
        input_layer = InputLayer(3)
        hidden_layer_1 = HiddenLayer(10, input_layer, activation=Sigmoid)
        hidden_layer_2 = HiddenLayer(2, hidden_layer_1)
        softmax_layer = SoftmaxOutputLayer(2)

        return [input_layer, hidden_layer_1, hidden_layer_2, softmax_layer]

    def fit(self, training_data, training_targets, epochs=10):
        for epoch in range(epochs):
            # divide data into batches
            training_data_batches = batches(training_data, 3)
            training_targets_batches = batches(training_targets, 3)

            # for each batch
            for i in range(len(training_data_batches)):

                # Propagate values through network to produce output
                inputs = training_data_batches[i]  # training batch i
                targets = training_targets_batches[i]  # target batch i
                for layer in self.network:
                    inputs = layer.forward_pass(inputs)

                # Evaluate performance
                outputs = inputs
                loss, batch_loss_average = self.loss.error(outputs, targets)
                print('\nOutputs')
                print(outputs)
                #print('Loss: ', loss)
                print('Targets')
                print(targets)
                print('Batch average loss')
                print(batch_loss_average)

                # Backpropagate and update weights
                J_L_Z = MSE.derivative(outputs, targets)
                for layer in reversed(self.network):
                    J_L_Z = layer.backward_pass(J_L_Z)

            print('Epoch', epoch + 1, 'done')

# 3 - 2 - 2 network from slides, with batch size = 3
# 2 batches
inputs = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 0],
])

targets = np.array([
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
])


n = Network(CrossEntropy)
n.fit(inputs, targets, epochs=200)
