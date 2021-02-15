import numpy as np
import math
from network.layers.dense import DenseLayer
from network.layers.input import InputLayer
from network.layers.softmax import SoftmaxOutputLayer
from network.activation.sigmoid import Sigmoid
from network.activation.relu import Relu
from network.activation.tanh import TanH
from network.loss.mse import MSE
from network.loss.crossentropy import CrossEntropy


class Network:
    def __init__(self, loss, layers_config, wreg, wrt):
        self.loss = get_loss(loss)
        self.wreg = wreg
        self.wrt = wrt
        self.network = self.build_network(layers_config)


    def build_network(self, layers_config):
        layers = []

        # add input layer
        input_layer = InputLayer(layers_config[0]['size'])
        layers.append(input_layer)

        # add hidden layers
        prev_layer = input_layer
        for layer_config in layers_config[1:-1]:
            layer = DenseLayer(layer_config['size'],
                               prev_layer,
                               activation=get_activation(layer_config['activation']),
                               learning_rate=layer_config['learning_rate'],
                               wreg=self.wreg,
                               wrt=self.wrt if self.wrt is not None else None)
            layers.append(layer)
            prev_layer = layer

        # add output layer
        output_config = layers_config[-1]
        if output_config['activation'] == 'softmax':
            linear_layer = DenseLayer(output_config['size'],
                                      prev_layer,
                                      learning_rate=output_config['learning_rate'],
                                      wreg=self.wreg,
                                      wrt=self.wrt if self.wrt is not None else None)
            softmax_layer = SoftmaxOutputLayer(output_config['size'])
            layers.append(linear_layer)
            layers.append(softmax_layer)

        else:
            output_layer = DenseLayer(output_config['size'],
                                      prev_layer,
                                      activation=get_activation(output_config['activation']),
                                      learning_rate=output_config['learning_rate'],
                                      wreg=self.wreg,
                                      wrt=self.wrt if self.wrt is not None else None)
            layers.append(output_layer)

        return layers


    def fit(self, train_set, train_targets, val_set, val_targets, batch_size=32, epochs=10, verbose=False):
        return self.run(True, train_set, train_targets, val_set, val_targets, batch_size, epochs, verbose)


    def predict(self, data_set, target_set):
        return self.run(False, data_set, target_set, None, None, None, 1, False)


    def run(self, train, data_set, target_set, val_set, val_targets, batch_size, epochs, verbose):
        loss_history_train = []
        loss_history_val = [] if train else None
        for epoch in range(epochs):
            # divide data into batches
            data_batches = batches(data_set, batch_size)
            targets_batches = batches(target_set, batch_size)

            # for each batch
            for i in range(len(data_batches) if batch_size else 1):
                # Propagate values through network to produce output
                inputs = data_batches[i] if batch_size else data_set  # data batch i
                targets = targets_batches[i] if batch_size else target_set  # target batch i
                network_inputs = inputs  # save inputs for printing
                for layer in self.network:
                    inputs = layer.forward_pass(inputs)

                # Evaluate performance
                outputs = inputs
                loss = self.loss.error(outputs, targets)
                loss_history_train.append(loss)

                if verbose:
                    print('\nInputs')
                    print(network_inputs)
                    print('\nOutputs')
                    print(outputs)
                    print('\nTargets')
                    print(targets)
                    print('\nLoss: ', loss)
                    print('='*100)

                if train:
                    # Backpropagate and update weights
                    J_L_Z = self.loss.derivative(outputs, targets)
                    for layer in reversed(self.network):
                        J_L_Z = layer.backward_pass(J_L_Z)

            # predict validation data
            if train and val_set is not None and val_targets is not None:
                # Propagate validation values through network to produce output
                val_inputs = val_set
                val_targets = val_targets
                for layer in self.network:
                    val_inputs = layer.forward_pass(val_inputs)

                # Evaluate performance
                val_outputs = val_inputs
                val_loss = self.loss.error(val_outputs, val_targets)
                loss_history_val.append(val_loss)

        return loss_history_train, loss_history_val


def batches(data, batch_size):
    if not batch_size:
        return None
    number_of_batches = math.ceil(len(data) / batch_size)
    return np.array_split(data, number_of_batches)


def get_loss(loss):
    if loss == 'mse':
        return MSE
    if loss == 'cross_entropy':
        return CrossEntropy

    print('Invalid loss function')
    return None


def get_activation(act):
    if act == 'sigmoid':
        return Sigmoid
    if act == 'relu':
        return Relu
    if act == 'tanh':
        return TanH
    # else linear
    return None

