from network.layers.dense import DenseLayer
from network.layers.convolution import ConvolutionalLayer
from network.layers.input import InputLayer
from network.layers.input import ConvInputLayer
from network.layers.softmax import SoftmaxOutputLayer
from network.layers.helpers.visualizer import plot_kernel
from network.activation.sigmoid import Sigmoid
from network.activation.relu import Relu
from network.activation.tanh import TanH
from network.loss.mse import MSE
from network.loss.crossentropy import CrossEntropy
from network.progress import Progress
from network.utils import batches
from network.utils import print_batch_values


class Network:
    def __init__(self, loss, layers_config, wreg, wrt):
        self.loss = get_loss(loss)
        self.wreg = wreg
        self.wrt = wrt
        self.network = self.build_network(layers_config)


    def build_network(self, layers_config):
        layers = []

        """ add input layer """
        input_config = layers_config[0]
        input_layer = ConvInputLayer(1) if input_config['type'] == 'conv' else InputLayer(input_config['size'])
        layers.append(input_layer)

        """ add hidden layers """
        prev_layer = input_layer
        for i in range(1, len(layers_config) - 1):
            layer_config = layers_config[i]
            next_layer_config = layers_config[i + 1]

            # if convolutional layer
            if layer_config['type'] == 'conv':
                is_layer_1d_conv = layer_config['filter_shape'][0] == 1
                is_next_layer_dense = next_layer_config['type'] == 'dense'
                is_next_layer_output = next_layer_config['type'] == 'output'
                is_next_layer_1d_conv = not (is_next_layer_dense or is_next_layer_output) and \
                                        next_layer_config['filter_shape'][0] == 1

                flat_data = not is_layer_1d_conv and (is_next_layer_dense or is_next_layer_1d_conv)

                layer = ConvolutionalLayer(layer_config['filter_shape'],
                                           layer_config['num_filters'],
                                           layer_config['stride'], layer_config['mode'],
                                           prev_layer,
                                           activation=get_activation(layer_config['activation']),
                                           learning_rate=layer_config['learning_rate'],
                                           flatten_data=flat_data,
                                           flatten_channels=is_next_layer_dense or is_next_layer_output)

            # else dense layer
            else:
                is_next_layer_conv = next_layer_config['type'] == 'conv'
                layer = DenseLayer(layer_config['size'],
                                   prev_layer,
                                   activation=get_activation(layer_config['activation']),
                                   learning_rate=layer_config['learning_rate'],
                                   wreg=self.wreg,
                                   wrt=self.wrt if self.wrt is not None else None,
                                   wrap_output=is_next_layer_conv)
            layers.append(layer)
            prev_layer = layer

        """ add output layer """
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
        progress = Progress(epochs) if verbose == 1 else None
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

                if train:
                    if verbose == 1:
                        progress.batch_train_loss(loss)

                    if verbose == 2:
                        print_batch_values(network_inputs, outputs, targets, loss)

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

                if verbose == 1:
                    progress.val_loss(val_loss)
                    progress.print_epoch(epoch + 1)

        return loss_history_train, loss_history_val


    def visualize_kernels(self):
        """
        Visualizes kernels for all conolutional layers, if any.
        """
        for layer in self.network:
            if layer.type == 'conv':
                plot_kernel(layer.weights)


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
