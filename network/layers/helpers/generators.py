import numpy as np
from math import ceil


def generate_maps(input_map, kernel, stride, mode):
    """
    Used by the convolutional layer.
    :param input_map: input map to transform
    :param kernel: kernel to use in transformation
    :param stride: step size of kernel
    :param mode: transformation mode - 'full' | 'same' | 'valid'
    :return: output map skeleton filled with zeros, padded input map
    """
    if mode == 'same':
        """
        if strides == 1:
            # assume kernel dimensions x = y, else need x_padding and y_padding
            padding_size = len(kernel) // 2
            # print(padding_size)
            padded_map = self.apply_padding(input_map, padding_size)
            return np.zeros(input_map.shape), padded_map
        """
        # number of kernel transformations in each direction/dimension
        steps_row = input_map.shape[0]
        steps_col = input_map.shape[1]

        # number of zero-pads to be added to each dimension
        pad_row = ceil(((len(kernel)) + ((steps_row - 1) * stride) - input_map.shape[0]) / 2)
        pad_col = ceil(((len(kernel)) + ((steps_col - 1) * stride) - input_map.shape[1]) / 2)

        if kernel.shape[0] == 1:
            # data and conv layer is one dimensonal
            pad_row = 0

        padding_size = ((pad_row, pad_row), (pad_col, pad_col))     # specify zero-padding dimensions
        padded_map = apply_padding(input_map, padding_size)         # pad input map

        return np.zeros(input_map.shape), padded_map, pad_row, pad_col


    if mode == 'full':
        """
        Create padded input map first, then use padded map to calculate output map shape:
        """
        padding_size = len(kernel) - 1                              # specify zero-padding dimensions
        padded_map = apply_padding(input_map, padding_size)         # pad input map

        # number of kernel transformations in each direction/dimension, with step size = strides
        steps_row = 1 + ((padded_map.shape[0] - kernel.shape[0]) // stride)
        steps_col = 1 + ((padded_map.shape[1] - kernel.shape[1]) // stride)

        output_map_shape = (steps_row, steps_col)  # output map shape given by steps

        return np.zeros(output_map_shape), padded_map, padding_size, padding_size


    if mode == 'valid':
        """
        Create output map shape first, then use output map shape to calculate zero-pads (opposite of full):
        """
        # number of kernel transformations in each direction/dimension, with step size = strides
        steps_row = 1 + ceil((input_map.shape[0] - kernel.shape[0]) / stride)
        steps_col = 1 + ceil((input_map.shape[1] - kernel.shape[1]) / stride)

        output_map_shape = (steps_row, steps_col)  # output map shape given by steps

        # number of zero-pads to be added to THE END of each dimension
        pad_row_end = ((steps_row - 1) * stride) + kernel.shape[0] - input_map.shape[0]
        pad_col_end = ((steps_col - 1) * stride) + kernel.shape[1] - input_map.shape[1]

        padding_size = ((0, pad_row_end), (0, pad_col_end))         # specify zero-padding dimensions
        padded_map = apply_padding(input_map, padding_size)         # pad input map

        return np.zeros(output_map_shape), padded_map, 0, 0


    print('Invalid mode')
    return None


def apply_padding(input_map, padding_size):
    return np.pad(input_map,
                  padding_size,
                  'constant',
                  constant_values=0)


"""
steps_row = 1 + ceil((input_map.shape[0] - kernel.shape[0]) / strides) 
    print(steps_row)
    steps_col = 1 + ceil((input_map.shape[1] - kernel.shape[1]) / strides) 
    output_map_shape = (steps_row, steps_col)

    pad_row = ((steps_row - 1) * strides) + kernel.shape[0] - input_map.shape[0]
    pad_col = ((steps_col - 1) * strides) + kernel.shape[1] - input_map.shape[1]

    padding_size = ((0, pad_row), (0, pad_col))
    padded_map = apply_padding(input_map, padding_size)

    output_map_shape = (steps_row, steps_col)
    return np.zeros(output_map_shape), padded_map
"""