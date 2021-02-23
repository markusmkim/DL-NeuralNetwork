import math
import numpy as np


def batches(data, batch_size):
    if not batch_size:
        return None
    number_of_batches = math.ceil(len(data) / batch_size)
    return np.array_split(data, number_of_batches)


def print_batch_values(inputs, outputs, targets, loss):
    print('\nInputs')
    print(inputs)
    print('\nOutputs')
    print(outputs)
    print('\nTargets')
    print(targets)
    print('\nLoss: ', loss)
    print('=' * 100)
