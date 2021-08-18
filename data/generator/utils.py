import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def split_data_set(data, share_train, share_validate):
    train_size = math.floor(len(data)*share_train)
    val_size = len(data) - train_size if share_train + share_validate == 1 else math.floor(len(data)*share_validate)
    train = data[:train_size]
    val = data[train_size: train_size + val_size]
    test = data[train_size + val_size:]
    return train, val, test


def one_hot_encoder(target, num_classes):
    encoded_target = np.zeros(num_classes)
    # target is index
    encoded_target[target] = 1
    return encoded_target


# images = [data_array, target_array]
def split_into_data_and_targets(images):
    data = []  # features
    targets = []
    for image in images:
        data.append(image[0])
        targets.append(image[1])

    return np.array(data), np.array(targets)


def visualize_image(matrix):
    prepare_image(matrix)
    plt.show()


def save_image(matrix, path):
    prepare_image(matrix)
    plt.savefig(path, bbox_inches='tight')


def prepare_image(matrix):
    colors = ListedColormap(['whitesmoke', 'darkolivegreen'])
    plt.matshow(matrix, cmap=colors)
    plt.axis('off')