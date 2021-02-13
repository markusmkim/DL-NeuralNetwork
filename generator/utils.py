import numpy as np


# images = [data_array, target_array]
def split_into_data_and_targets(images):
    data = []  # features
    targets = []
    for image in images:
        data.append(image[0])
        targets.append(image[1])

    return np.array(data), np.array(targets)