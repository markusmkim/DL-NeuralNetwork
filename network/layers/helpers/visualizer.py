import numpy as np
import matplotlib.pyplot as plt


def plot_kernel(kernels, layer_index, save_kernel_image=False):
    fig, axs = plt.subplots(kernels.shape[1], kernels.shape[0])
    title = 'Kernels layer ' + str(layer_index) + ': ' + str(kernels.shape[1]) + ' input channels, ' + str(kernels.shape[0]) + ' output channels'
    fig.suptitle(title)

    if kernels.shape[0] > 1:
        for output_map_index in range(kernels.shape[0]):
            if len(axs.shape) > 1:
                for input_map_index in range(kernels.shape[1]):
                    kernel = kernels[output_map_index][input_map_index]
                    hinton(kernel, ax=axs[input_map_index, output_map_index])
            else:
                kernel = kernels[output_map_index][0]
                hinton(kernel, ax=axs[output_map_index])

    else:
        hinton(kernels[0][0])

    if save_kernel_image:
        plt.savefig(f'data/result/kernels-layer-{layer_index}', bbox_inches='tight')

    fig.show()


def hinton(matrix, max_weight=None, ax=None):
    """
    Draw Hinton diagram for visualizing a weight matrix.
    Source: https://matplotlib.org/stable/gallery/specialty_plots/hinton_demo.html
    """
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        if matrix.shape[0] == 1:
            x, y = y, x
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
