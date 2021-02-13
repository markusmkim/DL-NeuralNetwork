import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


colors = ListedColormap(['whitesmoke', 'darkolivegreen'])


def visualize_image(matrix):
    plt.matshow(matrix, cmap=colors)
    plt.axis('off')
    # plt.savefig(f'fig-{index}.png', bbox_inches='tight')
    plt.show()


