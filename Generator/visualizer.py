import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from Generator.generator import ImageGenerator
import numpy as np

colors = ListedColormap(['whitesmoke', 'darkolivegreen'])


def visualize_image(matrix):
    plt.matshow(matrix, cmap=colors)
    plt.axis('off')
    plt.show()


g = ImageGenerator(10)
zeros = np.zeros((10, 10))
image = g.draw_cross()

print(image)
visualize_image(image)
