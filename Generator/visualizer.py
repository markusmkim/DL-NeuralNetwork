import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from Generator.generator import ImageGenerator
import numpy as np
from time import sleep


colors = ListedColormap(['whitesmoke', 'darkolivegreen'])


def visualize_image(matrix):
    plt.matshow(matrix, cmap=colors)
    plt.axis('off')
    plt.show()


g = ImageGenerator(size=10, centered=False)


for i in range(25):
    image = g.draw_cross()

    visualize_image(image)

    sleep(0.2)
