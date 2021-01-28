import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from Generator.generator import ImageGenerator
from time import sleep


colors = ListedColormap(['whitesmoke', 'darkolivegreen'])


def visualize_image(matrix):
    plt.matshow(matrix, cmap=colors)
    plt.axis('off')
    # plt.savefig(f'fig-{index}.png', bbox_inches='tight')
    plt.show()


g = ImageGenerator(size=25, centered=False, noise_rate=0.02)

images_train, images_val, images_test = g.generate_image_sets(20)

print(len(images_train), len(images_val), len(images_test))

for image in images_train:
    visualize_image(image)
    sleep(0.2)
