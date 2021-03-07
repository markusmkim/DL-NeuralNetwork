from time import sleep
from generator.generator import ImageGenerator
from generator.visualizer import visualize_image


def demo_2d():
    generator = ImageGenerator(20, centered=False, noise_rate=0.02)
    images_train, _, _ = generator.generate_image_sets(12, flatten=False, share_train=1, share_validate=0)

    for image, _ in images_train:
        visualize_image(image)
        sleep(0.2)


def demo_1d():
    generator = ImageGenerator(20, two_dim=False)
    images_train, _, _ = generator.generate_image_sets(20, flatten=False, share_train=1, share_validate=0)

    all_images = []
    for image, target in images_train:
        all_images.append(image)

    visualize_image(all_images)


demo_1d()
