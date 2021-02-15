from time import sleep
from generator.generator import ImageGenerator
from generator.visualizer import visualize_image


generator = ImageGenerator(30, centered=False, noise_rate=0.02)

images_train, _, _ = generator.generate_image_sets(12, flatten=False, share_train=1, share_validate=0)


for image, _ in images_train:
    visualize_image(image)
    sleep(0.2)
