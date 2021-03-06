from time import sleep
from generator.generator import ImageGenerator
from generator.visualizer import visualize_image


generator = ImageGenerator(20, two_dim=False)

images_train, _, _ = generator.generate_image_sets(1000, flatten=False, share_train=1, share_validate=0)

all_images = []
for image, target in images_train:
    all_images.append(image)


visualize_image(all_images[:20])
print(generator.counter)
