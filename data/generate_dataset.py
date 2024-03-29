from time import sleep
from data.generator.generator import ImageGenerator
from data.generator.utils import split_into_data_and_targets
from data.generator.utils import visualize_image, save_image


def generate_dataset(config, visualize=False, save_examples=False):
    """
    Use this function to generate images with the image generator
    """

    # GENERATE DATASET #
    image_generator = ImageGenerator(size=config['image_size'],
                                     two_dim=config['two_dimensional'],
                                     centered=config['centered'],
                                     noise_rate=config['noise_rate'])

    print('\nGenerating images...')
    images_train, images_val, images_test = image_generator.generate_image_sets(config['number_of_images'],
                                                                                flatten=config['flatten'],
                                                                                share_train=config['share_train'],
                                                                                share_validate=config['share_validate'])

    print('Done generating images. Image set sizes: Train:', len(images_train),
          '- Validation:', len(images_val), '- Test:', len(images_test))

    if visualize:
        for image, _ in images_train:
            visualize_image(image)
            sleep(0.2)

    if save_examples:
        for i in range(8):  # save 8 first images as examples
            save_image(images_train[i][0], f'data/examples/fig-{i}.png')

    # SPLIT DATA INTO FEATURES AND TARGETS #
    train_data, train_targets = split_into_data_and_targets(images_train)
    val_data, val_targets = split_into_data_and_targets(images_val)
    test_data, test_targets = split_into_data_and_targets(images_test)

    return (train_data, train_targets), (val_data, val_targets), (test_data, test_targets)

