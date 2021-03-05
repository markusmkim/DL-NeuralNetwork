from config.reader import ConfigReader
from config.utils import print_config
from generator.generator import ImageGenerator
from generator.utils import split_into_data_and_targets
from network.network import Network
from network.loss.visualizer import plot_loss_per_minibatch


# --- READ CONFIGURATIONS --- #
path_config_demo = 'config/config_demo.ini'
path_config_custom = 'config/config_custom.ini'
path_config_conv_test = 'config/conv_test.ini'
config_reader = ConfigReader(filepath=path_config_conv_test)
config = config_reader.get_data()
print('\nConfiguration summary:\n----------------------')
print_config(config)


# --- GENERATE IMAGE DATASET --- #
image_generator = ImageGenerator(size=config['image_size'],
                                 centered=config['centered'],
                                 noise_rate=config['noise_rate'])
print('\nGenerating images...')
images_train, images_val, images_test = image_generator.generate_image_sets(config['number_of_images'],
                                                                            flatten=config['flatten'],
                                                                            share_train=config['share_train'],
                                                                            share_validate=config['share_validate'])
print('Done generating images. Image set sizes: Train:', len(images_train),
      '- Validation:', len(images_val), '- Test:', len(images_test))


# --- SPLIT DATA INTO FEATURES AND TARGETS --- #
train_set, train_targets = split_into_data_and_targets(images_train)
val_set, val_targets = split_into_data_and_targets(images_val)
test_set, test_targets = split_into_data_and_targets(images_test)


# --- CLASSIFY IMAGES --- #
model = Network(config['loss'], config['layers'], config['wreg'], config['wrt'])

# train model
print('\nTraining model...')
train_loss_history, val_loss_history = model.fit(train_set, train_targets,
                                                 val_set, val_targets,
                                                 batch_size=config['batch_size'],
                                                 epochs=config['epochs'],
                                                 verbose=config['verbose'])

# visualize kernels if any
model.visualize_kernels()

# test model
print('Done training. Testing...')
test_loss_history, _ = model.predict(test_set, test_targets)
print('Done testing. Testing error:', test_loss_history[0])

# visualize learning
plot_loss_per_minibatch(config['loss'], train_loss_history, val_loss_history, test_loss_history)

