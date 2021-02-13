from time import sleep
from config.reader import ConfigReader
from config.utils import print_config
from generator.generator import ImageGenerator
from generator.utils import split_into_data_and_targets
from generator.visualizer import visualize_image
from network.network import Network
from network.loss.visualizer import plot_loss_per_minibatch


# read config
config_reader = ConfigReader(filepath='config.ini')
config = config_reader.get_data()
print_config(config)


image_generator = ImageGenerator(size=15, centered=True, noise_rate=0.01)

print('Generating images')
images_train, images_val, images_test = image_generator.generate_image_sets(3000, flatten=True)

print('Image sets size: Train - Validation - Test')
print(len(images_train), len(images_val), len(images_test))

# split data into data features and targets
train_set, train_targets = split_into_data_and_targets(images_train)
val_set, val_targets = split_into_data_and_targets(images_val)
test_set, test_targets = split_into_data_and_targets(images_test)

# create and compile dense neural network model
model = Network(config['loss'], config['layers'])

# train model
train_loss_history, val_loss_history = model.fit(train_set, train_targets, val_set, val_targets,
                                                 batch_size=8, epochs=10)  # default batch size = 32

test_loss_history, _ = model.predict(test_set, test_targets)

print(len(train_loss_history))
print(len(val_loss_history))
print(len(test_loss_history))

plot_loss_per_minibatch(config['loss'], train_loss_history, val_loss_history, test_loss_history)

#for image_data in images_train:
 #   print(image_data[0], ':', image_data[1], '\n')


#for datarow in images_train:
#    visualize_image(datarow[0])
#    sleep(0.2)








"""
# 3 - 2 - 2 network from slides, with batch size = 3
# 2 batches
inputs = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 0],
])

targets = np.array([
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
])
"""