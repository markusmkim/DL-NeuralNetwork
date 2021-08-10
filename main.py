from config.reader import ConfigReader
from config.utils import print_config
from network.network import Network
from network.loss.visualizer import plot_loss_per_minibatch
from generate_images import generate_images


# READ CONFIGURATION
config_reader = ConfigReader(filepath='config/config.ini')
config = config_reader.get_data()
print('\nConfiguration summary:\n----------------------')
print_config(config)


# DATA
(train_data, train_targets), (val_data, val_targets), (test_data, test_targets) = generate_images(config)


# build model
model = Network(config['loss'], config['layers'], config['wreg'], config['wrt'])


# train model
print('\nTraining model...')
train_loss_history, val_loss_history = model.fit(train_data, train_targets,
                                                 val_data, val_targets,
                                                 batch_size=config['batch_size'],
                                                 epochs=config['epochs'],
                                                 verbose=config['verbose'])

# visualize kernels if any
model.visualize_kernels()

# test model
print('Done training. Testing...')
test_loss_history, _ = model.predict(test_data, test_targets)
print('Done testing. Testing error:', test_loss_history[0])

# visualize learning
plot_loss_per_minibatch(config['loss'], train_loss_history, val_loss_history, test_loss_history)

