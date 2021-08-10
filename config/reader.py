import configparser


class ConfigReader:
    def __init__(self, filepath):
        self.config = configparser.ConfigParser()
        self.config.read(filepath)

    def get_data(self):
        data = {
            'loss': self.config['GLOBALS']['loss'],
            'batch_size': int(self.config['GLOBALS']['batch_size']),
            'epochs': int(self.config['GLOBALS']['epochs']),
            'verbose': int(self.config['GLOBALS']['verbose']),
            'wreg': float(self.config['GLOBALS']['wreg']),
            'wrt': self.config['GLOBALS']['wrt'],
            'image_size': int(self.config['DATA']['image_size']),
            'two_dimensional': self.config['DATA']['two_dimensional'] == 'true',
            'centered': self.config['DATA']['centered'] == 'true',
            'noise_rate': float(self.config['DATA']['noise_rate']),
            'number_of_images': int(self.config['DATA']['number_of_images']),
            'flatten': self.config['DATA']['flatten'] == 'true',
            'share_train': float(self.config['DATA']['share_train']),
            'share_validate': float(self.config['DATA']['share_validate'])
        }
        layers = []

        for key in self.config['LAYERS']:
            params = str(self.config['LAYERS'][key]).split(' - ')

            """ input layer """
            if key == 'i':
                layer = {
                    'type': params[0],
                    'size': int(params[1]) if params[1] != 'none' else None,
                }
                layers.append(layer)
                continue

            """ output layer """
            if key == 'o':
                layer = {
                    'type': 'output',
                    'size': int(params[0]),
                    'activation': params[1],
                    'learning_rate': float(params[2])
                }
                layers.append(layer)
                continue

            """ hidden layers """
            layer_type = params[0]
            # convolutional layers
            if layer_type == 'conv':
                kernel_dimensions = [int(dimension) for dimension in params[1][1:-1].split(',')]
                kernel_shape = (kernel_dimensions[0], kernel_dimensions[1])
                layer = {
                    'type': params[0],
                    'filter_shape': kernel_shape,
                    'num_filters': int(params[2]),
                    'stride': int(params[3]),
                    'mode': params[4],
                    'activation': params[5],
                    'learning_rate': float(params[6])
                }
                layers.append(layer)
                continue

            # dense layers
            if layer_type == 'dense':
                layer = {
                    'type': params[0],
                    'size': int(params[1]),
                    'activation': params[2],
                    'learning_rate': float(params[3])
                }
                layers.append(layer)

        data['layers'] = layers
        return data
