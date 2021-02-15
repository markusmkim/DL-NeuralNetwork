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
            'verbose': self.config['GLOBALS']['verbose'] == 'true',
            'wreg': float(self.config['GLOBALS']['wreg']),
            'wrt': self.config['GLOBALS']['wrt'],
            'image_size': int(self.config['DATA']['image_size']),
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

            if len(params) < 3:
                print('Config error, hidden layers missing values')
                continue

            layer = {
                'size': int(params[0]),
                'activation': params[1],
                'learning_rate': None if params[2] == 'none' else float(params[2])
            }
            layers.append(layer)

        data['layers'] = layers

        return data
