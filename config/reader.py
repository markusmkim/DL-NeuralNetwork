import configparser


class ConfigReader:
    def __init__(self, filepath):
        self.config = configparser.ConfigParser()
        self.config.read(filepath)

    def get_data(self):
        data = {
            'loss': self.config['GLOBALS']['loss'],
            'wreg': float(self.config['GLOBALS']['wreg']),
            'wrt': self.config['GLOBALS']['wrt'],
            'data': self.config['DATA']['data'],
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
