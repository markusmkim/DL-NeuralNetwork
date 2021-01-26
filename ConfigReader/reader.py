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
            'input_layer_size': int(self.config['INPUT LAYER']['size']),
            'output_activation': self.config['OUTPUT LAYER']['activation'],
            'data': self.config['DATA']['data'],
        }
        hidden_layers = []

        for key in self.config['HIDDEN LAYERS']:

            params = str(self.config['HIDDEN LAYERS'][key]).split(' - ')

            if len(params) < 3:
                print('Config error, hidden layers missing values')
                continue

            layer = {
                'size': int(params[0]),
                'activation': params[1],
                'learning_rate': float(params[2])
            }
            hidden_layers.append(layer)

        data['hidden layers'] = hidden_layers

        return data
