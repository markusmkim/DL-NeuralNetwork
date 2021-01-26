from ConfigReader.reader import ConfigReader


config_reader = ConfigReader(filepath='config.ini')

data = config_reader.get_data()

for key in data:
    print(key, ': ', data[key])
