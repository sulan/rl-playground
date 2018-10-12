import json

class ConfigParser:

    def __init__(self, config_path):
        self._options = dict()
        with open(config_path, mode = 'r') as f:
            self._options = json.load(f)
        if not isinstance(self._options, dict):
            raise TypeError('Configuration file should contain an object')

    def __getattr__(self, name):
        try:
            return self._options[name]
        except KeyError:
            raise AttributeError('No such option: "{}"'.format(name))

    def getOption(self, name, default = None):
        try:
            return self._options[name]
        except KeyError:
            return default
