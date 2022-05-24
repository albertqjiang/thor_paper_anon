"""Parse and return mrunner gin-config and set-up Neptune.

This is copied from alpacka (with removed ray setup).
"""
import datetime
import os
import pickle

import cloudpickle
import neptune.new as neptune


def get_configuration(spec_path):
    """Get mrunner experiment specification and gin-config overrides."""
    try:
        with open(spec_path, 'rb') as f:
            specification = cloudpickle.load(f)
    # except pickle.UnpicklingError:
    except:  # TODO
        with open(spec_path) as f:
            vars_ = {'script': os.path.basename(spec_path)}
            exec(f.read(), vars_)  # pylint: disable=exec-used
            specification = vars_['experiments_list'][0].to_dict()
            print('NOTE: Only the first experiment from the list will be run!')

    parameters = specification['parameters']
    gin_bindings = []
    for key, value in parameters.items():
        if isinstance(value, str) and not (value[0] == '@' or value[0] == '%'):
            binding = f'{key} = "{value}"'
        else:
            binding = f'{key} = {value}'
        gin_bindings.append(binding)

    return specification, gin_bindings


class NeptuneLogger:
    """Logs to Neptune."""

    def __init__(self, neptune_run):
        """Initialize NeptuneLogger with the Neptune experiment."""
        self._experiment = neptune_run

    def log_scalar(self, name, step, value):
        """Logs a scalar to Neptune."""
        self._experiment[name].log(step=step, value=value)

    def log_image(self, name, step, img):
        """Logs an image to Neptune."""
        self._experiment['name'].upload(img)

    def log_property(self, name, value):
        self._experiment[name] = value

    def log_dict_as_scalars(self, step, scalars_dict):
        for name, value in scalars_dict.items():
            self.log_scalar(name, step, value)

    def log_text(self, name, content, show_on_screen):
        """Logs a text to Neptune."""
        del show_on_screen
        self._experiment[name].log(content)


class NeptuneAPITokenException(Exception):
    def __init__(self):
        super().__init__('NEPTUNE_API_TOKEN environment variable is not set!')


def configure_neptune(specification):
    """Configures the Neptune experiment, then returns the Neptune logger."""
    if 'NEPTUNE_API_TOKEN' not in os.environ:
        raise NeptuneAPITokenException()

    git_info = specification.get('git_info', None)
    if git_info:
        git_info.commit_date = datetime.datetime.now()

    neptune_run = neptune.init(specification['project'],
                               source_files=['*.py', '*/*.py'])
    # Set pwd property with path to experiment.
    properties = {'pwd': os.getcwd(),
                  'random_name': specification['random_name'],
                  'unique_name': specification['unique_name']
                  }
    neptune_run['properties'] = properties
    neptune_run['sys/tags'].add(specification['tags'])
    neptune_run['sys/name'] = specification['name']
    neptune_run['parameters'] = specification['parameters']

    return NeptuneLogger(neptune_run)
