from functools import wraps
from copy import deepcopy
import inspect
import torch.nn as nn


def serialize(init):
    parameters = list(inspect.signature(init).parameters)
    @wraps(init)
    def new_init(self, *args, **kwargs):
        params = deepcopy(kwargs)
        for pname, value in zip(parameters[1:], args):
            params[pname] = value

        config = {
            'class': get_classname(self.__class__),
            'params': dict()
        }
        specified_params = set(params.keys())

        for pname, param in get_default_params(self.__class__).items():
            if pname not in params:
                params[pname] = param.default

        for name, value in list(params.items()):
            param_type = 'builtin'
            if inspect.isclass(value):
                param_type = 'class'
                value = get_classname(value)

            config['params'][name] = {
                'type': param_type,
                'value': value,
                'specified': name in specified_params
            }

        setattr(self, '_config', config)
        init(self, *args, **kwargs)

    return new_init


def get_default_params(some_class):
    params = dict()
    for mclass in some_class.mro():
        if mclass is nn.Module or mclass is object:
            continue

        mclass_params = inspect.signature(mclass.__init__).parameters
        for pname, param in mclass_params.items():
            if param.default != param.empty and pname not in params:
                params[pname] = param

    return params


def get_classname(cls):
    module = cls.__module__
    name = cls.__qualname__
    if module is not None and module != "__builtin__":
        name = module + "." + name
    return name