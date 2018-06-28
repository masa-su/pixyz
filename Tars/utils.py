import numpy as np

_EPSILON = np.finfo(np.float32).eps


def set_epsilon(eps):
    global _EPSILON
    _EPSILON = eps


def epsilon():
    return _EPSILON


def get_dict_values(dicts, keys, return_dict=False):
    # Exp.
    # get_dict_values({"a":1,"b":2,"c":3}, ["b","d"])
    # >> [2]
    # get_dict_values(x, ["b","d"], True)
    # >> {'b': 2}

    values = [dicts[key] for key in keys if key in list(dicts.keys())]
    if return_dict:
        return dict(zip(keys, values))

    return values


def tolist(a):
    if type(a) is list:
        return a
    return [a]
