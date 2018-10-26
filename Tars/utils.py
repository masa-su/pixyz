_EPSILON = 1e-7


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
    new_dicts = dict((key, dicts[key]) for key in keys if key in list(dicts.keys()))
    if return_dict is False:
        return list(new_dicts.values())

    return new_dicts


def delete_dict_values(dicts, keys):
    # Exp.
    # get_dict_values({"a":1,"b":2,"c":3}, ["b","d"])
    # >> {'a': 1, 'c': 3}
    new_dicts = dict((key, value) for key, value in dicts.items() if key not in keys)
    return new_dicts


def replace_dict_keys(dicts, replace_list_dict):
    replaced_dicts = dict({(replace_list_dict[key], value) if key in list(replace_list_dict.keys())
                           else (key, value) for key, value in dicts.items()})

    return replaced_dicts


def tolist(a):
    if type(a) is list:
        return a
    return [a]
