_EPSILON = 1e-7


def set_epsilon(eps):
    global _EPSILON
    _EPSILON = eps


def epsilon():
    return _EPSILON


def get_dict_values(dicts, keys, return_dict=False):
    """Get values from `dicts` specified by `keys`.

    When `return_dict` is True, return values are in dictionary format.

    Parameters
    ----------
    dicts : dict

    keys : list

    return_dict : bool

    Returns
    -------
    dict or list

    Examples
    --------
    >>> get_dict_values({"a":1,"b":2,"c":3}, ["b","d"])
    [2]
    >>> get_dict_values(x, ["b","d"], True)
    {'b': 2}
    """
    new_dicts = dict((key, dicts[key]) for key in keys if key in list(dicts.keys()))
    if return_dict is False:
        return list(new_dicts.values())

    return new_dicts


def delete_dict_values(dicts, keys):
    """Delete values from `dicts` specified by `keys`.

    Parameters
    ----------
    dicts : dict

    keys : list

    Returns
    -------
    new_dicts : dict

    Examples
    --------
    >>> get_dict_values({"a":1,"b":2,"c":3}, ["b","d"])
    {'a': 1, 'c': 3}
    """
    new_dicts = dict((key, value) for key, value in dicts.items() if key not in keys)
    return new_dicts


def detach_dict(dicts):
    """Detach all values in `dicts`.

    Parameters
    ----------
    dicts : dict

    Returns
    -------
    dict

    """
    return {k: v.detach() for k, v in dicts.items()}


def replace_dict_keys(dicts, replace_list_dict):
    """ Replace values in `dicts` according to `replace_list_dict`.

    Parameters
    ----------
    dicts : dict
    replace_list_dict : dict

    Returns
    -------
    replaced_dicts : dict

    Examples
    --------
    >>> replace_dict_keys({"a":1,"b":2,"c":3}, {"a":x,"b":y})
    {'x' : 1, 'y': 2, 'c' : 3}
    >>> replace_dict_keys({"a":1,"b":2,"c":3}, {"a":x,"e":y})  # keys of `replace_list_dict`
    {'x' : 1, 'b': 2, 'c' : 3}
    """
    replaced_dicts = dict({(replace_list_dict[key], value) if key in list(replace_list_dict.keys())
                           else (key, value) for key, value in dicts.items()})

    return replaced_dicts


def tolist(a):
    """Convert a given input to the dictionary format.

    Parameters
    ----------
    a : list or other

    Returns
    -------
    list

    """
    if type(a) is list:
        return a
    return [a]
