""" General utility functions. """


def filter_args(fun, kwargs):
    """
    Filters a kwargs dictionary with only the arguments that the function 'fun'
    accepts.
    """
    return {key: kwargs[key] for key, value in kwargs.items()
            if key in fun.__code__.co_varnames}
