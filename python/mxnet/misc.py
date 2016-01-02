"""helper functions for mxnet"""
# pylint: disable=exec-used
import inspect
import pickle

def dumps(obj, protocol=0):
    """pickle module, class, method, function

    Parameters:
    -----------
    obj: module, class, method, function
        object to be pickled

    protocol: int, optional
        pickle protocol

    Returns:
    -------
        pickled list with obj source and object
    """
    src = "".join(inspect.getsourcelines(type(obj))[0])
    pickled_obj = pickle.dumps(obj, protocol)
    return pickle.dumps([src, pickled_obj])

def loads(obj):
    """load pickled module, class, method, function

    Parameters:
    ----------
    obj: pickled list
        pickled list of obj source and obj

    Returns:
    -------
        class/function/method object
    """
    src, pickled_obj = pickle.loads(obj)
    exec(src)
    return pickle.loads(pickled_obj)
