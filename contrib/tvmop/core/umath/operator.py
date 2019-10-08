import tvm

__all__ = ['equal', 'not_equal', 'greater', 'less', 'greater_equal', 'less_equal']


def equal(a, b):
    return a == b


def not_equal(a, b):
    return a != b


def greater(a, b):
    return a > b


def less(a, b):
    return a < b


def greater_equal(a, b):
    return a >= b


def less_equal(a, b):
    return a <= b
