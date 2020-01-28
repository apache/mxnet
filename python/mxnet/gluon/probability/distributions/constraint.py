__all__ = ["Constraint", "Real", "Boolean", "Interval"]

class Constraint(object):
    """Base class for constraints.

    A constraint object represents a region over which a variable
    is valid.
    
    Parameters
    ----------
    F : ndarry or symbol
        Running mode parameter.
    """

    def __init__(self, F):
        self.F = F

    def check(self, value):
        """Check if `value` satisfies the constraint,
        return the origin value if valid,
        raise `ValueError` with given message otherwise.
        
        Parameters
        ----------
        value : Tensor
            Input tensor to be checked.
        """
        raise NotImplementedError

    @property
    def _check_func(self):
        return self.F.npx.constraint_check


class Real(Constraint):
    """
    Constrain to be a real number. (exclude `np.nan`)
    """
    def check(self, value):
        err_msg = "Constraint violated: {} should be a real tensor".format(value)
        condition = (value == value)
        _value = self._check_func(condition, err_msg) * value
        return _value

    


class Boolean(Constraint):
    """
    Constrain to `{0, 1}`.
    """
    def check(self, value):
        err_msg = "Constraint violated: {} should be either 0 or 1.".format(value)
        # FIXME: replace bitwise_or with logical_or instead
        condition = self.F.np.bitwise_or(value == 0, value == 1)
        _value = self._check_func(condition, err_msg) * value
        return _value


class Interval(Constraint):
    """
    Constrain to a real interval `[lower_bound, upper_bound]`
    """
    def __init__(self, F, lower_bound, upper_bound):
        super(Interval, self).__init__(F)
        self._low = lower_bound
        self._up = upper_bound

    def check(self, value):
        err_msg = "Constraint violated: {} should be between {} and {}.".format(
                    value, self._low, self._up)
        # FIXME: replace bitwise_and with logical_and
        condition = self.F.np.bitwise_and(value > self._low, value < self._up)
        _value = self._check_func(condition, err_msg) * value
        return _value


