import weakref

__all__ = ["Transformation", "ComposeTransform", "ExpTransform", "AffineTransform",
           "PowerTransform", "AbsTransform"]


class Transformation(object):
    r"""Abstract class for implementing invertible transformation
    with computable log  det jacobians
    
    Attributes
    ----------
    bijective : bool
        
    """
    bijective = False

    def __init__(self, F=None):
        self._inv = None
        self._F = F
        super(Transformation, self).__init__()

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, value):
        self._F = value

    @property
    def inv(self):
        inv = None
        if self._inv is not None:
            inv = self._inv()
        if inv is None:
            inv = _InverseTransformation(self)
            self._inv = weakref.ref(inv)
        return inv

    def __call__(self, x):
        return self._forward_compute(x)

    def _inv_call(self, y):
        return self._inverse_compute(y)

    def _forward_compute(self, x):
        raise NotImplementedError

    def _inverse_compute(self, x):
        raise NotImplementedError
    
    def log_det_jacobian(self, x, y):
        """
        Compute the value of log(|dy/dx|)
        """
        raise NotImplementedError


class _InverseTransformation(Transformation):
    """
    A private class representing the invert of `Transformation`,
    which should be accessed through `Transformation.inv` property.
    """
    def __init__(self, forward_transformation):
        super(_InverseTransformation, self).__init__()
        self._inv = forward_transformation

    @property
    def inv(self):
        return self._inv

    def __call__(self, x):
        return self._inv._inverse_compute(x)

    def log_det_jacobian(self, x, y):
        return -self._inv.log_det_jacobian(y, x)


class ComposeTransform(Transformation):
    def __init__(self, parts):
        super(ComposeTransform, self).__init__()
        self._parts = parts

    def _forward_compute(self, x):
        for t in self._parts:
            x = t(x)
        return x

    @property
    def F(self):
        # FIXME
        return self._parts[0].F

    @F.setter
    def F(self, value):
        for t in self._parts:
            t.F = value

    @property
    def inv(self):
        inv = None
        if self._inv is not None:
            inv = self._inv()
        if inv is None:
            inv = ComposeTransform([t.inv for t in reversed(self._parts)])
            self._inv = weakref.ref(inv)
            inv._inv = weakref.ref(self)
        return inv

    def log_det_jacobian(self, x, y):
        if not self._parts:
            return self.F.np.zeros_like(x)
        result = 0
        x_prime = None
        # FIXME: handle multivariate cases.
        for t in self._parts[:-1]:
            x_prime = t(x)
            result = result + t.log_det_jacobian(x, x_prime)
            x = x_prime
        result = result + self._parts[-1].log_det_jacobian(x, y)
        return result
    

class ExpTransform(Transformation):
    r"""
    Perform the exponential transform: y = exp{x}.
    """
    bijective = True

    def _forward_compute(self, x):
        return self.F.np.exp(x)

    def _inverse_compute(self, y):
        return self.F.np.log(y)

    def log_det_jacobian(self, x, y):
        return x


class AffineTransform(Transformation):
    r"""
    Perform pointwise affine transform: y = loc + scale * x.
    """
    bijective = True

    def __init__(self, loc, scale):
        super(AffineTransform, self).__init__()
        self._loc = loc
        self._scale = scale

    def _forward_compute(self, x):
        return self._loc + self._scale * x
    
    def _inverse_compute(self, y):
        return (y - self._loc) /  self._scale

    def log_det_jacobian(self, x, y):
        abs_fn = self.F.np.abs
        log_fn = self.F.np.log
        ones_fn = self.F.np.ones_like
        # FIXME: handle multivariate cases.
        return ones_fn(x) * log_fn(abs_fn(self._scale))


class PowerTransform(Transformation):
    bijective = True

    def __init__(self, exponent):
        super(PowerTransform, self).__init__()
        self._exponent = exponent

    def _forward_compute(self, x):
        return self.F.np.power(x, self._exponent)

    def _inverse_compute(self, y):
        return self.F.np.power(y, 1 / self._exponent)

    def log_det_jacobian(self, x, y):
        log_fn = self.F.np.log
        abs_fn = self.F.np.abs
        return log_fn(abs_fn(self._exponent * y / x))


class AbsTransform(Transformation):
    def _forward_compute(self, x):
        return self.F.np.abs(x)

    def _inverse_compute(self, y):
        return y