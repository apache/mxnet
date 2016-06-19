import mxnet as mx

from mxnet.base import _LIB, check_call
from mxnet.base import c_array, mx_uint, mx_float, c_str
from mxnet.base import OptimizerHandle, OptimizerCreator
from mxnet.ndarray import NDArray, zeros, clip, sqrt
from mxnet.random import normal

@mx.optimizer.register
class speechSGD(mx.optimizer.Optimizer):
    """A very simple SGD optimizer with momentum and weight regularization.

    Parameters
    ----------
    learning_rate : float, optional
        learning_rate of SGD

    momentum : float, optional
       momentum value

    wd : float, optional
        L2 regularization coefficient add to all the weights

    rescale_grad : float, optional
        rescaling factor of gradient.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]

    param_idx2name : dict of string/int to float, optional
        special treat weight decay in parameter ends with bias, gamma, and beta
    """
    def __init__(self, momentum=0.0, **kwargs):
        super(speechSGD, self).__init__(**kwargs)
        self.momentum = momentum

    def create_state(self, index, weight):
        """Create additional optimizer state such as momentum.

        Parameters
        ----------
        weight : NDArray
            The weight data

        """
        if self.momentum == 0.0:
            return None
        else:
            return zeros(weight.shape, weight.context, dtype=weight.dtype)
    
    def _get_lr(self, index):
        """get learning rate for index.

        Parameters
        ----------
        index : int
            The index for weight

        Returns
        -------
        lr : float
            learning rate for this index
        """
        mom = 0.0
        if self.lr_scheduler is not None:
            (lr, mom) = self.lr_scheduler(self.num_update)
        else:
            lr = self.lr

        if index in self.lr_mult:
            lr *= self.lr_mult[index]
        elif index in self.idx2name:
            lr *= self.lr_mult.get(self.idx2name[index], 1.0)
        return lr, mom

    def update(self, index, weight, grad, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters

        weight : NDArray
            weight ndarray

        grad : NDArray
            grad ndarray

        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        (lr, momentum) = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        if state:
            mom = state
            mom[:] *= momentum
            mom[:] += -lr * (1.0 - momentum) * (grad + wd * weight)
            weight[:] += mom
        else:
            assert self.momentum == 0.0
            weight[:] += -lr * (grad + self.wd * weight)



