"""
learning rate scheduler, which adaptive changes the learning rate based on the
progress
"""
import logging

class LRScheduler(object):
    """Base class of a learning rate scheduler"""
    def __init__(self):
        """
        base_lr : float
            the initial learning rate
        """
        self.base_lr = 0.01

    def __call__(self, num_update):
        """
        Call to schedule current learning rate

        The training progress is presented by `num_update`, which can be roughly
        viewed as the number of minibatches executed so far. Its value is
        non-decreasing, and increases at most by one.

        The exact value is the upper bound of the number of updates applied to
        a weight/index

        See more details in https://github.com/dmlc/mxnet/issues/625

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """
        raise NotImplementedError("must override this")

class FactorScheduler(LRScheduler):
    """Reduce learning rate in factor

    Assume the weight has been updated by n times, then the learning rate will
    be

    base_lr * factor^(floor(n/step))

    Parameters
    ----------
    step: int
        schedule learning rate after n updates
    factor: float
        the factor for reducing the learning rate
    """
    def __init__(self, step, factor=1):
        super(FactorScheduler, self).__init__()
        if step < 1:
            raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor >= 1.0:
            raise ValueError("Factor must be less than 1 to make lr reduce")
        self.step = step
        self.factor = factor
        self.count = 0

    def __call__(self, num_update):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """

        if num_update > self.count + self.step:
            self.count += self.step
            self.base_lr *= self.factor
            logging.info("Update[%d]: Change learning rate to %0.5e",
                         num_update, self.base_lr)
        return self.base_lr
