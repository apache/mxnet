# pylint: disable=invalid-name, logging-not-lazy
"""learning rate scheduler"""

import math
import logging
class Factor(object):
    """Reduce learning rate in factor

    Parameters
    ----------
    base_lr: float
        learning rate at start time
    step: int
        schedule learning rate after every step batches
    factor: float
        reduce learning rate factor
    batch_per_round: int
        how many batches per round, must set when continue training
    """
    def __init__(self, base_lr, step, factor=0.1, batch_per_round=1):
        self.base_lr = base_lr
        self.step = step
        self.factor = factor
        self.old_lr = base_lr
        self.batch_per_round = batch_per_round
        self.epoch = 0
        self.init = False

    def __call__(self, optimizer, nbatch, iteration):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        optimizer: Optimizer
            Optimizer which contains learning rate field
        nbatch: int
            Current batch count
        iteration: int
            Current iteration count
        """

        if self.init == False:
            self.init = True
            self.epoch = max(self.epoch, iteration * self.batch_per_round + nbatch)
        self.epoch += 1
        lr = self.base_lr * math.pow(self.factor, int(self.epoch / self.step))
        optimizer.learning_rate = lr
        if lr != self.old_lr:
            self.old_lr = lr
            logging.info("At Iteration [%d], Batch [%d]: Swith to new learning rate %.5f" \
                    % (iteration, nbatch, lr))


