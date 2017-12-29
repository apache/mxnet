# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=invalid-name
"""Learning rate scheduler."""

import math
import logging

class LearningRateScheduler(object):
    """Base class of learning rate scheduler."""
    def __init__(self):
        self.base_lr = 0.01

    def __call__(self, iteration):
        """
        Call to schedule current learning rate.

        Parameters
        ----------
        iteration: int
            Current iteration count.
        """
        raise NotImplementedError("must override this")


class FactorScheduler(LearningRateScheduler):
    """Reduce learning rate in factor.

    Parameters
    ----------
    step: int
        Schedule learning rate after every round.
    factor: float
        Reduce learning rate factor.
    """
    def __init__(self, step, factor=0.1):
        super(FactorScheduler, self).__init__()
        if step < 1:
            raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor >= 1.0:
            raise ValueError("Factor must be less than 1 to make lr reduce")
        self.step = step
        self.factor = factor
        self.old_lr = self.base_lr
        self.init = False

    def __call__(self, iteration):
        """
        Call to schedule current learning rate.

        Parameters
        ----------
        iteration: int
            Current iteration count.
        """

        if not self.init:
            self.init = True
            self.old_lr = self.base_lr
        lr = self.base_lr * math.pow(self.factor, int(iteration / self.step))
        if lr != self.old_lr:
            self.old_lr = lr
            logging.info("At Iteration [%d]: Swith to new learning rate %.5f",
                         iteration, lr)
        return lr
