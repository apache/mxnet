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

from mxnet.ndarray import NDArray, topk, abs as NDabs
from mxnet.optimizer import SGD, register
import logging

log = 'Sparsity Update:\t'

@register
class SparseSGD(SGD):
    """The SGD optimizer with weight pruning.

    This class implements the optimizer described in the paper *DSD: Dense-Sparse-Dense Training for
    Deep Neural Networks*, available at https://arxiv.org/pdf/1607.04381.pdf

    The optimizer updates the weights the same way as done in SGD, but does the following
    preprocessing::

        if threshold given, all weights below the threshold in absolute value are pruned,
            mask    =   abs(weight) >= threshold
        if sparsity level given, the smallest (sparsity)% weights in absolute value are pruned
        (or the largest (100-sparsity)% weights in absolute value are used)
            mask    =   topk(abs(weight), ret_typ='mask', k=weight.size*(100-sparsity)/100)

        => mask[i,j]    =   {0 if weight[i,j] is pruned, 1 otherwise} (for a matrix representation)

        weight  =   weight  *   mask
        grad    =   grad    *   mask
        state   =   state   *   mask

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.SGD`.

    Parameters
    ----------
    pruning_switch_epoch : list of ints, optional
        The epochs at which there is a change in sparsity level (should be in ascending order).

    weight_sparsity : list of floats, optional
        The sparsity on the weights required on each iteration of sparse training.

    bias_sparsity : list of floats, optional
        The sparsity on the biases required on each iteration of sparse training.

    weight_threshold : list of floats, optional
        The absolute value threshold on the weights required on each iteration of sparse training.

    bias_threshold : list of floats, optional
        The absolute value threshold on the biases required on each iteration of sparse training.

    batches_per_epoch : int, optional
        The number of batches in each epoch.
        (The ceiling integer value of number_of_examples / batch_size)
    """
    def __init__(self, pruning_switch_epoch, batches_per_epoch,
                 weight_sparsity=None, bias_sparsity=None,
                 weight_threshold=None, bias_threshold=None, **kwargs):
        super(SparseSGD, self).__init__(**kwargs)

        self.masks = []
        self.masks_updated = False
        self.epoch = 0
        self.pruning_switch_epoch = pruning_switch_epoch
        self.batches_per_epoch = batches_per_epoch

        # get weight and bias sparsity percentages
        self.weight_sparsity = weight_sparsity
        self.bias_sparsity = bias_sparsity
        if weight_sparsity is not None:
            assert len(weight_sparsity) == len(bias_sparsity), \
                'weight_sparsity and bias_sparsity should have same length'
            assert len(weight_sparsity) == len(pruning_switch_epoch), \
                'pruning_switch_epoch and weight_sparsity should have same length'

        # get weight and bias sparsity thresholds
        self.weight_threshold = weight_threshold
        self.bias_threshold = bias_threshold
        if weight_threshold is not None:
            assert len(weight_threshold) == len(bias_threshold), \
                'weight_threshold and bias_threshold should have same length'
            assert len(weight_threshold) == len(pruning_switch_epoch), \
                'pruning_switch_epoch and weight_sparsity_threshold should have same length'

        # either percentages or thresholds must be given
        assert weight_sparsity is not None or weight_threshold is not None,\
            'weight_sparsity or weight_sparsity_threshold should be given'

    def update_masks(self, index, weight):
        """Updates the masks for sparse training.

        Parameters
        ----------
        index : int
            The index for weight.
        weight : NDArray
            The weight matrix.

        Returns
        -------
        boolean
            If the masks were changed
        """
        # determine number of updates without actually updating the count
        if index not in self._index_update_count:
            num_update = self.begin_num_update
        else:
            num_update = self._index_update_count[index]
        num_update += 1
        num_update = max(num_update, self.num_update)

        # calculate epoch
        epoch = int((num_update - 1) / self.batches_per_epoch) + 1

        # determine if masks need to be updated, and get corresponding parameters
        if index == 0:
            self.masks_updated = True
        if self.epoch != epoch:
            self.epoch = epoch
            if epoch == 1:
                self.masks_updated = False
                if self.weight_sparsity is not None:
                    logging.info(log + 'bias-sparsity={}, weight-sparsity={}'.format(self.bias_sparsity[0], self.weight_sparsity[0]))
                else:
                    logging.info(log + 'bias-threshold={}, weight-threshold={}'.format(self.bias_threshold[0], self.weight_threshold[0]))
            if self.pruning_switch_epoch[0] + 1 == epoch:
                self.masks_updated = False
                self.pruning_switch_epoch.pop(0)
                if self.weight_sparsity is not None:
                    self.weight_sparsity.pop(0)
                    self.bias_sparsity.pop(0)
                    logging.info(log + 'bias-sparsity={}, weight-sparsity={}'.format(self.bias_sparsity[0], self.weight_sparsity[0]))
                else:
                    self.weight_threshold.pop(0)
                    self.bias_threshold.pop(0)
                    logging.info(log + 'bias-threshold={}, weight-threshold={}'.format(self.bias_threshold[0], self.weight_threshold[0]))

        # update masks if needed
        if not self.masks_updated:
            # initialize masks
            if epoch == 1:
                self.masks.append(None)
            # if percentages are given
            if self.weight_sparsity is not None:
                if len(weight.shape) == 1:
                    sparsity = self.bias_sparsity[0]
                else:
                    sparsity = self.weight_sparsity[0]
                number_unpruned = int((100.0 - sparsity) * weight.size / 100.0)
                self.masks[index] = topk(NDabs(weight), axis=None, ret_typ='mask',
                                         k=number_unpruned)
            # if thresholds are given
            else:
                if len(weight.shape) == 1:
                    threshold = self.bias_threshold[0]
                else:
                    threshold = self.weight_threshold[0]
                self.masks[index] = NDabs(weight) >= threshold

        return not self.masks_updated

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))

        # preprocessing for pruning
        if self.update_masks(index, weight):
            weight[:] = weight * self.masks[index]
        grad[:] = grad * self.masks[index]
        if state is not None:
            state[:] = state * self.masks[index]

        super(SparseSGD, self).update(index, weight, grad, state)
