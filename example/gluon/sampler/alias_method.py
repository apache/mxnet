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

# coding: utf-8
# pylint: skip-file
import mxnet.ndarray as nd
import numpy as np
from mxnet.test_utils import verify_generator


class AliasMethodSampler(object):
    """ The Alias Method: Efficient Sampling with Many Discrete Outcomes.
    Can be use in NCELoss.

    Parameters
    ----------
    K : int
        Number of events.
    probs : array
        Probability of each events, corresponds to K.

    References
    -----------
        https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """

    def __init__(self, F, K, probs):
        if K != len(probs):
            raise ValueError("K should be equal to len(probs). K:%d, len(probs):%d" % (K, len(probs)))
        self.K = K
        self.prob = F.zeros(K)
        self.alias = F.zeros(K).astype('int32')

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller + larger:
            self.prob[last_one] = 1

    def draw(self, F, n):
        """Draw N samples from multinomial
        """
        samples = F.zeros(n, dtype='int32')

        kk = F.floor(F.random.uniform(0, self.K, n)).astype('int32')
        rand = F.random.uniform(0, 1, n)

        prob = self.prob[kk]
        alias = self.alias[kk]

        for i in xrange(n):
            if rand[i] < prob[i]:
                samples[i] = kk[i]
            else:
                samples[i] = alias[i]
        return samples


def speed():
    # use numpy
    import time
    import numpy.random as npr

    K = 500
    N = 10000

    # Get a random probability vector.
    probs = npr.dirichlet(np.ones(K), 1).ravel()

    # Construct the table.
    elaps = []
    F = nd
    alias_method_sampler = AliasMethodSampler(F, K, probs)
    for i in range(100):
        start = time.time()
        X = alias_method_sampler.draw(F, N)
        time_elaps = time.time() - start
        elaps.append(time_elaps)
    print(
        'Use NDArray. Avg:%.2f ms, Min:%.2f ms, Max:%.2f ms, Std:%.2f ms'
        % (np.average(elaps), np.min(elaps), np.max(elaps), np.std(elaps)))

    # Construct the table.
    elaps = []
    F = nd
    alias_method_sampler = AliasMethodSampler(F, K, probs)
    for i in range(100):
        start = time.time()
        X = alias_method_sampler.draw(F, N)
        time_elaps = time.time() - start
        elaps.append(time_elaps)
    print(
        'Use Numpy. Avg:%.2f ms, Min:%.2f ms, Max:%.2f ms, Std:%.2f ms'
        % (np.average(elaps), np.min(elaps), np.max(elaps), np.std(elaps)))


def chi_square_test():
    probs = [0.1, 0.2, 0.3, 0.05, 0.15, 0.2]
    buckets = list(range(6))

    F = np
    alias_method_sampler = AliasMethodSampler(F, len(probs), probs)

    generator_mx = lambda x: alias_method_sampler.draw(F, x)
    verify_generator(generator_mx, buckets, probs)

    generator_mx_same_seed = \
        lambda x: np.concatenate(
            [alias_method_sampler.draw(F, x // 10) for _ in range(10)]
        )
    verify_generator(generator=generator_mx_same_seed, buckets=buckets, probs=probs)


if __name__ == '__main__':
    speed()
    chi_square_test()

