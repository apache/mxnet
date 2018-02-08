import mxnet as mx
import numpy as np

class AliasMethod(object):
    def __init__(self, weights):
        self.N = weights.size
        total_weights = weights.sum()
        self.prob = (weights * self.N / total_weights).asnumpy().tolist()
        self.alias = [0] * self.N

        # sort the data into the outcomes with probabilities
        # that are high and low than 1/N.
        low = []
        high = []
        for i in range(self.N):
            if self.prob[i] < 1.0:
                low.append(i)
            else:
                high.append(i)

        # pair low with high
        while len(low) > 0 and len(high) > 0:
            l = low.pop()
            h = high.pop()

            self.alias[l] = h
            self.prob[h] = self.prob[h] - (1.0 - self.prob[l])

            if self.prob[h] < 1.0:
                low.append(h)
            else:
                high.append(h)

        for i in low + high:
            self.prob[i] = 1
            self.alias[i] = i

        # convert to ndarrays
        self.prob = mx.nd.array(self.prob)
        self.alias = mx.nd.array(self.alias)

    def draw(self, k):
        ''' Draw k samples from the distribution '''
        idx = mx.nd.array(np.random.randint(0, self.N, size=k))
        prob = self.prob[idx]
        alias = self.alias[idx]
        where = mx.nd.random.uniform(shape=k) < prob
        hit = idx * where
        alt = alias * (1 - where)
        return hit + alt

class MXLogUniformSampler(object):
    def __init__(self, n):
        self.range = n
        self.log_range = np.log(n + 1)
        classes = mx.nd.arange(0, n, dtype='float64')
        self.prob = ((classes + 2.0) / (classes + 1.0)).log() / self.log_range

    def sample_unique_avoid(self, k, labels):
        import math
        num_tries = 0
        labels_list = labels.asnumpy().reshape((-1,)).tolist()
        out_set = set()
        for l in labels_list:
            out_set.add(l)

        out_list = []
        while len(out_list) < k:
            rand = np.random.uniform(low=0, high=self.log_range)
            idx = np.rint(np.exp(rand)) % self.range
            num_tries += 1
            if idx not in out_set:
                out_set.add(idx)
                out_list.append(idx)
        return mx.nd.array(out_list), num_tries

    def sample(self, k):
        # TODO default dtype ?
        rand = mx.nd.random.uniform(0, self.log_range, shape=(k,), dtype='float64')
        samples = (rand.exp() - 1).astype('int64')
        samples = samples % self.range
        return samples.astype('float32')

    def probability(self, classes):
        # TODO scale by batch size?
        return self.prob[classes].astype('float32')

    def probability_avoid(self, classes, num_tries):
        return -mx.nd.expm1(num_tries * mx.nd.log1p(-self.prob[classes]))

def test_log_uniform():
    def check_prob():
        n = 1000000
        classes = 100
        sampler = MXLogUniformSampler(n)
        while classes < n:
            ratio = sampler.probability(classes) / sampler.probability(classes / 2)
            mx.test_utils.assert_almost_equal(ratio.asscalar(), 0.5, rtol=1e-1)
            classes *= 2
    def check_sum():
        n = 10
        sampler = MXLogUniformSampler(n)
        classes = mx.nd.arange(0, n)
        probs = sampler.probability(classes)
        mx.test_utils.assert_almost_equal(probs.sum().asscalar(), 1, rtol=1e-4)
    check_prob()
    check_sum()
    print('pass')

#test_log_uniform()
