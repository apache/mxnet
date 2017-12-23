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
