# This is just a sanity check of the bucketing API
# with MNIST data. We will be using a bucket of different
# models, except that they all look the same. In practice,
# each model in the bucket will be different. For example,
# they could corresponds to sequence models of different
# length.
#
# The key for each bucket is an integer. We will use
# different batch-size for each different bucket, by
# simply duplicating the training data k times, where
# k is the bucket key.

from copy import deepcopy

import numpy as np
import mxnet as mx
import train_mnist

class BucketIter(mx.io.DataIter):
    def __init__(self, data_iter, buckets):
        self.data_iter = data_iter
        self.buckets = buckets
        self.default_bucket_key = buckets[0]
        self.stats = np.zeros(len(buckets))

    @property
    def provide_data(self):
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        return self.data_iter.provide_label

    @property
    def batch_size(self):
        return self.data_iter.batch_size

    def reset(self):
        self.data_iter.reset()

    def __iter__(self):
        for i, batch in enumerate(self.data_iter):
            bucket_batch = batch
            bucket_batch.bucket_key = np.random.choice(self.buckets)
            bucket_batch.provide_data = deepcopy(self.provide_data)
            bucket_batch.provide_label = deepcopy(self.provide_label)

            if bucket_batch.bucket_key > 1:
                # change batch-size by duplicating
                def modify(s):
                    s = list(s)
                    s[0] = s[0] * bucket_batch.bucket_key
                    return tuple(s)
                bucket_batch.provide_data = \
                        [(k,modify(s)) for k,s in bucket_batch.provide_data]
                bucket_batch.provide_label = \
                        [(k,modify(s)) for k,s in bucket_batch.provide_label]

                bucket_batch.data = [
                        mx.nd.array(np.vstack([x.asnumpy() for i in range(bucket_batch.bucket_key)]))
                        for x in bucket_batch.data]
                bucket_batch.label = [
                        mx.nd.array(np.hstack([y.asnumpy() for i in range(bucket_batch.bucket_key)]))
                        for y in bucket_batch.label]

            # accumulate statistics for debugging
            self.stats[self.buckets.index(bucket_batch.bucket_key)] += 1

            yield bucket_batch

        print("===================")
        for b, c in zip(self.buckets, self.stats):
            print("%6s : %d" % (b, c))
        print("")


def get_iterator(buckets):
    data_shape = (784, )
    impl = train_mnist.get_iterator(data_shape)
    def get_iterator_impl(args, kv):
        train, val = impl(args, kv)
        return (BucketIter(train, buckets), BucketIter(val, buckets))
    return get_iterator_impl

if __name__ == '__main__':
    args = train_mnist.parse_args()

    args.network = 'mlp'
    net = train_mnist.get_mlp()

    buckets = [1, 2, 3]
    def symbol_generator(key):
        return net # all the symbols are the same for all bucket entries

    train_mnist.train_model.fit(args, symbol_generator, get_iterator(buckets))
