# This is just a sanity check of the bucketing API
# with MNIST data. We will be using a bucket of different
# models, except that they all look the same. In practice,
# each model in the bucket will be different. For example,
# they could corresponds to sequence models of different
# length.

import numpy as np
import mxnet as mx
import train_mnist

class BucketBatch(object):
    def __init__(self, iter, batch):
        self.iter = iter
        self.batch = batch

        # we just pick a random bucket for each
        # batch as all the buckets are trivially
        # the same
        self.bucket_key = np.random.choice(iter.buckets)

    @property
    def provide_data(self):
        return self.iter.provide_data

    @property
    def provide_label(self):
        return self.iter.provide_label

    @property
    def data(self):
        return self.batch.data

    @property
    def label(self):
        return self.batch.label


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
            bucket_batch = BucketBatch(self, batch)

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

    buckets = ['foo', 'bar', 'baz']
    def symbol_generator(key):
        return net # all the symbols are the same for all bucket entries

    train_mnist.train_model.fit(args, net, get_iterator(buckets),
            sym_gen=symbol_generator)
