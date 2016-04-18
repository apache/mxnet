import sys
import numpy
import math
import logging
import theano
import theano.tensor as T

class NormParamEstimator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.x = T.matrix('x')
        self.logger = logging.getLogger(__name__)

    def zeroth_order_stats(self, x):
        return x.shape[0]

    def first_order_stats(self, x):
        return T.sum(x, axis=0)

    def second_order_stats(self, x):
        return T.sum(x**2, axis=0)

    def build_norm_estimation_functions(self, data_sets):
        (corpus_feats, _) = data_sets.get_shared()
        
        start_idx = T.lscalar('start_idx')  # index to a [mini]batch        
        end_idx = T.lscalar('end_idx')  # index to a [mini]batch        
        
        # gives a vector of 0's and 1's where the 0's are correct hypotheses
        norm_func = theano.function(inputs = [start_idx, end_idx],
                                    outputs = [ self.zeroth_order_stats(self.x),
                                                self.first_order_stats(self.x),
                                                self.second_order_stats(self.x) ],
                                    givens={self.x: corpus_feats[start_idx:end_idx]})
    
    
        return norm_func

    def estimate_norm_params(self, data_sets):
        gpu_norm_estimation_fn = self.build_norm_estimation_functions(data_sets)

        zeroth = 0
        first = None
        second = None

        total_feats = 0
        data_sets.initialize_read()
        while True:
            num_feats = data_sets.load_next_block()
            if num_feats < 0:
                break
            num_batches = int(math.ceil(float(num_feats) / self.batch_size))
            for batch_index in xrange(num_batches):
                start_idx = batch_index * self.batch_size
                end_idx = min(start_idx + self.batch_size, num_feats)
                
                stats = gpu_norm_estimation_fn(start_idx, end_idx)
        
                zeroth += stats[0]
                if first is None:
                    first = stats[1]
                    second = stats[2]
                else:
                    first += stats[1]
                    second += stats[2]

                """
                if first[1]/zeroth > 1000000:
                    sys.exit(1)
                    print start_idx, end_idx
                    arr = data_sets.shared_x.get_value()[start_idx:end_idx]
                    print "max", arr.max()
                    for i in xrange(arr.shape[0]):
                        if arr[i][1] > 100:
                            print ">>>>>>>>>>>>>>>>>>>>>>>>"
                        print i, arr[i][1]
                    sys.exit(1)
                """

            total_feats += num_feats
            self.logger.debug("feats: %d - mean[1] %f - first[0] %f - second[0] %f" % (total_feats, first[1]/zeroth, first[0], second[0]))

        mean = first/zeroth
        var = second/zeroth - mean**2
        inv_std = 1.0/numpy.sqrt(var)

        return [mean, inv_std]

