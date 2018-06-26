import math
import os
import numpy as np
import mxnet as mx
import log_uniform

class LogUniformSampler():
    def __init__(self, range_max, num_sampled):
        self.range_max = range_max
        self.num_sampled = num_sampled
        self.sampler = log_uniform.LogUniformSampler(range_max)

    def prob_helper(self, num_tries, num_sampled, prob):
        if num_tries == num_sampled:
            return prob * num_sampled
        return (num_tries * (-prob).log1p()).expm1() * -1

    def draw(self, true_classes):
        from mxnet import ndarray
        range_max = self.range_max
        num_sampled = self.num_sampled
        ctx = true_classes.context
        log_range = math.log(range_max + 1)
        num_tries = 0
        true_classes = true_classes.reshape((-1,))
        sampled_classes, num_tries = self.sampler.sample_unique(num_sampled)

        true_cls = true_classes.as_in_context(ctx).astype('float64')
        prob_true = ((true_cls + 2.0) / (true_cls + 1.0)).log() / log_range
        count_true = self.prob_helper(num_tries, num_sampled, prob_true)

        sampled_classes = ndarray.array(sampled_classes, ctx=ctx, dtype='int64')
        sampled_cls_fp64 = sampled_classes.astype('float64')
        prob_sampled = ((sampled_cls_fp64 + 2.0) / (sampled_cls_fp64 + 1.0)).log() / log_range
        count_sampled = self.prob_helper(num_tries, num_sampled, prob_sampled)
        return [sampled_classes, count_true, count_sampled]

