import os
import time
import mxnet as mx
import numpy as np
import net 
import cPickle as pickle
from matplotlib import pyplot as plt
from skimage import io, transform

import utils

class Maker():
    def __init__(self, model_prefix, output_shape):
        s1, s0 = output_shape
        s0 = s0//32*32
        s1 = s1//32*32
        self.s0 = s0
        self.s1 = s1
        generator = net.generator_symbol()
        args = mx.nd.load('%s_args.nd'%model_prefix)
        auxs = mx.nd.load('%s_auxs.nd'%model_prefix)
        args['data'] = mx.nd.zeros([1,3,s0+128,s1+128], mx.gpu())
        self.gene_executor = generator.bind(ctx=mx.gpu(), args=args, aux_states=auxs)

    def generate(self, save_path, content_path):
        self.gene_executor.arg_dict['data'][:] = utils.preprocess_img_test(content_path, (self.s0, self.s1))
        self.gene_executor.forward(is_train=True)
        out = self.gene_executor.outputs[0].asnumpy()
        im = utils.postprocess_img(out[0])
        io.imsave(save_path, im)


def test():
    model_prefix = 'test1'
    output_shape = (512, 512) 
    output_path = 'output.jpg'
    content_image_path = 'content/shenyang.jpg'
    maker = Maker(model_prefix, output_shape) 
    maker.generate(output_path, content_image_path)

if __name__ == "__main__":
   test()
