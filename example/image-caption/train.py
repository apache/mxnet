# coding: utf-8

from dataloader import ImageCaptionIter
import mxnet as mx
import numpy as np
from net import vgg16_fc7_symbol, network_unroll
import json


def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)

batch_size = 8
num_hidden = 256
num_embed = 256
num_lstm_layer = 1
ctx = mx.gpu(0)

init_states = [('l%d_init_c' % t, (batch_size, num_hidden))
               for t in range(num_lstm_layer)] + \
              [('l%d_init_h' % t, (batch_size, num_hidden))
               for t in range(num_lstm_layer)]

info = json.load(open('./flickr8k.json'))
vocab_size = len(info['ix_to_word'])

train_data = ImageCaptionIter('./flickr8k.h5', batch_size, init_states, vocab_size)

vgg_params = mx.nd.load('./vgg-0001.params')

cnn_in_shapes = dict(train_data.provide_data[:1])

lm_in_shapes = {'image_feature': (batch_size, 4096)}
lm_in_shapes.update(dict(train_data.provide_data[1:] + train_data.provide_label))

cnn = vgg16_fc7_symbol('image')
lm = network_unroll(num_lstm_layer, train_data.seq_length, len(info['ix_to_word'])+2,
                   num_hidden, num_embed,)

cnn_exec = cnn.simple_bind(ctx=ctx, is_train=False, **cnn_in_shapes)
lm_exec = lm.simple_bind(ctx=ctx, grad_req='add', **lm_in_shapes)

def init_lm(lm):
    for key, arr in lm.arg_dict.items():
        if 'weight' in key:
            arr[:] = mx.random.uniform(-0.07, 0.07, arr.shape)
        elif 'bias' in key:
            arr[:] = 0.
        else:
            arr[:] = mx.random.uniform(-0.07, 0.07, arr.shape)

def init_cnn(cnn, pretrained_cnn):
    for key, arr in cnn.arg_dict.items():
        if key == 'image':
            continue
        arr[:] = pretrained_cnn['arg:' + key]

init_lm(lm_exec)
init_cnn(cnn_exec, vgg_params)

opt = mx.optimizer.create('sgd')
opt.lr = 0.01
updater = mx.optimizer.get_updater(opt)

for epoch in range(1, 10):
    for i, batch in enumerate(train_data):
        cnn_input = dict(zip(map(lambda x: x[0], batch.provide_data), batch.data)[:1])
        lm_input = dict(zip(map(lambda x: x[0], batch.provide_data), batch.data)[1:])
        # cnn forward
        # update cnn input
        cnn_exec.arg_dict['image'][:] = cnn_input['image']
        cnn_exec.forward()
        image_feature = cnn_exec.outputs[0]
        # lm forward
        # updata lm input
        lm_input['image_feature'] = image_feature
        for key, arr in lm_input.items():
            lm_exec.arg_dict[key][:] = arr
        # update lm grag
        for key, arr in lm_exec.grad_dict.items():
            arr[:] = 0.
        lm_exec.forward()
        print Perplexity(batch._label, lm_exec.outputs[0].asnumpy())
        lm_exec.backward()

        for j, name in enumerate(lm.list_arguments()):
            if name in lm_in_shapes.keys():
                continue
            updater(j, lm_exec.grad_dict[name], lm_exec.arg_dict[name])
