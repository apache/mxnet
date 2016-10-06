import os
import sys
import mxnet as mx
import logging

import sd_module

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "image-classification")))
from train_cifar10 import get_iterator


def residual_module(death_rate, n_channel, name_scope, context, stride=1, bn_momentum=0.9):
    data = mx.sym.Variable(name_scope + '_data')

    # computation branch:
    #   BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    bn1 = mx.symbol.BatchNorm(data=data, name=name_scope + '_bn1', fix_gamma=False,
        momentum=bn_momentum,
        # Same with https://github.com/soumith/cudnn.torch/blob/master/BatchNormalization.lua
        # cuDNN v5 don't allow a small eps of 1e-5
        eps=2e-5
    )
    relu1 = mx.symbol.Activation(data=bn1, act_type='relu', name=name_scope+'_relu1')
    conv1 = mx.symbol.Convolution(data=relu1, num_filter=n_channel, kernel=(3, 3), pad=(1,1),
                                  stride=(stride, stride), name=name_scope+'_conv1')
    bn2 = mx.symbol.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_momentum,
                              eps=2e-5, name=name_scope+'_bn2')
    relu2 = mx.symbol.Activation(data=bn2, act_type='relu', name=name_scope+'_relu2')
    conv2 = mx.symbol.Convolution(data=relu2, num_filter=n_channel, kernel=(3, 3), pad=(1,1),
                                  stride=(1, 1), name=name_scope+'_conv2')
    sym_compute = conv2

    # skip branch
    if stride > 1:
        sym_skip = mx.symbol.Convolution(data=data, num_filter=n_channel, kernel=(3, 3), pad=(1, 1),
                                         stride=(stride, stride), name=name_scope+'_skip_conv')
    else:
        sym_skip = None

    mod = sd_module.StochasticDepthModule(sym_compute, sym_skip, data_names=[name_scope+'_data'],
                                          context=context, death_rate=death_rate)
    return mod


#################################################################################
# Build architecture
# Configurations
bn_momentum = 0.9
contexts = [mx.context.gpu(i) for i in range(8)]
n_residual_blocks = 18
death_rate = 0.5
death_mode = 'linear_decay'  # 'linear_decay' or 'uniform'

n_classes = 10

def get_death_rate(i_res_block):
    n_total_res_blocks = n_residual_blocks * 3
    if death_mode == 'linear_decay':
        my_death_rate = float(i_res_block) / n_total_res_blocks * death_rate
    else:
        my_death_rate = death_rate
    return my_death_rate

# 0. base ConvNet
sym_base = mx.sym.Variable('data')
sym_base = mx.sym.Convolution(data=sym_base, num_filter=16, kernel=(3, 3), pad=(1, 1), name='conv1')
sym_base = mx.sym.BatchNorm(data=sym_base, name='bn1', fix_gamma=False, momentum=bn_momentum, eps=2e-5)
sym_base = mx.sym.Activation(data=sym_base, name='relu1', act_type='relu')
mod_base = mx.mod.Module(sym_base, context=contexts, label_names=None)

# 1. container
mod_seq = mx.mod.SequentialModule()
mod_seq.add(mod_base)

# 2. first group, 16 x 28 x 28
i_res_block = 0
for i in range(n_residual_blocks):
    mod_seq.add(residual_module(get_death_rate(i_res_block), 16, 'res_A_%d' % i, contexts), auto_wiring=True)
    i_res_block += 1

# 3. second group, 32 x 14 x 14
mod_seq.add(residual_module(get_death_rate(i_res_block), 32, 'res_AB', contexts, stride=2), auto_wiring=True)
i_res_block += 1

for i in range(n_residual_blocks-1):
    mod_seq.add(residual_module(get_death_rate(i_res_block), 32, 'res_B_%d' % i, contexts), auto_wiring=True)
    i_res_block += 1

# 4. third group, 64 x 7 x 7
mod_seq.add(residual_module(get_death_rate(i_res_block), 64, 'res_BC', contexts, stride=2), auto_wiring=True)
i_res_block += 1

for i in range(n_residual_blocks-1):
    mod_seq.add(residual_module(get_death_rate(i_res_block), 64, 'res_C_%d' % i, contexts), auto_wiring=True)
    i_res_block += 1

# 5. final module
sym_final = mx.sym.Variable('data')
sym_final = mx.sym.Pooling(data=sym_final, kernel=(7, 7), pool_type='avg', name='global_pool')
sym_final = mx.sym.FullyConnected(data=sym_final, num_hidden=n_classes, name='logits')
sym_final = mx.sym.SoftmaxOutput(data=sym_final, name='softmax')
mod_final = mx.mod.Module(sym_final, context=contexts)
mod_seq.add(mod_final, auto_wiring=True, take_labels=True)


#################################################################################
# Training
num_examples = 60000
batch_size = 128
base_lr = 0.05
lr_factor = 0.1
lr_factor_epoch = 10
momentum = 0.9
weight_decay = 0.00001
kv_store = 'local'

initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)
num_epochs = 50

epoch_size = num_examples / batch_size
lr_scheduler = mx.lr_scheduler.FactorScheduler(step=max(int(epoch_size * lr_factor_epoch), 1), factor=lr_factor)

batch_end_callbacks = [mx.callback.Speedometer(batch_size, 50)]
epoch_end_callbacks = [mx.callback.do_checkpoint('sd-%d' % (n_residual_blocks * 6 + 2))]


args = type('', (), {})()
args.batch_size = batch_size
args.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "image-classification", "cifar10")) + '/'
kv = mx.kvstore.create(kv_store)
train, val = get_iterator(args, kv)

logging.basicConfig(level=logging.DEBUG)
mod_seq.fit(train, val,
            optimizer_params={'learning_rate': base_lr, 'momentum': momentum,
                              'lr_scheduler': lr_scheduler, 'wd': weight_decay},
            num_epoch=num_epochs, batch_end_callback=batch_end_callbacks,
            epoch_end_callback=epoch_end_callbacks,
            initializer=initializer)

