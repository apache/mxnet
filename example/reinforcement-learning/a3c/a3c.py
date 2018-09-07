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

from __future__ import print_function
import mxnet as mx
import numpy as np
import rl_data
import sym
import argparse
import logging
import os
import gym
from datetime import datetime
import time
import sys
try:
    from importlib import reload
except ImportError:
    pass

parser = argparse.ArgumentParser(description='Traing A3C with OpenAI Gym')
parser.add_argument('--test', action='store_true', help='run testing', default=False)
parser.add_argument('--log-file', type=str, help='the name of log file')
parser.add_argument('--log-dir', type=str, default="./log", help='directory of the log file')
parser.add_argument('--model-prefix', type=str, help='the prefix of the model to load')
parser.add_argument('--save-model-prefix', type=str, help='the prefix of the model to save')
parser.add_argument('--load-epoch', type=int, help="load the model on an epoch using the model-prefix")

parser.add_argument('--kv-store', type=str, default='device', help='the kvstore type')
parser.add_argument('--gpus', type=str, help='the gpus will be used, e.g "0,1,2,3"')

parser.add_argument('--num-epochs', type=int, default=120, help='the number of training epochs')
parser.add_argument('--num-examples', type=int, default=1000000, help='the number of training examples')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--input-length', type=int, default=4)

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--t-max', type=int, default=4)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--beta', type=float, default=0.08)

args = parser.parse_args()

def log_config(log_dir=None, log_file=None, prefix=None, rank=0):
    reload(logging)
    head = '%(asctime)-15s Node[' + str(rank) + '] %(message)s'
    if log_dir:
        logging.basicConfig(level=logging.DEBUG, format=head)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not log_file:
            log_file = (prefix if prefix else '') + datetime.now().strftime('_%Y_%m_%d-%H_%M.log')
            log_file = log_file.replace('/', '-')
        else:
            log_file = log_file
        log_file_full_name = os.path.join(log_dir, log_file)
        handler = logging.FileHandler(log_file_full_name, mode='w')
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        logging.info('start with arguments %s', args)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('start with arguments %s', args)

def train():
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    model_prefix = args.model_prefix
    if model_prefix is not None:
        model_prefix += "-%d" % (kv.rank)
    save_model_prefix = args.save_model_prefix
    if save_model_prefix is None:
        save_model_prefix = model_prefix

    log_config(args.log_dir, args.log_file, save_model_prefix, kv.rank)

    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    epoch_size = args.num_examples / args.batch_size

    if args.kv_store == 'dist_sync':
        epoch_size /= kv.num_workers

    # disable kvstore for single device
    if 'local' in kv.type and (
            args.gpus is None or len(args.gpus.split(',')) is 1):
        kv = None

    # module
    dataiter = rl_data.GymDataIter('Breakout-v0', args.batch_size, args.input_length, web_viz=True)
    net = sym.get_symbol_atari(dataiter.act_dim)
    module = mx.mod.Module(net, data_names=[d[0] for d in dataiter.provide_data], label_names=('policy_label', 'value_label'), context=devs)
    module.bind(data_shapes=dataiter.provide_data,
                label_shapes=[('policy_label', (args.batch_size,)), ('value_label', (args.batch_size, 1))],
                grad_req='add')

    # load model

    if args.load_epoch is not None:
        assert model_prefix is not None
        _, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.load_epoch)
    else:
        arg_params = aux_params = None

    # save model
    checkpoint = None if save_model_prefix is None else mx.callback.do_checkpoint(save_model_prefix)

    init = mx.init.Mixed(['fc_value_weight|fc_policy_weight', '.*'],
                         [mx.init.Uniform(0.001), mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)])
    module.init_params(initializer=init,
                       arg_params=arg_params, aux_params=aux_params)

    # optimizer
    module.init_optimizer(kvstore=kv, optimizer='adam',
                          optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'epsilon': 1e-3})

    # logging
    np.set_printoptions(precision=3, suppress=True)

    T = 0
    dataiter.reset()
    score = np.zeros((args.batch_size, 1))
    final_score = np.zeros((args.batch_size, 1))
    for epoch in range(args.num_epochs):
        if save_model_prefix:
            module.save_params('%s-%04d.params'%(save_model_prefix, epoch))


        for _ in range(int(epoch_size/args.t_max)):
            tic = time.time()
            # clear gradients
            for exe in module._exec_group.grad_arrays:
                for g in exe:
                    g[:] = 0

            S, A, V, r, D = [], [], [], [], []
            for t in range(args.t_max + 1):
                data = dataiter.data()
                module.forward(mx.io.DataBatch(data=data, label=None), is_train=False)
                act, _, val = module.get_outputs()
                V.append(val.asnumpy())
                if t < args.t_max:
                    act = act.asnumpy()
                    act = [np.random.choice(dataiter.act_dim, p=act[i]) for i in range(act.shape[0])]
                    reward, done = dataiter.act(act)
                    S.append(data)
                    A.append(act)
                    r.append(reward.reshape((-1, 1)))
                    D.append(done.reshape((-1, 1)))

            err = 0
            R = V[args.t_max]
            for i in reversed(range(args.t_max)):
                R = r[i] + args.gamma * (1 - D[i]) * R
                adv = np.tile(R - V[i], (1, dataiter.act_dim))

                batch = mx.io.DataBatch(data=S[i], label=[mx.nd.array(A[i]), mx.nd.array(R)])
                module.forward(batch, is_train=True)

                pi = module.get_outputs()[1]
                h = -args.beta*(mx.nd.log(pi+1e-7)*pi)
                out_acts = np.amax(pi.asnumpy(), 1)
                out_acts=np.reshape(out_acts,(-1,1))
                out_acts_tile=np.tile(-np.log(out_acts + 1e-7),(1, dataiter.act_dim))
                module.backward([mx.nd.array(out_acts_tile*adv), h])

                print('pi', pi[0].asnumpy())
                print('h', h[0].asnumpy())
                err += (adv**2).mean()
                score += r[i]
                final_score *= (1-D[i])
                final_score += score * D[i]
                score *= 1-D[i]
                T += D[i].sum()

            module.update()
            logging.info('fps: %f err: %f score: %f final: %f T: %f'%(args.batch_size/(time.time()-tic), err/args.t_max, score.mean(), final_score.mean(), T))
            print(score.squeeze())
            print(final_score.squeeze())

def test():
    log_config()

    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # module
    dataiter = rl_data.GymDataIter('scenes', args.batch_size, args.input_length, web_viz=True)
    print(dataiter.provide_data)
    net = sym.get_symbol_thor(dataiter.act_dim)
    module = mx.mod.Module(net, data_names=[d[0] for d in dataiter.provide_data], label_names=('policy_label', 'value_label'), context=devs)
    module.bind(data_shapes=dataiter.provide_data,
                label_shapes=[('policy_label', (args.batch_size,)), ('value_label', (args.batch_size, 1))],
                for_training=False)

    # load model
    assert args.load_epoch is not None
    assert args.model_prefix is not None
    module.load_params('%s-%04d.params'%(args.model_prefix, args.load_epoch))

    N = args.num_epochs * args.num_examples / args.batch_size

    R = 0
    T = 1e-20
    score = np.zeros((args.batch_size,))
    for t in range(N):
        dataiter.clear_history()
        data = dataiter.next()
        module.forward(data, is_train=False)
        act = module.get_outputs()[0].asnumpy()
        act = [np.random.choice(dataiter.act_dim, p=act[i]) for i in range(act.shape[0])]
        dataiter.act(act)
        time.sleep(0.05)
        _, reward, _, done = dataiter.history[0]
        T += done.sum()
        score += reward
        R += (done*score).sum()
        score *= (1-done)

        if t % 100 == 0:
            logging.info('n %d score: %f T: %f'%(t, R/T, T))


if __name__ == '__main__':
    if args.test:
        test()
    else:
        train()


