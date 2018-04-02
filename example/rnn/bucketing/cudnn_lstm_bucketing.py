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

import numpy as np
import mxnet as mx
import argparse

parser = argparse.ArgumentParser(description="Train RNN on Penn Tree Bank",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--test', default=False, action='store_true',
                    help='whether to do testing instead of training')
parser.add_argument('--model-prefix', type=str, default=None,
                    help='path to save/load model')
parser.add_argument('--load-epoch', type=int, default=0,
                    help='load from epoch')
parser.add_argument('--num-layers', type=int, default=2,
                    help='number of stacked RNN layers')
parser.add_argument('--num-hidden', type=int, default=200,
                    help='hidden layer size')
parser.add_argument('--num-embed', type=int, default=200,
                    help='embedding layer size')
parser.add_argument('--bidirectional', action='store_true',
                    help='uses bidirectional layers if specified')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ' \
                         'Increase batch size when using multiple gpus for best performance.')
parser.add_argument('--kv-store', type=str, default='device',
                    help='key-value store type')
parser.add_argument('--num-epochs', type=int, default=25,
                    help='max num of epochs')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='the optimizer type')
parser.add_argument('--mom', type=float, default=0.0,
                    help='momentum for sgd')
parser.add_argument('--wd', type=float, default=0.00001,
                    help='weight decay for sgd')
parser.add_argument('--batch-size', type=int, default=32,
                    help='the batch size.')
parser.add_argument('--disp-batches', type=int, default=50,
                    help='show progress for every n batches')
# When training a deep, complex model *on multiple GPUs* it's recommended to
# stack fused RNN cells (one layer per cell) together instead of one with all
# layers. The reason is that fused RNN cells don't set gradients to be ready
# until the computation for the entire layer is completed. Breaking a
# multi-layer fused RNN cell into several one-layer ones allows gradients to be
# processed ealier. This reduces communication overhead, especially with
# multiple GPUs.
parser.add_argument('--stack-rnn', default=False,
                    help='stack fused RNN cells to reduce communication overhead')
parser.add_argument('--dropout', type=float, default='0.0',
                    help='dropout probability (1.0 - keep probability)')

#buckets = [32]
buckets = [10, 20, 30, 40, 50, 60]

start_label = 1
invalid_label = 0

def tokenize_text(fname, vocab=None, invalid_label=-1, start_label=0):
    lines = open(fname).readlines()
    lines = [filter(None, i.split(' ')) for i in lines]
    sentences, vocab = mx.rnn.encode_sentences(lines, vocab=vocab, invalid_label=invalid_label, start_label=start_label)
    return sentences, vocab

def get_data(layout):
    train_sent, vocab = tokenize_text("./data/ptb.train.txt", start_label=start_label,
                                      invalid_label=invalid_label)
    val_sent, _ = tokenize_text("./data/ptb.test.txt", vocab=vocab, start_label=start_label,
                                invalid_label=invalid_label)

    data_train  = mx.rnn.BucketSentenceIter(train_sent, args.batch_size, buckets=buckets,
                                            invalid_label=invalid_label, layout=layout)
    data_val    = mx.rnn.BucketSentenceIter(val_sent, args.batch_size, buckets=buckets,
                                            invalid_label=invalid_label, layout=layout)
    return data_train, data_val, vocab


def train(args):
    data_train, data_val, vocab = get_data('TN')
    if args.stack_rnn:
        cell = mx.rnn.SequentialRNNCell()
        for i in range(args.num_layers):
            cell.add(mx.rnn.FusedRNNCell(args.num_hidden, num_layers=1,
                                         mode='lstm', prefix='lstm_l%d'%i,
                                         bidirectional=args.bidirectional))
            if args.dropout > 0 and i < args.num_layers - 1:
                cell.add(mx.rnn.DropoutCell(args.dropout, prefix='lstm_d%d'%i))
    else:
        cell = mx.rnn.FusedRNNCell(args.num_hidden, num_layers=args.num_layers, dropout=args.dropout,
                                   mode='lstm', bidirectional=args.bidirectional)

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        embed = mx.sym.Embedding(data=data, input_dim=len(vocab), output_dim=args.num_embed,name='embed')

        output, _ = cell.unroll(seq_len, inputs=embed, merge_outputs=True, layout='TNC')

        pred = mx.sym.Reshape(output,
                shape=(-1, args.num_hidden*(1+args.bidirectional)))
        pred = mx.sym.FullyConnected(data=pred, num_hidden=len(vocab), name='pred')

        label = mx.sym.Reshape(label, shape=(-1,))
        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('data',), ('softmax_label',)

    if args.gpus:
        contexts = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        contexts = mx.cpu(0)

    model = mx.mod.BucketingModule(
        sym_gen             = sym_gen,
        default_bucket_key  = data_train.default_bucket_key,
        context             = contexts)

    if args.load_epoch:
        _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(
            cell, args.model_prefix, args.load_epoch)
    else:
        arg_params = None
        aux_params = None

    opt_params = {
      'learning_rate': args.lr,
      'wd': args.wd
    }

    if args.optimizer not in ['adadelta', 'adagrad', 'adam', 'rmsprop']:
        opt_params['momentum'] = args.mom

    model.fit(
        train_data          = data_train,
        eval_data           = data_val,
        eval_metric         = mx.metric.Perplexity(invalid_label),
        kvstore             = args.kv_store,
        optimizer           = args.optimizer,
        optimizer_params    = opt_params,
        initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
        arg_params          = arg_params,
        aux_params          = aux_params,
        begin_epoch         = args.load_epoch,
        num_epoch           = args.num_epochs,
        batch_end_callback  = mx.callback.Speedometer(args.batch_size, args.disp_batches, auto_reset=False),
        epoch_end_callback  = mx.rnn.do_rnn_checkpoint(cell, args.model_prefix, 1)
                              if args.model_prefix else None)

def test(args):
    assert args.model_prefix, "Must specifiy path to load from"
    _, data_val, vocab = get_data('NT')

    if not args.stack_rnn:
        stack = mx.rnn.FusedRNNCell(args.num_hidden, num_layers=args.num_layers,
                mode='lstm', bidirectional=args.bidirectional).unfuse()
    else:
        stack = mx.rnn.SequentialRNNCell()
        for i in range(args.num_layers):
            cell = mx.rnn.LSTMCell(num_hidden=args.num_hidden, prefix='lstm_%dl0_'%i)
            if args.bidirectional:
                cell = mx.rnn.BidirectionalCell(
                        cell,
                        mx.rnn.LSTMCell(num_hidden=args.num_hidden, prefix='lstm_%dr0_'%i),
                        output_prefix='bi_lstm_%d'%i)
            stack.add(cell)

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        embed = mx.sym.Embedding(data=data, input_dim=len(vocab),
                                 output_dim=args.num_embed, name='embed')

        stack.reset()
        outputs, states = stack.unroll(seq_len, inputs=embed, merge_outputs=True)

        pred = mx.sym.Reshape(outputs,
                shape=(-1, args.num_hidden*(1+args.bidirectional)))
        pred = mx.sym.FullyConnected(data=pred, num_hidden=len(vocab), name='pred')

        label = mx.sym.Reshape(label, shape=(-1,))
        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('data',), ('softmax_label',)

    if args.gpus:
        contexts = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        contexts = mx.cpu(0)

    model = mx.mod.BucketingModule(
        sym_gen             = sym_gen,
        default_bucket_key  = data_val.default_bucket_key,
        context             = contexts)
    model.bind(data_val.provide_data, data_val.provide_label, for_training=False)

    # note here we load using SequentialRNNCell instead of FusedRNNCell.
    _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(stack, args.model_prefix, args.load_epoch)
    model.set_params(arg_params, aux_params)

    model.score(data_val, mx.metric.Perplexity(invalid_label),
                batch_end_callback=mx.callback.Speedometer(args.batch_size, 5))

if __name__ == '__main__':
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    args = parser.parse_args()

    if args.num_layers >= 4 and len(args.gpus.split(',')) >= 4 and not args.stack_rnn:
        print('WARNING: stack-rnn is recommended to train complex model on multiple GPUs')

    if args.test:
        # Demonstrates how to load a model trained with CuDNN RNN and predict
        # with non-fused MXNet symbol
        test(args)
    else:
        train(args)
