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

# This file compares different RNN implementation on the PTB benchmark using the LSTM bucketing module.
# Note that this file is not exactly the same with the original source file. 
# Specifically, the following major changes have been made: 
#     (1) ported FusedRNNCell (from cudnn_lstm_bucketing.py) and OpenLSTMRNNCell
#     (2) replaced MXNet Speedometer with TensorboardSpeedometer
#     (3) ported TensorboardLogValidationMetricsCallback

import os
import argparse
import mxnet as mx

parser = argparse.ArgumentParser(description="Train RNN on PennTreeBank",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-layers', type=int, default=2,
                    help='number of stacked RNN layers')
parser.add_argument('--num-hidden', type=int, default=200,
                    help='hidden layer size')
parser.add_argument('--num-embed', type=int, default=200,
                    help='embedding layer size')
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
parser.add_argument('--backend', type=str, default='open',
                    help='select the RNN backend implementation')


def tokenize_text(fname, vocab=None, invalid_label=-1, start_label=0):
    assert os.path.isfile(fname), \
        "Cannot find the dataset, please make sure that you have " \
        "run the download script in the dataset directory."
    lines = open(fname).readlines()
    lines = [filter(None, i.split(' ')) for i in lines]
    sentences, vocab = mx.rnn.encode_sentences(sentences=lines, vocab=vocab, 
                                               start_label=start_label,
                                               invalid_label=invalid_label)

    return sentences, vocab


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)-15s %(message)s')

    args = parser.parse_args()

    # Cannot set the upper bound of bucket size to be 
    buckets = [10, 20, 30, 40, 50, 60]
    
    start_label, invalid_label = 1, 0

    train_sentences, vocab = tokenize_text("./dataset/ptb/ptb.train.txt", 
                                           start_label=start_label,
                                           invalid_label=invalid_label)
    valid_sentences, _ = tokenize_text("./dataset/ptb/ptb.valid.txt", vocab=vocab,
                                       start_label=start_label,
                                       invalid_label=invalid_label)

    backend = args.backend

    data_layout = 'TN' if backend == 'cudnn' else 'NT'
    data_train = mx.rnn.BucketSentenceIter(sentences=train_sentences, batch_size=args.batch_size,
                                           buckets=buckets, invalid_label=invalid_label,
                                           layout=data_layout)
    data_valid = mx.rnn.BucketSentenceIter(sentences=valid_sentences, batch_size=args.batch_size,
                                           buckets=buckets, invalid_label=invalid_label,
                                           layout=data_layout)
    
    rnn = mx.rnn.BaseRNNCell()

    if backend == 'default':
        rnn = mx.rnn.SequentialRNNCell()
    
        for i in range(args.num_layers):
            rnn.add(mx.rnn.LSTMCell(num_hidden=args.num_hidden, prefix='lstm_l%d_'%i))
    elif backend == 'cudnn':
        rnn = mx.rnn. FusedRNNCell(num_hidden=args.num_hidden,
                                   num_layers=args.num_layers,
                                   prefix='lstm_')
    elif backend == 'open':
        rnn = mx.rnn.OpenLSTMRNNCell(num_hidden=args.num_hidden,
                                     num_layers=args.num_layers,
                                     prefix='lstm_')
    else:
        assert 0, "Invalid backend argument. " \
                  "Valid ones are default/cudnn/open."

    def sym_gen(seq_len):
        data  = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        embed = mx.sym.Embedding(data=data, input_dim=len(vocab),
                                 output_dim=args.num_embed, name='embed')

        rnn.reset()
        
        if backend == 'default':
            output, _ = rnn.unroll(length=seq_len,
                                   inputs=embed,
                                   layout='NTC',
                                   merge_outputs=True)
        elif backend == 'cudnn':
            output, _ = rnn.unroll(length=seq_len,
                                   inputs=embed,
                                   layout='TNC',
                                   merge_outputs=True)
        elif backend == 'open':
            output, _ = rnn.unroll(length=seq_len,
                                   inputs=embed)
        else:
            assert 0, "Invalid backend argument. " \
                      "Valid ones are default/cudnn/open."
        
        pred = mx.sym.Reshape(output, shape=(-1, args.num_hidden))
        pred = mx.sym.FullyConnected(data=pred, num_hidden=len(vocab), name='pred')

        label = mx.sym.Reshape(label, shape=(-1,))
        pred  = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('data',), ('softmax_label',)

    model = mx.mod.BucketingModule(sym_gen = sym_gen,
                                   default_bucket_key = data_train.default_bucket_key,
                                   context = mx.gpu())

    logging.info("MXNet will be training using the RNN backend: %s.", backend)

    try:
        import mxboard
    except ImportError:
        logging.error("Please install mxboard using `sudo -H pip install mxboard`.")

    summary_writer = mxboard.SummaryWriter('./log')

    model.fit(
        train_data = data_train, 
        eval_data = data_valid, 
        eval_metric = mx.metric.Perplexity(invalid_label),
        kvstore = args.kv_store,
        optimizer = args.optimizer, 
        optimizer_params = {'learning_rate': args.lr,
                            'momentum': args.mom, 'wd': args.wd},
        initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
        num_epoch = args.num_epochs,
        batch_end_callback = mx.callback.TensorboardSpeedometer(summary_writer=summary_writer,
                                                                batch_size=args.batch_size, 
                                                                frequent=args.disp_batches, 
                                                                auto_reset=False),
        eval_end_callback = mx.callback.TensorboardLogValidationMetricsCallback(summary_writer=summary_writer))
