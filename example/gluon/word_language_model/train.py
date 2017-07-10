import argparse
import time
import math
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn
import model
import data

parser = argparse.ArgumentParser(description='MXNet Autograd PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/ptb.',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='lstm',
                    help='type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='Whether to use gpu')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.params',
                    help='path to save the final model')
args = parser.parse_args()


###############################################################################
# Load data
###############################################################################


if args.cuda:
    context = mx.gpu(0)
else:
    context = mx.cpu(0)

corpus = data.Corpus(args.data)

def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data

train_data = batchify(corpus.train, args.batch_size).as_in_context(context)
val_data = batchify(corpus.valid, args.batch_size).as_in_context(context)
test_data = batchify(corpus.test, args.batch_size).as_in_context(context)


###############################################################################
# Build the model
###############################################################################


ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
model.collect_params().initialize(mx.init.Xavier(), ctx=context)
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                      {'learning_rate': args.lr,
                       'momentum': 0,
                       'wd': 0})
loss = gluon.loss.SoftmaxCrossEntropyLoss()

###############################################################################
# Training code
###############################################################################


def get_batch(source, i):
    seq_len = min(args.bptt, source.shape[0] - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    return data, target.reshape((-1,))


def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden


def eval(data_source):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=args.batch_size, ctx=context)
    for ibatch, i in enumerate(range(0, data_source.shape[0] - 1, args.bptt)):
        data, target = get_batch(data_source, i)
        output, hidden = model(data, hidden)
        L = loss(output, target)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal


def train():
    best_val = None
    for epoch in range(args.epochs):
        total_L = 0.0
        start_time = time.time()
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=args.batch_size, ctx=context)
        for ibatch, i in enumerate(range(0, train_data.shape[0] - 1, args.bptt)):
            data, target = get_batch(train_data, i)
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target)
                L.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]
            # Here gradient is not divided by batch_size yet.
            # So we multiply max_norm by batch_size to balance it.
            gluon.utils.clip_global_norm(grads, args.clip * args.batch_size)

            trainer.step(args.batch_size)
            total_L += mx.nd.sum(L).asscalar()

            if ibatch % args.log_interval == 0 and ibatch > 0:
                cur_L = total_L / args.batch_size / args.bptt / args.log_interval
                print('[Epoch %d Batch %d] loss %.2f, ppl %.2f'%(
                    epoch, ibatch, cur_L, math.exp(cur_L)))
                total_L = 0.0

        val_L = eval(val_data)

        print('[Epoch %d] time cost %.2fs, valid loss %.2f, valid ppl %.2f'%(
            epoch, time.time()-start_time, val_L, math.exp(val_L)))


if __name__ == '__main__':
    train()
    test_L = eval(test_data)
    print('test loss %.2f, test ppl %.2f'%(test_L, math.exp(test_L)))
