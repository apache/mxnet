# pylint:skip-file
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import os
import argparse
import time


def safe_eval(expr):
    import ast
    if type(expr) is str:
        return ast.literal_eval(expr)
    else:
        return expr

class Corpus(object):
    def __init__(self, train_path, valid_path, test_path):
        self.dictionary = dict()
        self.dictionary['<eos>'] = 0
        self.dictionary_idx = 1
        self.train = self.tokenize(train_path)
        self.valid = self.tokenize(valid_path)
        self.test = self.tokenize(test_path)
        self._train_path = train_path
        self._test_path = test_path
        self._valid_path = valid_path

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        content = open(path, 'r').read()
        content = content.replace('\n', '<eos>')
        content = list(filter(None, content.split(' ')))
        data = np.zeros(len(content), dtype=np.int)
        for i in range(len(content)):
            word = content[i]
            if not word in self.dictionary:
                self.dictionary[word] = self.dictionary_idx
                self.dictionary_idx += 1
            data[i] = self.dictionary[word]
        return data

    def summary(self):
        print("Vocabulary size = %d" %len(self.dictionary))
        print("Training data: %s, number of words = %d" %(self._train_path, self.train.shape[0]))
        print("Testing data: %s, number of words = %d" % (self._test_path, self.test.shape[0]))
        print("Validation data: %s, number of words = %d" % (self._valid_path, self.valid.shape[0]))


def parse_args():
    parser = argparse.ArgumentParser(description="Train RNN on Penn Tree Bank",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default="train", choices=['train', 'test'],
                        help='whether to do testing or training')
    parser.add_argument('--model-prefix', type=str, default=None,
                        help='path to save/load model')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of stacked RNN layers')
    parser.add_argument('--num-hidden', type=int, default=200,
                        help='hidden layer size')
    parser.add_argument('--num-embed', type=int, default=200,
                        help='embedding layer size')
    parser.add_argument('--gpus', type=str,
                        help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ' \
                             'Increase batch size when using multiple gpus for best performance.')
    parser.add_argument('--kv-store', type=str, default='device',
                        help='key-value store type')
    parser.add_argument('--num-epochs', type=int, default=25,
                        help='max num of epochs')
    parser.add_argument('--bptt', type=int, default=35,
                        help='length of the sequence length used in BPTT.')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='initial learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.5,
                        help='the decay rate of lr')
    parser.add_argument('--decay-epoch-num', type=float, default=4,
                        help='the number of epochs per learning rate decay')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='the optimizer type')
    parser.add_argument('--max-norm', type=float, default=5.0,
                        help='the norm clipping value')
    parser.add_argument('--mom', type=float, default=0.0,
                        help='momentum for sgd')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay for sgd')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='the batch size.')
    parser.add_argument('--disp-batches', type=int, default=200,
                        help='show progress for every n batches')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout probability (1.0 - keep probability)')
    args = parser.parse_args()
    return args

def batchify(data, batch_size, no_trim_off=False):
    seg_seq_len = data.shape[0] // batch_size
    if no_trim_off:
        assert seg_seq_len * batch_size == data.shape[0]
    x_cut = data[:seg_seq_len * batch_size]  # Here we just trim off the extra elements
    data = x_cut.reshape((batch_size, seg_seq_len)).T
    return data


def lstm_ptb_sym(seq_len, vocab_size, args):
    data = mx.sym.var('data')  # Layout (T, N)
    target = mx.sym.var('target')  # Layout (T, N)
    stack_lstms = [mx.rnn.FusedRNNCell(num_hidden=args.num_hidden,
                                       get_next_state=True,
                                       forget_bias=0.0,
                                       prefix="lstm%d_" % i) for i in range(args.num_layers)]
    prev_state_list = sum([lstm.begin_state(func=mx.sym.var) for lstm in stack_lstms], [])
    # Here the prev_state_list should have names like
    #  [lstm0_begin_state_0, lstm0_begin_state_1, ...]
    embed = mx.sym.Embedding(data=data, input_dim=vocab_size,
                             output_dim=args.num_embed,
                             name='embed')  # Layout (T, N, C)
    out = embed
    next_states = []
    for i in range(args.num_layers):
        out, states = stack_lstms[i].unroll(length=seq_len, inputs=out,
                                            begin_state=prev_state_list[2*i:2*i + 2],
                                            layout="TNC", merge_outputs=True)
        out = mx.sym.Dropout(out, p=args.dropout)
        next_states += states
    out = mx.sym.Reshape(out, shape=(-1, args.num_hidden))  # Shape (T*N, C)
    logits = mx.sym.FullyConnected(data=out, num_hidden=vocab_size, name='logits')
    target = mx.sym.reshape(target, shape=(-1,))  # Shape (T*N, )
    loss = mx.sym.pick(mx.sym.log_softmax(logits), target, axis=-1)
    loss = mx.sym.make_loss(- mx.sym.sum(loss), name="nll")
    return loss, mx.sym.reshape(logits, shape=(seq_len, -1, vocab_size)),\
           prev_state_list, next_states


def evaluate(test_net, test_data, args):
    exe_num = len(test_net._context)
    curr_states = test_net.get_states(merge_multi_context=False)
    # Set the state to zero when a new epoch begins
    for state_id in range(len(curr_states)):
        for exe_id in range(exe_num):
            curr_states[state_id][exe_id][:] = 0
    test_net.set_states(curr_states)
    total_nll = 0.0
    for i, start in enumerate(range(0, test_data.shape[0] - 1, args.bptt)):
        start = i * args.bptt
        data_batch_npy = np.take(test_data,
                                 np.arange(start, start + args.bptt),
                                 axis=0,
                                 mode="clip")
        target_batch_npy = np.take(test_data,
                                   np.arange(start + 1, start + 1 + args.bptt),
                                   axis=0,
                                   mode="clip")
        if start + args.bptt > test_data.shape[0]:
            valid_seq_len = test_data.shape[0] - start
        else:
            valid_seq_len = args.bptt
        test_net.forward(data_batch=mx.io.DataBatch(data=[mx.nd.array(data_batch_npy)]),
                         is_train=False)
        outputs = test_net.get_outputs(merge_multi_context=False)
        local_nll = 0.0
        for exe_id in range(exe_num):
            logits = outputs[0][exe_id]
            nll = - nd.pick(nd.log_softmax(logits), nd.array(target_batch_npy, ctx=logits.context),
                            axis=-1).asnumpy()
            local_nll += nll[:valid_seq_len, :].mean() * valid_seq_len
        total_nll += local_nll / exe_num
        for out_id in range(1, len(outputs)):
            for exe_id in range(exe_num):
                curr_states[out_id - 1][exe_id] = outputs[out_id][exe_id]
        test_net.set_states(states=curr_states)
    avg_nll = total_nll / test_data.shape[0]
    return avg_nll

def train(args, contexts):
    exe_num = len(contexts)
    ptb_data = Corpus(train_path=os.path.join('data', 'ptb.train.txt'),
                      test_path=os.path.join('data', 'ptb.test.txt'),
                      valid_path=os.path.join('data', 'ptb.valid.txt'))
    ptb_data.summary()

    eval_batch = 10
    train_data = batchify(ptb_data.train, args.batch_size)
    test_data = batchify(ptb_data.test, eval_batch, True)
    valid_data = batchify(ptb_data.valid, eval_batch, True)

    loss_sym, logits_sym, prev_state_list, next_states =\
        lstm_ptb_sym(seq_len=args.bptt, vocab_size=len(ptb_data.dictionary), args=args)
    data_desc = [mx.io.DataDesc(name="data", shape=(args.bptt, args.batch_size), layout="TN")]
    test_data_desc = [mx.io.DataDesc(name="data", shape=(args.bptt, eval_batch), layout="TN")]
    label_desc = [mx.io.DataDesc(name="target", shape=(args.bptt, args.batch_size), layout="TN")]
    net = mx.mod.Module(mx.sym.Group([loss_sym] + [mx.sym.BlockGrad(ele) for ele in next_states]),
                        data_names=["data"],
                        label_names=["target"],
                        state_names=[state.name for state in prev_state_list],
                        context=contexts)
    net.bind(data_shapes=data_desc, label_shapes=label_desc)
    net.init_params(initializer=mx.init.Uniform(0.1))
    net.summary()
    net.init_optimizer(kvstore=args.kv_store, optimizer=args.optimizer,
                       optimizer_params={'learning_rate': args.lr,
                                         'wd': args.wd})
    test_net = mx.mod.Module(mx.sym.Group([logits_sym] +
                                          [mx.sym.BlockGrad(ele) for ele in next_states]),
                             data_names=["data"],
                             label_names=None,
                             state_names=[state.name for state in prev_state_list],
                             context=contexts)
    test_net.bind(data_shapes=test_data_desc, for_training=False, shared_module=net)
    # Preparing the states of the RNN
    curr_states = net.get_states(merge_multi_context=False)
    print("curr_states:", curr_states)
    for epoch in range(args.num_epochs):
        start_time = time.time()
        # Set the state to zero when a new epoch begins
        for state_id in range(len(curr_states)):
            for exe_id in range(exe_num):
                curr_states[state_id][exe_id][:] = 0
        net.set_states(states=curr_states)
        total_loss = 0
        total_batch_num = (train_data.shape[0] - 1) // args.bptt
        for i in range(total_batch_num):
            start = i * args.bptt
            data_batch_npy = train_data[start:start + args.bptt]
            target_batch_npy = train_data[(start + 1):(start + 1 + args.bptt)]
            net.forward_backward(data_batch=mx.io.DataBatch(data=[mx.nd.array(data_batch_npy)],
                                                            label=[mx.nd.array(target_batch_npy)]))
            grad_norm = net.clip_by_global_norm(max_norm=args.max_norm)
            net.update()
            outputs = net.get_outputs(merge_multi_context=False)
            # Accumulate the loss values
            for exe_id in range(len(outputs[0])):
                total_loss += outputs[0][exe_id].asscalar()
            # Update the state of the LSTM (Truncated BPTT)
            for out_id in range(1, len(outputs)):
                for exe_id in range(exe_num):
                    curr_states[out_id - 1][exe_id] = outputs[out_id][exe_id]
            net.set_states(states=curr_states)
            if i % args.disp_batches == 0 and i > 0:
                elapsed = time.time() - start_time
                total_loss = total_loss / (args.disp_batches * args.batch_size * args.bptt)
                logging.info("Epoch:[%d], Batch: [%d]/[%d], lr: %f, nll: %g, ppl: %g, grad_norm: %g,"
                                " ms/batch: %5.2f"
                                %(epoch, i, total_batch_num, net._optimizer.lr,
                                  total_loss, np.exp(total_loss),
                                  grad_norm, elapsed * 1000 / args.disp_batches))
                total_loss = 0
                start_time = time.time()
        valid_avg_nll = evaluate(test_net=test_net, test_data=valid_data, args=args)
        test_avg_nll = evaluate(test_net=test_net, test_data=test_data, args=args)
        logging.info("Epoch:[%d], valid nll: %g, valid ppl: %g"
                     % (epoch, valid_avg_nll, np.exp(valid_avg_nll)))
        logging.info("Epoch:[%d], test nll: %g, test ppl: %g"
                     % (epoch, test_avg_nll, np.exp(test_avg_nll)))
        if (epoch + 1) % args.decay_epoch_num == 0:
            net._optimizer.lr *= args.lr_decay


if __name__ == '__main__':
    import logging

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    args = parse_args()
    if args.gpus:
        contexts = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        contexts = mx.cpu(0)
    if args.mode == "train":
        train(args, contexts)
    elif args.mode == "test":
        raise NotImplementedError()
