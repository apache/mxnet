import re
import sys
sys.path.insert(0, "../../python")
import time
import logging
import os.path

import mxnet as mx
import numpy as np

from lstm import lstm_unroll
from io_util import BucketSentenceIter, TruncatedSentenceIter, SimpleIter, DataReadStream
from config_util import parse_args, get_checkpoint_path, parse_contexts

from io_func.feat_readers.writer_kaldi import KaldiWriteOut

# some constants
METHOD_BUCKETING = 'bucketing'
METHOD_TBPTT = 'truncated-bptt'
METHOD_SIMPLE = 'simple'


def prepare_data(args):
    batch_size = args.config.getint('train', 'batch_size')
    num_hidden = args.config.getint('arch', 'num_hidden')
    num_lstm_layer = args.config.getint('arch', 'num_lstm_layer')

    init_c = [('l%d_init_c' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]

    init_states = init_c + init_h

    file_test = args.config.get('data', 'train')

    file_format = args.config.get('data', 'format')
    feat_dim = args.config.getint('data', 'xdim')

    test_data_args = {
            "gpu_chunk": 32768,
            "lst_file": file_test,
            "file_format": file_format,
            "separate_lines": True,
            "has_labels": True
            }

    test_sets = DataReadStream(test_data_args, feat_dim)

    return (init_states, test_sets)


if __name__ == '__main__':
    args = parse_args()
    args.config.write(sys.stderr)

    decoding_method = args.config.get('train', 'method')
    contexts = parse_contexts(args)

    init_states, test_sets = prepare_data(args)
    state_names = [x[0] for x in init_states]

    batch_size = args.config.getint('train', 'batch_size')
    num_hidden = args.config.getint('arch', 'num_hidden')
    num_lstm_layer = args.config.getint('arch', 'num_lstm_layer')
    feat_dim = args.config.getint('data', 'xdim')
    label_dim = args.config.getint('data', 'ydim')
    out_file = args.config.get('data', 'out_file')
    num_epoch = args.config.getint('train', 'num_epoch')
    model_name = get_checkpoint_path(args)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

    # load the model
    label_mean = np.zeros((label_dim,1), dtype='float32')
    data_test = TruncatedSentenceIter(test_sets, batch_size, init_states,
                                         20, feat_dim=feat_dim,
                                         do_shuffling=False, pad_zeros=True, has_label=True)

    for i, batch in enumerate(data_test.labels):
        hist, edges = np.histogram(batch.flat, bins=range(0,label_dim+1))
        label_mean += hist.reshape(label_dim,1)

    kaldiWriter = KaldiWriteOut(None, out_file)
    kaldiWriter.open_or_fd()
    kaldiWriter.write("label_mean", label_mean)


    args.config.write(sys.stderr)
