import re
import sys
sys.path.insert(0, "../../python")
import time
import logging
import os.path

import mxnet as mx
import numpy as np

from lstm_proj import lstm_unroll
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
    num_hidden_proj = args.config.getint('arch', 'num_hidden_proj')
    num_lstm_layer = args.config.getint('arch', 'num_lstm_layer')

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    if num_hidden_proj > 0:
        init_h = [('l%d_init_h'%l, (batch_size, num_hidden_proj)) for l in range(num_lstm_layer)]
    else:
        init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]

    init_states = init_c + init_h

    file_test = args.config.get('data', 'test')
    file_label_mean =  args.config.get('data', 'label_mean')
    file_format = args.config.get('data', 'format')
    feat_dim = args.config.getint('data', 'xdim')
    label_dim = args.config.getint('data', 'ydim')

    test_data_args = {
            "gpu_chunk": 32768,
            "lst_file": file_test,
            "file_format": file_format,
            "separate_lines":True,
            "has_labels":False
            }

    label_mean_args = {
            "gpu_chunk": 32768,
            "lst_file": file_label_mean,
            "file_format": file_format,
            "separate_lines":True,
            "has_labels":False
            }

    test_sets = DataReadStream(test_data_args, feat_dim)
    label_mean_sets = DataReadStream(label_mean_args, label_dim)
    return (init_states, test_sets, label_mean_sets)


if __name__ == '__main__':
    args = parse_args()
    args.config.write(sys.stderr)

    decoding_method = args.config.get('train', 'method')
    contexts = parse_contexts(args)

    init_states, test_sets, label_mean_sets = prepare_data(args)
    state_names = [x[0] for x in init_states]

    batch_size = args.config.getint('train', 'batch_size')
    num_hidden = args.config.getint('arch', 'num_hidden')
    num_hidden_proj = args.config.getint('arch', 'num_hidden_proj')
    num_lstm_layer = args.config.getint('arch', 'num_lstm_layer')
    feat_dim = args.config.getint('data', 'xdim')
    label_dim = args.config.getint('data', 'ydim')
    out_file = args.config.get('data', 'out_file')
    num_epoch = args.config.getint('train', 'num_epoch')
    model_name = get_checkpoint_path(args)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')
    
    # load the model
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, num_epoch)

    if decoding_method == METHOD_BUCKETING:
        buckets = args.config.get('train', 'buckets')
        buckets = list(map(int, re.split(r'\W+', buckets)))
        data_test   = BucketSentenceIter(test_sets, buckets, batch_size, init_states, feat_dim=feat_dim, has_label=False)
        def sym_gen(seq_len):
            sym = lstm_unroll(num_lstm_layer, seq_len, feat_dim, num_hidden=num_hidden, 
                              num_label=label_dim, take_softmax=True, num_hidden_proj=num_hidden_proj)
            data_names = ['data'] + state_names
            label_names = ['softmax_label']
            return (sym, data_names, label_names)

        module = mx.mod.BucketingModule(sym_gen,
                            default_bucket_key=data_test.default_bucket_key,
                            context=contexts)
    elif decoding_method == METHOD_SIMPLE:
        data_test = SimpleIter(test_sets, batch_size, init_states, feat_dim=feat_dim, label_dim=label_dim,
                label_mean_sets=label_mean_sets, has_label=False)
        def sym_gen(seq_len):
            sym = lstm_unroll(num_lstm_layer, seq_len, feat_dim, num_hidden=num_hidden, 
                              num_label=label_dim, take_softmax=False, num_hidden_proj=num_hidden_proj)
            data_names = ['data'] + state_names
            label_names = []
            return (sym, data_names, label_names)

        module = mx.mod.BucketingModule(sym_gen,
                            default_bucket_key=data_test.default_bucket_key,
                            context=contexts)

    else:
        truncate_len=20
        data_test = TruncatedSentenceIter(test_sets, batch_size, init_states,
                                         truncate_len, feat_dim=feat_dim,
                                         do_shuffling=False, pad_zeros=True, has_label=True)

        sym = lstm_unroll(num_lstm_layer, truncate_len, feat_dim, num_hidden=num_hidden,
                          num_label=label_dim, output_states=True, num_hidden_proj=num_hidden_proj)
        data_names = [x[0] for x in data_test.provide_data]
        label_names = ['softmax_label']
        module = mx.mod.Module(sym, context=contexts, data_names=data_names,
                               label_names=label_names)
    # set the parameters
    module.bind(data_shapes=data_test.provide_data, label_shapes=None, for_training=False)
    module.set_params(arg_params=arg_params, aux_params=aux_params)
    
    kaldiWriter = KaldiWriteOut(None, out_file)
    kaldiWriter.open_or_fd()

    for preds, i_batch, batch in module.iter_predict(data_test):
        #pred_label = np.array(preds[0].asnumpy().argmax(axis=1))
        label = batch.label[0].asnumpy().astype('int32')
        posteriors = preds[0].asnumpy().astype('float32')[0]
        #print np.sum(posteriors[1][:])
        # copy over states
        if decoding_method == METHOD_BUCKETING:
            for (ind, utt) in enumerate(batch.utt_id):
                if utt != "GAP_UTT":
                    #print sum(posteriors[0,:])
                    posteriors = np.log(posteriors[:label[0][0],1:] + 1e-20) - np.log(data_train.label_mean).T
                    kaldiWriter.write(utt, posteriors)
        elif decoding_method == METHOD_SIMPLE:
            for (ind, utt) in enumerate(batch.utt_id):
                if utt != "GAP_UTT":
                    #print label[0][0]
                    posteriors = posteriors[:batch.utt_len,1:] - np.log(data_test.label_mean[1:]).T
                    kaldiWriter.write(utt, posteriors)
        else:
            outputs = module.get_outputs()
            # outputs[0] is softmax, 1:end are states
            for i in range(1, len(outputs)):
                outputs[i].copyto(data_test.init_state_arrays[i-1])
            for (ind, utt) in enumerate(batch.utt_id):
                if utt != "GAP_UTT":
                    posteriors = np.log(posteriors[:,1:])# - np.log(data_train.label_mean).T
                    kaldiWriter.write(utt, posteriors)


    kaldiWriter.close()
    args.config.write(sys.stderr)

