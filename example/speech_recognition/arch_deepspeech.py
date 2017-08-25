import json
import math

import mxnet as mx

from stt_layer_batchnorm import batchnorm
from stt_layer_conv import conv
from stt_layer_fc import sequence_fc
from stt_layer_gru import bi_gru_unroll, gru_unroll
from stt_layer_lstm import bi_lstm_unroll
from stt_layer_slice import slice_symbol_to_seq_symobls
from stt_layer_warpctc import warpctc_layer


def prepare_data(args):
    rnn_type = args.config.get("arch", "rnn_type")
    num_rnn_layer = args.config.getint("arch", "num_rnn_layer")
    num_hidden_rnn_list = json.loads(args.config.get("arch", "num_hidden_rnn_list"))

    batch_size = args.config.getint("common", "batch_size")

    if rnn_type == 'lstm':
        init_c = [('l%d_init_c' % l, (batch_size, num_hidden_rnn_list[l])) for l in range(num_rnn_layer)]
        init_h = [('l%d_init_h' % l, (batch_size, num_hidden_rnn_list[l])) for l in range(num_rnn_layer)]
    elif rnn_type == 'bilstm':
        forward_init_c = [('forward_l%d_init_c' % l, (batch_size, num_hidden_rnn_list[l])) for l in
                          range(num_rnn_layer)]
        backward_init_c = [('backward_l%d_init_c' % l, (batch_size, num_hidden_rnn_list[l])) for l in
                           range(num_rnn_layer)]
        init_c = forward_init_c + backward_init_c
        forward_init_h = [('forward_l%d_init_h' % l, (batch_size, num_hidden_rnn_list[l])) for l in
                          range(num_rnn_layer)]
        backward_init_h = [('backward_l%d_init_h' % l, (batch_size, num_hidden_rnn_list[l])) for l in
                           range(num_rnn_layer)]
        init_h = forward_init_h + backward_init_h
    elif rnn_type == 'gru':
        init_h = [('l%d_init_h' % l, (batch_size, num_hidden_rnn_list[l])) for l in range(num_rnn_layer)]
    elif rnn_type == 'bigru':
        forward_init_h = [('forward_l%d_init_h' % l, (batch_size, num_hidden_rnn_list[l])) for l in
                          range(num_rnn_layer)]
        backward_init_h = [('backward_l%d_init_h' % l, (batch_size, num_hidden_rnn_list[l])) for l in
                           range(num_rnn_layer)]
        init_h = forward_init_h + backward_init_h
    else:
        raise Exception('network type should be one of the lstm,bilstm,gru,bigru')

    if rnn_type == 'lstm' or rnn_type == 'bilstm':
        init_states = init_c + init_h
    elif rnn_type == 'gru' or rnn_type == 'bigru':
        init_states = init_h
    return init_states


def arch(args):
    mode = args.config.get("common", "mode")
    if mode == "train":
        channel_num = args.config.getint("arch", "channel_num")
        conv_layer1_filter_dim = tuple(json.loads(args.config.get("arch", "conv_layer1_filter_dim")))
        conv_layer1_stride = tuple(json.loads(args.config.get("arch", "conv_layer1_stride")))
        conv_layer2_filter_dim = tuple(json.loads(args.config.get("arch", "conv_layer2_filter_dim")))
        conv_layer2_stride = tuple(json.loads(args.config.get("arch", "conv_layer2_stride")))

        rnn_type = args.config.get("arch", "rnn_type")
        num_rnn_layer = args.config.getint("arch", "num_rnn_layer")
        num_hidden_rnn_list = json.loads(args.config.get("arch", "num_hidden_rnn_list"))

        is_batchnorm = args.config.getboolean("arch", "is_batchnorm")

        seq_len = args.config.getint('arch', 'max_t_count')
        num_label = args.config.getint('arch', 'max_label_length')

        num_rear_fc_layers = args.config.getint("arch", "num_rear_fc_layers")
        num_hidden_rear_fc_list = json.loads(args.config.get("arch", "num_hidden_rear_fc_list"))
        act_type_rear_fc_list = json.loads(args.config.get("arch", "act_type_rear_fc_list"))
        # model symbol generation
        # input preparation
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('label')

        net = mx.sym.Reshape(data=data, shape=(-4, -1, 1, 0, 0))
        net = conv(net=net,
                   channels=channel_num,
                   filter_dimension=conv_layer1_filter_dim,
                   stride=conv_layer1_stride,
                   no_bias=is_batchnorm
                   )
        if is_batchnorm:
            # batch norm normalizes axis 1
            net = batchnorm(net)

        net = conv(net=net,
                   channels=channel_num,
                   filter_dimension=conv_layer2_filter_dim,
                   stride=conv_layer2_stride,
                   no_bias=is_batchnorm
                   )
        if is_batchnorm:
            # batch norm normalizes axis 1
            net = batchnorm(net)
        net = mx.sym.transpose(data=net, axes=(0, 2, 1, 3))
        net = mx.sym.Reshape(data=net, shape=(0, 0, -3))
        seq_len_after_conv_layer1 = int(
            math.floor((seq_len - conv_layer1_filter_dim[0]) / conv_layer1_stride[0])) + 1
        seq_len_after_conv_layer2 = int(
            math.floor((seq_len_after_conv_layer1 - conv_layer2_filter_dim[0]) / conv_layer2_stride[0])) + 1
        net = slice_symbol_to_seq_symobls(net=net, seq_len=seq_len_after_conv_layer2, axis=1)
        if rnn_type == "bilstm":
            net = bi_lstm_unroll(net=net,
                                 seq_len=seq_len_after_conv_layer2,
                                 num_hidden_lstm_list=num_hidden_rnn_list,
                                 num_lstm_layer=num_rnn_layer,
                                 dropout=0.,
                                 is_batchnorm=is_batchnorm)
        elif rnn_type == "gru":
            net = gru_unroll(net=net,
                             seq_len=seq_len_after_conv_layer2,
                             num_hidden_gru_list=num_hidden_rnn_list,
                             num_gru_layer=num_rnn_layer,
                             dropout=0.,
                             is_batchnorm=is_batchnorm)
        elif rnn_type == "bigru":
            net = bi_gru_unroll(net=net,
                                seq_len=seq_len_after_conv_layer2,
                                num_hidden_gru_list=num_hidden_rnn_list,
                                num_gru_layer=num_rnn_layer,
                                dropout=0.,
                                is_batchnorm=is_batchnorm)
        else:
            raise Exception('rnn_type should be one of the followings, bilstm,gru,bigru')

        # rear fc layers
        net = sequence_fc(net=net, seq_len=seq_len_after_conv_layer2, num_layer=num_rear_fc_layers, prefix="rear",
                          num_hidden_list=num_hidden_rear_fc_list, act_type_list=act_type_rear_fc_list,
                          is_batchnorm=is_batchnorm)
        if is_batchnorm:
            hidden_all = []
            # batch norm normalizes axis 1
            for seq_index in range(seq_len_after_conv_layer2):
                hidden = net[seq_index]
                hidden = batchnorm(hidden)
                hidden_all.append(hidden)
            net = hidden_all

        # warpctc layer
        net = warpctc_layer(net=net,
                            seq_len=seq_len_after_conv_layer2,
                            label=label,
                            num_label=num_label,
                            character_classes_count=(args.config.getint('arch', 'n_classes') + 1)
                            )
        args.config.set('arch', 'max_t_count', str(seq_len_after_conv_layer2))
        return net
    else:
        conv_layer1_filter_dim = tuple(json.loads(args.config.get("arch", "conv_layer1_filter_dim")))
        conv_layer1_stride = tuple(json.loads(args.config.get("arch", "conv_layer1_stride")))
        conv_layer2_filter_dim = tuple(json.loads(args.config.get("arch", "conv_layer2_filter_dim")))
        conv_layer2_stride = tuple(json.loads(args.config.get("arch", "conv_layer2_stride")))
        seq_len = args.config.getint('arch', 'max_t_count')
        seq_len_after_conv_layer1 = int(
            math.floor((seq_len - conv_layer1_filter_dim[0]) / conv_layer1_stride[0])) + 1
        seq_len_after_conv_layer2 = int(
            math.floor((seq_len_after_conv_layer1 - conv_layer2_filter_dim[0]) / conv_layer2_stride[0])) + 1
        args.config.set('arch', 'max_t_count', str(seq_len_after_conv_layer2))


