# CTC with Mxnet
this is mx.contrib.sym.ctc_loss example. It was modified from example [warpctc](https://github.com/dmlc/mxnet/tree/master/example/warpctc) 

# Core code
this is core change in lstm.py
```Cython
def lstm_unroll(num_lstm_layer, seq_len,
                num_hidden, num_label):
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert (len(last_states) == num_lstm_layer)

    # embeding layer
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    wordvec = mx.sym.SliceChannel(data=data, num_outputs=seq_len, squeeze_axis=1)

    hidden_all = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]
        for i in range(num_lstm_layer):
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i)
            hidden = next_state.h
            last_states[i] = next_state
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)

    pred_fc = mx.sym.FullyConnected(data=hidden_concat, num_hidden=11)
    pred_ctc = mx.sym.Reshape(data=pred_fc, shape=(-4, seq_len, -1, 0))

    loss = mx.contrib.sym.ctc_loss(data=pred_ctc, label=label)
    ctc_loss = mx.sym.MakeLoss(loss)

    softmax_class = mx.symbol.SoftmaxActivation(data=pred_fc)
    softmax_loss = mx.sym.MakeLoss(softmax_class)
    softmax_loss = mx.sym.BlockGrad(softmax_loss)

    return mx.sym.Group([softmax_loss, ctc_loss])
```
# Some Result
If there were more training, the result would be better

```
2017-07-08 13:22:01,155 Epoch[94] Batch [50]    Speed: 4273.43 samples/sec	Accuracy=0.808747
2017-07-08 13:22:13,141 Epoch[94] Batch [100]	Speed: 4271.84 samples/sec	Accuracy=0.786855
2017-07-08 13:22:25,179 Epoch[94] Batch [150]	Speed: 4253.81 samples/sec	Accuracy=0.810625
2017-07-08 13:22:37,198 Epoch[94] Batch [200]	Speed: 4259.96 samples/sec	Accuracy=0.808809
2017-07-08 13:22:49,233 Epoch[94] Batch [250]	Speed: 4254.13 samples/sec	Accuracy=0.806426
2017-07-08 13:23:01,308 Epoch[94] Batch [300]	Speed: 4239.98 samples/sec	Accuracy=0.817305
2017-07-08 13:23:02,030 Epoch[94] Train-Accuracy=0.819336
2017-07-08 13:23:02,030 Epoch[94] Time cost=73.092
2017-07-08 13:23:02,101 Saved checkpoint to "ocr-0095.params"
2017-07-08 13:23:07,192 Epoch[94] Validation-Accuracy=0.819417
2017-07-08 13:23:20,579 Epoch[95] Batch [50]	Speed: 4288.76 samples/sec	Accuracy=0.817459
2017-07-08 13:23:32,573 Epoch[95] Batch [100]	Speed: 4268.75 samples/sec	Accuracy=0.815215
2017-07-08 13:23:44,635 Epoch[95] Batch [150]	Speed: 4244.85 samples/sec	Accuracy=0.820215
2017-07-08 13:23:56,670 Epoch[95] Batch [200]	Speed: 4254.38 samples/sec	Accuracy=0.823613
2017-07-08 13:24:08,650 Epoch[95] Batch [250]	Speed: 4273.83 samples/sec	Accuracy=0.827109
2017-07-08 13:24:20,680 Epoch[95] Batch [300]	Speed: 4256.49 samples/sec	Accuracy=0.824961
2017-07-08 13:24:21,401 Epoch[95] Train-Accuracy=0.840495
2017-07-08 13:24:21,401 Epoch[95] Time cost=73.008
2017-07-08 13:24:21,441 Saved checkpoint to "ocr-0096.params"
2017-07-08 13:24:26,508 Epoch[95] Validation-Accuracy=0.834798
2017-07-08 13:24:39,938 Epoch[96] Batch [50]	Speed: 4259.32 samples/sec	Accuracy=0.825578
2017-07-08 13:24:51,987 Epoch[96] Batch [100]	Speed: 4249.67 samples/sec	Accuracy=0.826562
2017-07-08 13:25:04,041 Epoch[96] Batch [150]	Speed: 4247.44 samples/sec	Accuracy=0.831855
2017-07-08 13:25:16,058 Epoch[96] Batch [200]	Speed: 4260.77 samples/sec	Accuracy=0.830840
2017-07-08 13:25:28,109 Epoch[96] Batch [250]	Speed: 4248.44 samples/sec	Accuracy=0.827168
2017-07-08 13:25:40,057 Epoch[96] Batch [300]	Speed: 4285.23 samples/sec	Accuracy=0.832715
2017-07-08 13:25:40,782 Epoch[96] Train-Accuracy=0.830729
2017-07-08 13:25:40,782 Epoch[96] Time cost=73.098
2017-07-08 13:25:40,821 Saved checkpoint to "ocr-0097.params"
2017-07-08 13:25:45,886 Epoch[96] Validation-Accuracy=0.840820
2017-07-08 13:25:59,283 Epoch[97] Batch [50]	Speed: 4271.85 samples/sec	Accuracy=0.831648
2017-07-08 13:26:11,243 Epoch[97] Batch [100]	Speed: 4280.89 samples/sec	Accuracy=0.835371
2017-07-08 13:26:23,263 Epoch[97] Batch [150]	Speed: 4259.89 samples/sec	Accuracy=0.831094
2017-07-08 13:26:35,230 Epoch[97] Batch [200]	Speed: 4278.40 samples/sec	Accuracy=0.827129
2017-07-08 13:26:47,199 Epoch[97] Batch [250]	Speed: 4277.77 samples/sec	Accuracy=0.834258
2017-07-08 13:26:59,257 Epoch[97] Batch [300]	Speed: 4245.93 samples/sec	Accuracy=0.833770
2017-07-08 13:26:59,971 Epoch[97] Train-Accuracy=0.844727
2017-07-08 13:26:59,971 Epoch[97] Time cost=72.908
2017-07-08 13:27:00,020 Saved checkpoint to "ocr-0098.params"
2017-07-08 13:27:05,130 Epoch[97] Validation-Accuracy=0.827962
2017-07-08 13:27:18,521 Epoch[98] Batch [50]	Speed: 4281.06 samples/sec	Accuracy=0.834118
2017-07-08 13:27:30,537 Epoch[98] Batch [100]	Speed: 4261.20 samples/sec	Accuracy=0.835352
2017-07-08 13:27:42,542 Epoch[98] Batch [150]	Speed: 4264.88 samples/sec	Accuracy=0.839395
2017-07-08 13:27:54,544 Epoch[98] Batch [200]	Speed: 4266.31 samples/sec	Accuracy=0.836328
2017-07-08 13:28:06,550 Epoch[98] Batch [250]	Speed: 4264.50 samples/sec	Accuracy=0.841465
2017-07-08 13:28:18,622 Epoch[98] Batch [300]	Speed: 4241.11 samples/sec	Accuracy=0.831680
2017-07-08 13:28:19,349 Epoch[98] Train-Accuracy=0.833984
2017-07-08 13:28:19,349 Epoch[98] Time cost=73.018
2017-07-08 13:28:19,393 Saved checkpoint to "ocr-0099.params"
2017-07-08 13:28:24,472 Epoch[98] Validation-Accuracy=0.818034
2017-07-08 13:28:37,961 Epoch[99] Batch [50]	Speed: 4242.14 samples/sec	Accuracy=0.835861
2017-07-08 13:28:50,031 Epoch[99] Batch [100]	Speed: 4241.94 samples/sec	Accuracy=0.846543
2017-07-08 13:29:02,108 Epoch[99] Batch [150]	Speed: 4239.22 samples/sec	Accuracy=0.850645
2017-07-08 13:29:14,160 Epoch[99] Batch [200]	Speed: 4248.34 samples/sec	Accuracy=0.844141
2017-07-08 13:29:26,225 Epoch[99] Batch [250]	Speed: 4243.71 samples/sec	Accuracy=0.842129
2017-07-08 13:29:38,277 Epoch[99] Batch [300]	Speed: 4248.07 samples/sec	Accuracy=0.851250
2017-07-08 13:29:38,975 Epoch[99] Train-Accuracy=0.854492
2017-07-08 13:29:38,976 Epoch[99] Time cost=73.315
2017-07-08 13:29:39,023 Saved checkpoint to "ocr-0100.params"
2017-07-08 13:29:44,110 Epoch[99] Validation-Accuracy=0.851969
```
