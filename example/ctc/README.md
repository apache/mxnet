# CTC with Mxnet

## Overview
This example is a modification of [warpctc](https://github.com/dmlc/mxnet/tree/master/example/warpctc)
It demonstrates the usage of  ```mx.contrib.sym.ctc_loss``` 

## Core code change

The following implementation of ```lstm_unroll```  function is introduced in ```lstm.py``` demonstrates the usage of
```mx.contrib.sym.ctc_loss```.

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

## Prerequisites

Please ensure that following prerequisites are satisfied before running this examples.

- ```captcha``` python package is installed.
- ```cv2``` (or ```openCV```) python package is installed.
- The test requires font file (```ttf``` format). The user either would need to create ```.\data\```  directory and place the font file in that directory. The user can also edit following line to specify path to the font file.
```cython
        # you can get this font from http://font.ubuntu.com/
        self.captcha = ImageCaptcha(fonts=['./data/Xerox.ttf'])
```

## How to run

The users would need to run the script ```lstm_ocr.py``` in order to exercise the above code change.
```cython
python lstm_ocr.py
``` 

## Further reading

In order to run the ```ocr_predict.py```  please refer to [ReadMe](https://github.com/apache/incubator-mxnet/blob/master/example/warpctc/README.md) file in [warpctc](https://github.com/dmlc/mxnet/tree/master/example/warpctc)
