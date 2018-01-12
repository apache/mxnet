# Connectionist Temporal Classification

[Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) (CTC) is a cost function that is used to train Recurrent Neural Networks (RNNs) to label unsegmented input sequence data in supervised learning. For example, in a speech recognition application, using a typical cross-entropy loss, the input signal needs to be segmented into words or sub-words. However, using CTC-loss, it suffices to provide one label sequence for input sequence and the network learns both the alignment as well labeling. Baidu's warp-ctc page contains a more detailed [introduction to CTC-loss](https://github.com/baidu-research/warp-ctc#introduction).

## CTC-loss in MXNet
MXNet supports two CTC-loss layers in Symbol API:

* `mxnet.symbol.contrib.ctc_loss` is implemented in MXNet and included as part of the standard package.
* `mxnet.symbol.WarpCTC` uses Baidu's warp-ctc library and requires building warp-ctc library and mxnet library both from source.

## LSTM OCR Example
MXNet's example folder contains a [CTC example](https://github.com/apache/incubator-mxnet/tree/master/example/ctc) for using CTC loss with an LSTM network to perform Optical Character Recognition (OCR) prediction on CAPTCHA images. The example demonstrates use of both CTC loss options, as well as inference after training using network symbol and parameter checkpoints.

## Next Steps
* [MXNet tutorials index](http://mxnet.io/tutorials/index.html)
