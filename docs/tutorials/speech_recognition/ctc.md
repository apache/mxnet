<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Connectionist Temporal Classification

```python

import mxnet as mx
print(mx.__version__)
```

`1.1.0`<!--notebook-skip-line-->


[Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) (CTC) is a cost function that is used to train Recurrent Neural Networks (RNNs) to label unsegmented input sequence data in supervised learning. For example, in a speech recognition application, using a typical cross-entropy loss, the input signal needs to be segmented into words or sub-words. However, using CTC-loss, it suffices to provide one label sequence for input sequence and the network learns both the alignment as well labeling. Baidu's warp-ctc page contains a more detailed [introduction to CTC-loss](https://github.com/baidu-research/warp-ctc#introduction).

## CTC-loss in MXNet
MXNet supports two CTC-loss layers in Symbol API:

* `mxnet.symbol.contrib.ctc_loss` is implemented in MXNet and included as part of the standard package.
* `mxnet.symbol.WarpCTC` uses Baidu's warp-ctc library and requires building warp-ctc library and mxnet library both from source.

## LSTM OCR Example
MXNet's example folder contains a [CTC example](https://github.com/apache/incubator-mxnet/tree/master/example/ctc) for using CTC loss with an LSTM network to perform Optical Character Recognition (OCR) prediction on CAPTCHA images. The example demonstrates use of both CTC loss options, as well as inference after training using network symbol and parameter checkpoints.

## Next Steps
* [MXNet tutorials index](http://mxnet.io/tutorials/index.html)

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->