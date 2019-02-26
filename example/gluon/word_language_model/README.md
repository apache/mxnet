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

# Word-level language modeling RNN

This example trains a multi-layer RNN (Elman, GRU, or LSTM) on WikiText-2 language modeling benchmark.

The model obtains ~107 ppl in WikiText-2 using LSTM.

The following techniques have been adopted for SOTA results:
- [LSTM for LM](https://arxiv.org/pdf/1409.2329.pdf)
- [Weight tying](https://arxiv.org/abs/1608.05859) between word vectors and softmax output embeddings

## Data

### Wiki Text

The wikitext-2 data is from [(The wikitext long term dependency language modeling dataset)](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/). The training script automatically loads the dataset into `$PWD/data`.


## Usage

Example runs and the results:

```
python train.py --cuda --tied --nhid 200 --emsize 200 --epochs 20  --dropout 0.2        # Test ppl of 107.49
```
```
python train.py --cuda --tied --nhid 650 --emsize 650 --epochs 40  --dropout 0.5        # Test ppl of 91.51
```
```
python train.py --cuda --tied --nhid 1500 --emsize 1500 --epochs 60  --dropout 0.65     # Test ppl of 88.42
```
```
python train.py --export-model # hybridize and export model graph. See below for visualization options.
```

<br>

`python train.py --help` gives the following arguments:
```
usage: train.py [-h] [--model MODEL] [--emsize EMSIZE] [--nhid NHID]
                [--nlayers NLAYERS] [--lr LR] [--clip CLIP] [--epochs EPOCHS]
                [--batch_size N] [--bptt BPTT] [--dropout DROPOUT] [--tied]
                [--cuda] [--log-interval N] [--save SAVE] [--gctype GCTYPE]
                [--gcthreshold GCTHRESHOLD] [--hybridize] [--static-alloc]
                [--static-shape] [--export-model]

MXNet Autograd RNN/LSTM Language Model on Wikitext-2.

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --bptt BPTT           sequence length
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --cuda                Whether to use gpu
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --gctype GCTYPE       type of gradient compression to use, takes `2bit` or
                        `none` for now.
  --gcthreshold GCTHRESHOLD
                        threshold for 2bit gradient compression
  --hybridize           whether to hybridize in mxnet>=1.3 (default=False)
  --static-alloc        whether to use static-alloc hybridize in mxnet>=1.3
                        (default=False)
  --static-shape        whether to use static-shape hybridize in mxnet>=1.3
                        (default=False)
  --export-model        export a symbol graph and exit (default=False)
```

You may visualize the graph with `mxnet.viz.plot_network` without any additional dependencies. Alternatively, if [mxboard](https://github.com/awslabs/mxboard) is installed, use the following approach for interactive visualization.
```python
#!python
import mxnet, mxboard
with mxboard.SummaryWriter(logdir='./model-graph') as sw:
    sw.add_graph(mxnet.sym.load('./model-symbol.json'))
```
```bash
#!/bin/bash
tensorboard --logdir=./model-graph/
```
![model graph](./model-graph.png?raw=true "rnn model graph")
