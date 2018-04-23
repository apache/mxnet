
# Model parallelism
- This is model parallelized version of http://gluon.mxnet.io/chapter05_recurrent-neural-networks/rnns-gluon.html.
- Similar to https://mxnet.incubator.apache.org/faq/model_parallel_lstm.html.


```python
import math
import os
import time
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn
import collections
```


```python
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
```


```python
class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(path + 'train.txt')
        self.valid = self.tokenize(path + 'valid.txt')
        self.test = self.tokenize(path + 'test.txt')

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = np.zeros((tokens,), dtype='int32')
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return mx.nd.array(ids, dtype='int32')
```

`MultiGPULSTM` creates stacked LSTM with layers spread across multiple GPUs. 
For example, `MultiGPULSTM(0, [1, 2, 2, 1], 400, 200, 0.5)` will create a stacked LSTM with one layer on GPU(0), two layers on GPU(1), two layers on GPU(2), one layer on GPU(3) with a hidden size of 400 embedding size of 200 and dropout probability of .5.


```python
class MultiGPULSTM(object):
    
    def __init__(self, start_device, num_layers_list, num_hidden, input_size, dropout):
        """Create a MultiGPULSTM. num_layers_list dictates how many layers of LSTM
        gets places in which device. For example, [1, 2, 2, 1] will create a stacked LSTM
        with one layer on GPU(0), two layers on GPU(1), two layers on GPU(2), one layer on GPU(3)"""
        self.lstm_dict = collections.OrderedDict()
        device_index = start_device
        self.trainers = []
        
        for num_layers in num_layers_list:
            lstm = gluon.rnn.LSTM(num_hidden, num_layers, dropout=dropout, input_size=input_size)
            input_size = num_hidden
            self.lstm_dict[device_index] = lstm
            device_index += 1
        
    def begin_state(self, *args, **kwargs):
        """Return a list of hidden state for each LSTM in the stack"""
        return [lstm.begin_state(ctx=mx.gpu(gpu_num), *args, **kwargs) 
                for gpu_num, lstm in self.lstm_dict.items()]
    
    def forward(self, inputs, hidden):
        """Pass the data through all LSTM in the stack
        copying intermediate outputs to other contexts as necessary"""
        hidden_indx = 0

        output = inputs
        for gpu_num, lstm in self.lstm_dict.items():
            next_input = output.as_in_context(mx.gpu(gpu_num))
            output, hidden[hidden_indx] = lstm(next_input, hidden[hidden_indx])
            hidden_indx += 1
        
        return output, hidden
    
    def init_params(self, init=mx.init.Xavier(), force_reinit=False):
        """For each LSTM in the stack,
        initialize its parameters in the context specified by num_layers_list"""
        for gpu_num, lstm in self.lstm_dict.items():
            lstm.collect_params().initialize(init, ctx=mx.gpu(gpu_num), force_reinit=force_reinit)
    
    def init_trainer(self, optimizer, optimizer_params=None, kvstore='device'):
        """Create seperate trainer for each LSTM
        since one trainer cannot have parameters from multiple contexts"""
        for gpu_num, lstm in self.lstm_dict.items():
            self.trainers.append(gluon.Trainer(lstm.collect_params(), optimizer, optimizer_params, kvstore))

    def step(self, batch_size, ignore_stale_grad=False):
        """Call step on each LSTM's trainer"""
        for trainer in self.trainers:
            trainer.step(batch_size, ignore_stale_grad)

    def clip_global_norm(self, max_norm):
        """Clip gradients for each LSTM"""
        for gpu_num, lstm in self.lstm_dict.items():
            grads = [i.grad(mx.gpu(gpu_num)) for i in lstm.collect_params().values()]
            gluon.utils.clip_global_norm(grads, max_norm)
            
    def reset_optimizer(self, optimizer, optimizer_params=None):
        """Used to change learning rate. Not used tight now."""
        for trainer in self.trainers:
            trainer._init_optimizer(optimizer, optimizer_params)
```

`LSTMModel` adds an encoder in the beginning and decoder at the end to a `MultiGPULSTM`


```python
class LSTMModel():
    def __init__(self, vocab_size, embedding_size, num_hidden,
                 num_layers_list, dropout=0.5, **kwargs):
        self.encoder = nn.Embedding(vocab_size, embedding_size,
                                    weight_initializer = mx.init.Uniform(0.1))
        self.lstm = MultiGPULSTM(0, num_layers_list, num_hidden, embedding_size, dropout)
        self.decoder = nn.Dense(vocab_size, in_units = num_hidden)
        self.dropout = nn.Dropout(dropout)
        self.num_hidden = num_hidden
        self.num_layers_list = num_layers_list
        
    def forward(self, inputs, hidden):
        embedding = self.encoder(inputs)
        embedding = self.dropout(embedding)
        
        output, hidden = self.lstm.forward(embedding, hidden)
        output = self.dropout(output)

        decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden
    
    def begin_state(self, *args, **kwargs):
        return self.lstm.begin_state(*args, **kwargs)
    
    def init_params(self, init=mx.init.Xavier(), force_reinit=False):
        self.encoder.collect_params().initialize(init, ctx=mx.gpu(0), force_reinit=force_reinit)
        self.lstm.init_params(init, force_reinit)
        last_gpu = len(self.num_layers_list) - 1
        self.decoder.collect_params().initialize(init, ctx=mx.gpu(last_gpu), force_reinit=force_reinit)
    
    def init_trainer(self, optimizer, optimizer_params=None, kvstore='device'):
        self.encoder_trainer = gluon.Trainer(self.encoder.collect_params(), optimizer, optimizer_params, kvstore)
        self.decoder_trainer = gluon.Trainer(self.decoder.collect_params(), optimizer, optimizer_params, kvstore)
        self.lstm.init_trainer(optimizer, optimizer_params, kvstore)

    def step(self, batch_size, ignore_stale_grad=False):
        self.encoder_trainer.step(batch_size, ignore_stale_grad)
        self.decoder_trainer.step(batch_size, ignore_stale_grad)
        self.lstm.step(batch_size, ignore_stale_grad)

    def clip_global_norm(self, max_norm):
        self.lstm.clip_global_norm(max_norm)
    
    def reset_optimizer(self, optimizer, optimizer_params=None):
        self.encoder_trainer._init_optimizer(optimizer, optimizer_params)
        self.decoder_trainer._init_optimizer(optimizer, optimizer_params)
        self.lstm.reset_optimizer(optimizer, optimizer_params)
```


```python
args_data = 'data/ptb.'
args_model = 'lstm'
args_emsize = 200
args_nhid = 400
args_lr = 0.01
args_clip = 0.2
args_epochs = 1
args_batch_size = 32
args_bptt = 6
args_dropout = 0.2
args_log_interval = 100
args_save = 'model.param'
```


```python
corpus = Corpus(args_data)

def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data

train_data = batchify(corpus.train, args_batch_size)
val_data = batchify(corpus.valid, args_batch_size)
test_data = batchify(corpus.test, args_batch_size)
```

In this example we will create a stacked LSTM with two layers on two GPUs.


```python
num_layers_list = [1, 1]
ctx_begin = mx.gpu(0)
ctx_end = mx.gpu(len(num_layers_list) - 1)
```


```python
ntokens = len(corpus.dictionary)

model = LSTMModel(ntokens, args_emsize, args_nhid, num_layers_list, args_dropout)
model.init_params()
model.init_trainer('adadelta', {'learning_rate': args_lr})

loss = gluon.loss.SoftmaxCrossEntropyLoss()
```


```python
def get_batch(source, i):
    seq_len = min(args_bptt, source.shape[0] - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len]
    return data, target.reshape((-1,))

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(i) for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden
```


```python
def eval(data_source):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size)
    for i in range(0, data_source.shape[0] - 1, args_bptt):
        data, target = get_batch(data_source, i)
        data = data.as_in_context(ctx_begin)
        target = target.as_in_context(ctx_end)
        output, hidden = model.forward(data, hidden)
        L = loss(output, target)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal
```


```python
def train():
    for epoch in range(args_epochs):
        total_L = 0.0
        start_time = time.time()
        hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size)
        for ibatch, i in enumerate(range(0, train_data.shape[0] - 1, args_bptt)):
            data, target = get_batch(train_data, i)
            data = data.as_in_context(ctx_begin)
            target = target.as_in_context(ctx_end)
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model.forward(data, hidden)
                L = loss(output, target)
                L.backward()

            model.clip_global_norm(args_clip * args_bptt * args_batch_size)

            model.step(args_batch_size)
            total_L += mx.nd.sum(L).asscalar()

            if ibatch % args_log_interval == 0 and ibatch > 0:
                cur_L = total_L / args_bptt / args_batch_size / args_log_interval
                print('[Epoch %d Batch %d] loss %.2f, perplexity %.2f' % (
                    epoch + 1, ibatch, cur_L, math.exp(cur_L)))
                total_L = 0.0

        val_L = eval(val_data)

        print('[Epoch %d] time cost %.2fs, validation loss %.2f, validation perplexity %.2f' % (
            epoch + 1, time.time() - start_time, val_L, math.exp(val_L)))

```


```python
train()
```

    [Epoch 1 Batch 100] loss 7.20, perplexity 1340.10
    [Epoch 1 Batch 200] loss 6.66, perplexity 781.10
    [Epoch 1 Batch 300] loss 6.65, perplexity 776.09
    [Epoch 1 Batch 400] loss 6.55, perplexity 697.08
    [Epoch 1 Batch 500] loss 6.46, perplexity 640.04
    [Epoch 1 Batch 600] loss 6.38, perplexity 590.49
    [Epoch 1 Batch 700] loss 6.36, perplexity 577.39
    [Epoch 1 Batch 800] loss 6.22, perplexity 501.58
    [Epoch 1 Batch 900] loss 6.10, perplexity 443.65
    [Epoch 1 Batch 1000] loss 6.04, perplexity 418.13
    [Epoch 1 Batch 1100] loss 6.15, perplexity 469.13
    [Epoch 1 Batch 1200] loss 6.10, perplexity 446.42
    [Epoch 1 Batch 1300] loss 6.08, perplexity 435.85
    [Epoch 1 Batch 1400] loss 6.05, perplexity 424.83
    [Epoch 1 Batch 1500] loss 6.01, perplexity 406.97
    [Epoch 1 Batch 1600] loss 6.02, perplexity 410.54
    [Epoch 1 Batch 1700] loss 5.98, perplexity 395.66
    [Epoch 1 Batch 1800] loss 5.98, perplexity 396.62
    [Epoch 1 Batch 1900] loss 5.81, perplexity 333.70
    [Epoch 1 Batch 2000] loss 5.88, perplexity 356.13
    [Epoch 1 Batch 2100] loss 5.96, perplexity 386.80
    [Epoch 1 Batch 2200] loss 5.85, perplexity 347.08
    [Epoch 1 Batch 2300] loss 5.78, perplexity 323.07
    [Epoch 1 Batch 2400] loss 5.76, perplexity 318.46
    [Epoch 1 Batch 2500] loss 5.74, perplexity 311.50
    [Epoch 1 Batch 2600] loss 5.79, perplexity 328.09
    [Epoch 1 Batch 2700] loss 5.79, perplexity 328.51
    [Epoch 1 Batch 2800] loss 5.86, perplexity 350.40
    [Epoch 1 Batch 2900] loss 5.73, perplexity 309.27
    [Epoch 1 Batch 3000] loss 5.77, perplexity 319.66
    [Epoch 1 Batch 3100] loss 5.65, perplexity 285.61
    [Epoch 1 Batch 3200] loss 5.63, perplexity 279.74
    [Epoch 1 Batch 3300] loss 5.62, perplexity 275.63
    [Epoch 1 Batch 3400] loss 5.57, perplexity 263.28
    [Epoch 1 Batch 3500] loss 5.59, perplexity 267.68
    [Epoch 1 Batch 3600] loss 5.64, perplexity 281.89
    [Epoch 1 Batch 3700] loss 5.71, perplexity 301.17
    [Epoch 1 Batch 3800] loss 5.68, perplexity 293.86
    [Epoch 1 Batch 3900] loss 5.69, perplexity 296.83
    [Epoch 1 Batch 4000] loss 5.67, perplexity 289.74
    [Epoch 1 Batch 4100] loss 5.48, perplexity 240.49
    [Epoch 1 Batch 4200] loss 5.64, perplexity 281.97
    [Epoch 1 Batch 4300] loss 5.61, perplexity 273.13
    [Epoch 1 Batch 4400] loss 5.64, perplexity 281.21
    [Epoch 1 Batch 4500] loss 5.65, perplexity 283.48
    [Epoch 1 Batch 4600] loss 5.63, perplexity 278.12
    [Epoch 1 Batch 4700] loss 5.68, perplexity 291.96
    [Epoch 1 Batch 4800] loss 5.59, perplexity 267.57
    [Epoch 1] time cost 267.24s, validation loss 5.63, validation perplexity 279.41



```python
test_L = eval(test_data)
print('Best test loss %.2f, test perplexity %.2f'%(test_L, math.exp(test_L)))
```

    Best test loss 5.61, test perplexity 273.09

