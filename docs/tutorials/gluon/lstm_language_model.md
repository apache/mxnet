
# Language modeling with RNN

This tutorial walkthrough training a word-level language model with multilayer recurrent neural networks. 

Remember that we presented a basic language model with a single embedding and fully-connected layer on [word embedding tutorial](). This tutorial is different on several aspects

1. Multilayer RNN will be used between the input embedding and output fully-connected layer to generate state-of-the-art results.
2. We will batch the examples to speedup the computation
2. We will use GPU to accelerate the training
3. We will train on a real dataset  

Let's first construct the neural network. The data is fed into an embedding layer as before. But now it is assumed to have the shape *(seq_len, batch_size)*: each column consists of *seq_len* continously words, and we put *batch_size* sequences colume by columne. Therefore the output the embedding layer will be *(seq_len, batch_size, embed_size)*. 

Next we apply *seq_len* 

On this model, we first obtain the embedding vectors, *(time, batch,
embed_size)*. Then we apply *time*-step LSTMs to collect the output *(time,
batch, hidden_size)*. It will applied to upper layer LSTM if exists. Finally we
obtain the *(time, batch, vocab_size)* output through a fully-connected layer.



```python
from mxnet import gluon, nd
from mxnet.gluon import nn, rnn

class RNNModel(gluon.Block):
    def __init__(self, vocab_size, embed_size, hidden_size,
                 num_layers, dropout, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, embed_size)
            self.rnn = rnn.LSTM(hidden_size, num_layers, dropout=dropout)
            self.decoder = nn.Dense(vocab_size) 
            
    def forward(self, inputs, hidden):
        embed = self.drop(self.encoder(inputs))
        output, hidden = self.rnn(embed, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape((-1, output.shape[-1])))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```


```python
model = RNNModel(vocab_size=10, embed_size=3, hidden_size=4, num_layers=1, dropout=.5)
model.collect_params().initialize()
hidden = model.begin_state(func=nd.zeros, batch_size=2)
output, _ = model(nd.array([[0,1],[1,2],[2,3]]), hidden)

print(output.shape)
```

## Prepare data

### Download
We first download the PTB dataset provided by Tomas Mikolov and show the first 5 sentences in the training data.


```python
import mxnet as mx
url_root = 'https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/ptb/'
for f in ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']:
    mx.test_utils.download(url_root+f)
with open('ptb.train.txt', 'r') as f:
    for i in range(5):
        print(f.readline().rstrip())
```

### Tokenize
Next we build a dictionary that maps each unique word into a unique integer index. Then convert each file with *n* words into a length-*n* integer vector.


```python
class Dictionary(object):
    """Word <=> Index map"""
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
    
class Corpus(object):
    """Convert each text file into an integer vector"""
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(path + 'train.txt')
        self.valid = self.tokenize(path + 'valid.txt')
        self.test = self.tokenize(path + 'test.txt')

    def tokenize(self, fname):
        """Tokenizes a text file."""
        with open(fname, 'r') as f:
            words = []
            for line in f:
                words += line.split() + ['<eos>']
        ids = [self.dictionary.add_word(w) for w in words]
        return mx.nd.array(ids, dtype='int32')
                    
corpus = Corpus('ptb.')
print(corpus.train.shape)
```

### Make batches




```python
def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data

def get_batch(source, seq_len, i):
    seq_len = min(seq_len, source.shape[0] - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    return data, target.reshape((-1,))

data = mx.nd.arange(20)
batched = batchify(data, batch_size=3)
print(batched)
print(get_batch(batched, seq_len=2, i=1))
print(get_batch(batched, seq_len=10, i=2))

```

## Train



```python
import time
import math
from random import shuffle
from mxnet import autograd
from mxnet.gluon import Trainer, loss
from mxnet.gluon.utils import clip_global_norm

tiny_workload = True

if tiny_workload:
    context = mx.cpu()
    batch_size = 16
    num_embed = 32
    num_hidden = 32
    num_rnn_layers = 1
    seq_len = 10
    num_epochs = 1
    dropout = .5
    learning_rate = 20
    num_batches = 20
    disp_batch=5
else:
    # requires MXNet is compiled with CUDA and cuDNN>=5.1, and there is at least on available gpu
    context = mx.gpu(0)
    # nu
    batch_size = 32
    num_embed = 650
    num_hidden = 650
    num_rnn_layers = 2
    seq_len = 35
    num_epochs = 40
    dropout = .5
    num_batches = None
    disp_batch = 20
    learning_rate = 20



model = RNNModel(len(corpus.dictionary), num_embed, num_hidden, num_rnn_layers, dropout)
model.collect_params().initialize(mx.init.Xavier(), ctx=context)
trainer = Trainer(model.collect_params(), 'sgd',
                      {'learning_rate': learning_rate,
                       'momentum': 0,
                       'wd': 0})
softmax_ce_loss = loss.SoftmaxCrossEntropyLoss()



train_data = batchify(corpus.train, batch_size).as_in_context(context)
val_data = batchify(corpus.valid, batch_size).as_in_context(context)
test_data = batchify(corpus.test, batch_size).as_in_context(context)

def run_epoch(data, is_train):
    """Run one epoch"""
    batches = range(0, data.shape[0]-1, seq_len)
    if is_train:
        shuffle(batches)
    if num_batches and num_batches < len(batches):
        batches = batches[:num_batches]
        
    total_loss = 0
    total_num = 0
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)
    for i, b in enumerate(batches):
        source, target = get_batch(data, seq_len, b)
        if is_train:
            # Record the chain on the forward stage
            hidden = [h.detach() for h in hidden]
            with autograd.record():
                output, hidden = model(source, hidden)
                loss_value = softmax_ce_loss(output, target)
                #print(type(loss_value))
            # Compute the gradient
            loss_value.backward()

            # Here gradient is not divided by batch_size yet.
            # So we multiply max_norm by batch_size to balance it.
            grads = [p.grad(context) for p in model.collect_params().values()]
            clip_global_norm(grads, 0.25 * batch_size)
            
            # update the parameters
            trainer.step(batch_size)

        else:
            # Gradients are not needded for inference. Don't use record() here for better performance.
            output, hidden = model(source, hidden)
            loss_value = softmax_ce_loss(output, target)
                
        total_loss += mx.nd.sum(loss_value).asscalar()
        total_num += loss_value.size
        
        if is_train and i % disp_batch == 0 and i > 0:
            print(' - Batch %d: ppl %.2f'%(
                    i, math.exp(total_loss/total_num)))
    return math.exp(total_loss/total_num)

for epoch in range(num_epochs):
    print('Epoch %d:'%(epoch))
    start_time = time.time()
    print('Epoch %d: Training ppl = %.2f' %(epoch, run_epoch(train_data, True) ))
    print('Epoch %d: Validation ppl = %.2f' %(epoch, run_epoch(val_data, False) ))
    print('Epoch %d: Time = %.2f sec' %(epoch, time.time()-start_time))
          
          
print('Test ppl = %.2f'%(run_epoch(test_data, False)))

```
<!-- INSERT SOURCE DOWNLOAD BUTTONS -->