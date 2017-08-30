
# Word Embeddings

In this tutorial, we will discuss about embeddings, why they are needed and commonly
used model architectures to produce distributed words representation.

Word embeddings are dense vectors of real numbers, one per word in your vocabulary.
These embeddings can be used as features
for many machine learning and NLP applications like sentiment analysis and document classification.


## The embedding layer

We first present how to use the embedding layer. A embedding layer mantains a weight of size *(vocab_size, embedding_size)*, which stores the embedding vectors for each word in the vocabulary row by row. 

The following example defines a map that mapping words into indices and create a simple embedding layer.


```python
from mxnet import nd
from mxnet.gluon import nn

# word-to-index mapping 
word_map = {'hello':0, 'world':1, '!':2}

# the vocabulary size
vocab_size = len(word_map)
# lenght of an embedding vector 
embed_size = 2

# A readable embedding weight
weight = nd.arange(vocab_size*embed_size).reshape((vocab_size, embed_size))+10 

# create an embeeding layer and set its weight
embed = nn.Embedding(vocab_size, embed_size)
embed.params.initialize()
embed.params.get('weight').set_data(weight)

print('Embedding weight')
print(embed.params.get('weight').data())
```

Now we can lookup the embedding vectors. First we try a single word, which will return a vector of size `(1, embed_size)`.


```python
x = nd.array([word_map['world']])
print(x)
print(embed(x))
```

Then we feed a 2D `(n, m)` input, whose output size should be `(n, m, embed_size)`


```python
y = nd.array([[word_map['hello'], word_map['world']], [word_map['world'], word_map['!']]])
print(y)
print(embed(y))
```

## N-gram language modeling


Now we discuss how to use word embedding for a n-gram language model. The n-gram language model predicts the next word by using the previous *n-1* words. Namely it predicts

$$ P(w_i|w_{i-1},w_{i-2},…,w_{i-n+1}) $$

Where $w_i$ is the *i*-th word of the sequence.

The following codes define a simple corpus and construct its trigram representation. 


```python
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold."""

# build the word-to-index map as before
words = [w.lower() for w in test_sentence.split()]
vocab = set(words) 
word_map = {word: ind for ind, word in enumerate(vocab)}
print('Word-to-index map:')
print(word_map)

# A list of tuples, each tuple contains the target word and the previous two context words.
trigrams = [([words[i-2], words[i-1]], words[i]) for i in range(2, len(words))]
print('First 3 tri-grams:')
print(trigrams[:3])
```

Now we define a model that consits of one embedding layer and one fully-connected layer. The embedding layer each time takes *n-1* context word indices to output a *(n-1, embed_size)* embedding matrix. This matrix is then flattened to feed into the fully-connected layer, the output will be of *vocab_size* length. We can later on apply softmax to obtain the predicted probabilities of the next word.


```python
class NGram(nn.Block):
    def __init__(self, vocab_size, embed_size, **kwargs):
        super(NGram, self).__init__(**kwargs)
        with self.name_scope():
            self.embed = nn.Embedding(vocab_size, embed_size)
            self.fc = nn.Dense(vocab_size)

    def forward(self, x):                
        a = self.embed(x)
        # concate n-1 embedding vectors into a single vector. 
        # Here -1 means inference the shape, whose infered value should be a.size
        b = a.reshape((1, -1)) 
        y = self.fc(b)
        return y
    
# create one network instance
ngram = NGram(vocab_size=len(vocab), embed_size=10)

# collect all layer parameters and initialize them with random values. 
ngram.collect_params().initialize()

# construct a sample input and perform forward
x = nd.array([word_map['where'], word_map['all']])
y = ngram(x)
print(y.shape)
```

Both data and model are ready. We only needs a loss function (such as softmax-cross-entroy-loss) and an optimization method (such as stochastic gradient descent) before starting training. Since the training programs of typical language models are similar, we create a general training function here. 


```python
from mxnet import autograd
def train(data, word_map, model, loss, trainer, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0  # total loss value over all examples
        for context, target in data:
            # the indices of the context words
            context_ids = nd.array([word_map[w] for w in context])
            # the index of the target word
            label_id = nd.array([word_map[target]])
            # record the the forward so that we can call backward later
            with autograd.record():                
                predict = model(context_ids)
                loss_value = loss(predict, label_id)
            # backpropogate the error to calculate gradients
            loss_value.backward()
            # update the parameters, here 1 is the batch size 
            # because we process a single exmaple each time. 
            trainer.step(1)
            # again, we feed one example each time, the loss value is a scalar 
            total_loss += loss_value.asscalar()  
        print('Epoch %d, loss %.2f' % (epoch, total_loss))
```

Define the loss function and optmization method to start training:


```python
from mxnet.gluon import loss, Trainer
# normalize the predictions and compute the cross entropy loss against target
softmax_ce_loss = loss.SoftmaxCrossEntropyLoss()
# apply SGD on all layer parameters
ngram_trainer = Trainer(ngram.collect_params(), 'sgd', {'learning_rate': 0.1})
# train 10 epochs
train(trigrams, word_map, ngram, softmax_ce_loss, ngram_trainer, num_epochs=10)
```

Define a prediction function and run a few tests. (You way want to run more epochs to obtain more accurate predictions)


```python
def predict(word_map, model, x):
    x = nd.array([word_map[w] for w in x])
    y = model(x)
    i = nd.argmax(y, axis=1).asscalar()
    for w in word_map:
        if word_map[w] == i:
            return w
for i in range(8):
    print('%s vs %s'%(
            predict(word_map, ngram, trigrams[i][0]), trigrams[i][1]))

```

## Continuous Bag-of-Words

Continuous Bag-of-Words model (CBOW) is frequently used model to learn word embedding. It tries to predict words given the context of a few words before and a few words after the target word. This is distinct from language modeling, because it uses words after the target word. 


In particular, given a target word $w_i$ and an *n* context window on each side, $w_{i-1},…,w_{i-n}$ and $w_{i+1},…,w_{i+n}$, referring to all context words collectively as *C*, CBOW tries to minimize

  $$ - \log P(w_i | C) = - \log Softmax(A(\sum_{w \in C} q_w) + b)$$

where $q_w$ is the embedding for word w.

We first construct the data with context window size to be 2:


```python
cbow_data = [((words[i-2], words[i-1], words[i+1], words[i+2]), words[i]) for i in range(2, len(words)-2)]
print(cbow_data[:3])
```

The model definition of CBOW is similiar to the above n-gram model. The main difference is that we sum the word embeddings over the context words instead of concating. 


```python
class CBOW(nn.Block):
    def __init__(self, vocab_size, embed_size, **kwargs):
        super(CBOW, self).__init__(**kwargs)
        with self.name_scope():
            self.embed = nn.Embedding(vocab_size, embed_size)
            self.fc = nn.Dense(vocab_size)

    def forward(self, x):
        a = self.embed(x)
        # sum of the surrounding word embeddings
        b = nd.sum(a, axis=0, keepdims=True) 
        y = self.fc(b)
        return y
```

Now we train and predict as before:


```python
cbow = CBOW(vocab_size=len(vocab), embed_size=10)
cbow.collect_params().initialize()
cbow_trainer = Trainer(cbow.collect_params(), 'sgd', {'learning_rate': .1})
train(cbow_data, word_map, cbow, softmax_ce_loss, cbow_trainer, 10)
```


```python
for i in range(8):
    print('%s vs %s'%(
            predict(word_map, cbow, trigrams[i][0]), trigrams[i][1]))
```


```python

```
<!-- INSERT SOURCE DOWNLOAD BUTTONS -->