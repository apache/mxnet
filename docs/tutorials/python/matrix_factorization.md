# Matrix Factorization

In a recommendation system, there is a group of users and a set of items. Given
that each users have rated some items in the system, we would like to predict
how the users would rate the items that they have not yet rated, such that we
can make recommendations to the users.

Matrix factorization is one of the mainly used algorithm in recommendation
systems. It can be used to discover latent features underlying the interactions
between two different kinds of entities.

Assume we assign a k-dimensional vector to each user and a k-dimensional vector
to each item such that the dot product of these two vectors gives the user's
rating of that item. We can learn the user and item vectors directly, which is
essentially performing SVD on the user-item matrix. We can also try to learn the
latent features using multi-layer neural networks.

In this tutorial, we will work though the steps to implement these ideas in
MXNet.

## Prepare Data

We use the [MovieLens](http://grouplens.org/datasets/movielens/) data here, but
it can apply to other datasets as well. Each row of this dataset contains a
tuple of user id, movie id, rating, and time stamp, we will only use the first
three items. We first define the a batch which contains n tuples. It also
provides name and shape information to MXNet about the data and label.


```python
class Batch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]
```

Then we define a data iterator, which returns a batch of tuples each time.


```python
import mxnet as mx
import random

class Batch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class DataIter(mx.io.DataIter):
    def __init__(self, fname, batch_size):
        super(DataIter, self).__init__()
        self.batch_size = batch_size
        self.data = []
        for line in file(fname):
            tks = line.strip().split('\t')
            if len(tks) != 4:
                continue
            self.data.append((int(tks[0]), int(tks[1]), float(tks[2])))
        self.provide_data = [('user', (batch_size, )), ('item', (batch_size, ))]
        self.provide_label = [('score', (self.batch_size, ))]

    def __iter__(self):
        for k in range(len(self.data) / self.batch_size):
            users = []
            items = []
            scores = []
            for i in range(self.batch_size):
                j = k * self.batch_size + i
                user, item, score = self.data[j]
                users.append(user)
                items.append(item)
                scores.append(score)

            data_all = [mx.nd.array(users), mx.nd.array(items)]
            label_all = [mx.nd.array(scores)]
            data_names = ['user', 'item']
            label_names = ['score']

            data_batch = Batch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        random.shuffle(self.data)
```

Now we download the data and provide a function to obtain the data iterator:


```python
import os
import urllib
import zipfile
if not os.path.exists('ml-100k.zip'):
    urllib.urlretrieve('http://files.grouplens.org/datasets/movielens/ml-100k.zip', 'ml-100k.zip')
with zipfile.ZipFile("ml-100k.zip","r") as f:
    f.extractall("./")
def get_data(batch_size):
    return (DataIter('./ml-100k/u1.base', batch_size), DataIter('./ml-100k/u1.test', batch_size))
```

Finally we calculate the numbers of users and items for later use.

```python
def max_id(fname):
    mu = 0
    mi = 0
    for line in file(fname):
        tks = line.strip().split('\t')
        if len(tks) != 4:
            continue
        mu = max(mu, int(tks[0]))
        mi = max(mi, int(tks[1]))
    return mu + 1, mi + 1
max_user, max_item = max_id('./ml-100k/u.data')
(max_user, max_item)
```

## Optimization

We first implement the RMSE (root-mean-square error) measurement, which is
commonly used by matrix factorization.

```python
import math
def RMSE(label, pred):
    ret = 0.0
    n = 0.0
    pred = pred.flatten()
    for i in range(len(label)):
        ret += (label[i] - pred[i]) * (label[i] - pred[i])
        n += 1.0
    return math.sqrt(ret / n)
```

Then we define a general training module, which is borrowed from the image
classification application.

```python
def train(network, batch_size, num_epoch, learning_rate):
    model = mx.model.FeedForward(
        ctx = mx.gpu(0),
        symbol = network,
        num_epoch = num_epoch,
        learning_rate = learning_rate,
        wd = 0.0001,
        momentum = 0.9)

    batch_size = 64
    train, test = get_data(batch_size)

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG)

    model.fit(X = train,
              eval_data = test,
              eval_metric = RMSE,
              batch_end_callback=mx.callback.Speedometer(batch_size, 20000/batch_size),)
```

## Networks

Now we try various networks. We first learn the latent vectors directly.

```python
def plain_net(k):
    # input
    user = mx.symbol.Variable('user')
    item = mx.symbol.Variable('item')
    score = mx.symbol.Variable('score')
    # user feature lookup
    user = mx.symbol.Embedding(data = user, input_dim = max_user, output_dim = k)
    # item feature lookup
    item = mx.symbol.Embedding(data = item, input_dim = max_item, output_dim = k)
    # predict by the inner product, which is elementwise product and then sum
    pred = user * item
    pred = mx.symbol.sum_axis(data = pred, axis = 1)
    pred = mx.symbol.Flatten(data = pred)
    # loss layer
    pred = mx.symbol.LinearRegressionOutput(data = pred, label = score)
    return pred

train(plain_net(64), batch_size=64, num_epoch=10, learning_rate=.05)
```

Next we try to use 2 layers neural network to learn the latent variables, which stack a fully connected layer above the embedding layers:

```python
def get_one_layer_mlp(hidden, k):
    # input
    user = mx.symbol.Variable('user')
    item = mx.symbol.Variable('item')
    score = mx.symbol.Variable('score')
    # user latent features
    user = mx.symbol.Embedding(data = user, input_dim = max_user, output_dim = k)
    user = mx.symbol.Activation(data = user, act_type="relu")
    user = mx.symbol.FullyConnected(data = user, num_hidden = hidden)
    # item latent features
    item = mx.symbol.Embedding(data = item, input_dim = max_item, output_dim = k)
    item = mx.symbol.Activation(data = item, act_type="relu")
    item = mx.symbol.FullyConnected(data = item, num_hidden = hidden)
    # predict by the inner product
    pred = user * item
    pred = mx.symbol.sum_axis(data = pred, axis = 1)
    pred = mx.symbol.Flatten(data = pred)
    # loss layer
    pred = mx.symbol.LinearRegressionOutput(data = pred, label = score)
    return pred

train(get_one_layer_mlp(64, 64), batch_size=64, num_epoch=10, learning_rate=.05)
```

Adding dropout layers to relief the over-fitting.

```python
def get_one_layer_dropout_mlp(hidden, k):
    # input
    user = mx.symbol.Variable('user')
    item = mx.symbol.Variable('item')
    score = mx.symbol.Variable('score')
    # user latent features
    user = mx.symbol.Embedding(data = user, input_dim = max_user, output_dim = k)
    user = mx.symbol.Activation(data = user, act_type="relu")
    user = mx.symbol.FullyConnected(data = user, num_hidden = hidden)
    user = mx.symbol.Dropout(data=user, p=0.5)
    # item latent features
    item = mx.symbol.Embedding(data = item, input_dim = max_item, output_dim = k)
    item = mx.symbol.Activation(data = item, act_type="relu")
    item = mx.symbol.FullyConnected(data = item, num_hidden = hidden)
    item = mx.symbol.Dropout(data=item, p=0.5)
    # predict by the inner product
    pred = user * item
    pred = mx.symbol.sum_axis(data = pred, axis = 1)
    pred = mx.symbol.Flatten(data = pred)
    # loss layer
    pred = mx.symbol.LinearRegressionOutput(data = pred, label = score)
    return pred
train(get_one_layer_mlp(256, 512), batch_size=64, num_epoch=10, learning_rate=.05)
```

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
