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

# Custom Loss Blocks

All neural networks need a loss function for training. A loss function is a quantitive measure of how bad the predictions of the network are when compared to ground truth labels. Given this score, a network can improve by iteratively updating its weights to minimise this loss. Some tasks use a combination of multiple loss functions, but often you'll just use one. MXNet Gluon provides a number of the most commonly used loss functions, and you'll choose certain functions depending on your network and task. Some common task and loss function pairs include:

- Regression: [L1Loss](/api/python/docs/api/gluon/loss/index.html#mxnet.gluon.loss.L1Loss), [L2Loss](/api/python/docs/api/gluon/loss/index.html#mxnet.gluon.loss.L2Loss)
- Classification: [SigmoidBinaryCrossEntropyLoss](/api/python/docs/api/gluon/loss/index.html#mxnet.gluon.loss.SigmoidBinaryCrossEntropyLoss), [SoftmaxBinaryCrossEntropyLoss](/api/python/docs/api/gluon/loss/index.html#mxnet.gluon.loss.SoftmaxBinaryCrossEntropyLoss)
- Embeddings: [HingeLoss](/api/python/docs/api/gluon/loss/index.html#mxnet.gluon.loss.HingeLoss)

However, we may sometimes want to solve problems that require customized loss functions; this tutorial shows how we can do that in Gluon. We will implement contrastive loss which is typically used in Siamese networks.

```python
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import autograd, gluon, nd
from mxnet.gluon.loss import Loss
import random
```

### What is Contrastive Loss

[Contrastive loss](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf) is a distance-based loss function. During training, pairs of images are fed into a model. If the images are similar, the loss function will return 0, otherwise 1.

<img src="images/contrastive_loss.jpeg" width="400">

*Y* is a binary label indicating similarity between training images. Contrastive loss uses the Euclidean distance *D* between images and is the sum of 2 terms:
 - the loss for a pair of similar points
 - the loss for a pair of dissimilar points

The loss function uses a margin *m* which is has the effect that dissimlar pairs only contribute if their loss is within a certain margin.

In order to implement such a customized loss function in Gluon, we only need to define a new class that is inheriting from the [Loss](/api/python/docs/api/gluon/loss/index.html#mxnet.gluon.loss.Loss) base class. We then define the contrastive loss logic in the [hybrid_forward](/api/python/docs/api/gluon/hybrid_block.html#mxnet.gluon.HybridBlock.hybrid_forward) method. This method takes the images `image1`, `image2` and the label which defines whether  `image1` and `image2` are similar (=0) or  dissimilar (=1). The input F is an `mxnet.ndarry` or an `mxnet.symbol` if we hybridize the network. Gluon's `Loss` base class is in fact a [HybridBlock](/api/python/docs/api/gluon/hybrid_block.html). This means we can either run  imperatively or symbolically. When we hybridize our custom loss function, we can get performance speedups.


```python
class ContrastiveLoss(Loss):
    def __init__(self, margin=6., weight=None, batch_axis=0, **kwargs):
        super(ContrastiveLoss, self).__init__(weight, batch_axis, **kwargs)
        self.margin = margin

    def hybrid_forward(self, F, image1, image2, label):
        distances = image1 - image2
        distances_squared = F.sum(F.square(distances), 1, keepdims=True)
        euclidean_distances = F.sqrt(distances_squared + 0.0001)
        d = F.clip(self.margin - euclidean_distances, 0, self.margin)
        loss = (1 - label) * distances_squared + label * F.square(d)
        loss = 0.5*loss
        return loss
loss = ContrastiveLoss(margin=6.0)
```

### Define the Siamese network
A [Siamese network](https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf) consists of 2 identical networks, that share the same weights. They are trained on pairs of images and each network processes one image. The label defines whether the pair of images is similar or not. The Siamese network learns to differentiate between two input images.

Our network consists of 2 convolutional and max pooling layers that downsample the input image. The output is then fed through a fully connected layer with 256 hidden units and another fully connected layer with 2 hidden units.


```python
class Siamese(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(Siamese, self).__init__(**kwargs)
        with self.name_scope():
            self.cnn = gluon.nn.HybridSequential()
            with self.cnn.name_scope():
                self.cnn.add(gluon.nn.Conv2D(64, 5, activation='relu'))
                self.cnn.add(gluon.nn.MaxPool2D(2, 2))
                self.cnn.add(gluon.nn.Conv2D(64, 5, activation='relu'))
                self.cnn.add(gluon.nn.MaxPool2D(2, 2))
                self.cnn.add(gluon.nn.Dense(256, activation='relu'))
                self.cnn.add(gluon.nn.Dense(2, activation='softrelu'))

    def hybrid_forward(self, F, input0, input1):
        out0 = self.cnn(input0)
        out1 = self.cnn(input1)
        return out0, out1

```

### Prepare the training data

We train our network on the [Ominglot](http://www.omniglot.com/) dataset which is a collection of 1623 hand drawn characters from 50 alphabets. You can download it from [here](https://github.com/brendenlake/omniglot/tree/master/python). We need to create a dataset that contains a random set of similar and dissimilar images. We use Gluon's `ImageFolderDataset` where we overwrite `__getitem__` and randomly return similar and dissimilar pairs of images.


```python
class GetImagePairs(mx.gluon.data.vision.ImageFolderDataset):
    def __init__(self, root):
        super(GetImagePairs, self).__init__(root, flag=0)
        self.root = root

    def __getitem__(self, index):
        items_with_index = list(enumerate(self.items))
        image0_index, image0_tuple = random.choice(items_with_index)
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                image1_index, image1_tuple = random.choice(items_with_index)
                if image0_tuple[1] == image1_tuple[1]:
                    break
        else:
            image1_index, image1_tuple = random.choice(items_with_index)
        image0 = super().__getitem__(image0_index)
        image1 = super().__getitem__(image1_index)
        label = mx.nd.array([int(image1_tuple[1] != image0_tuple[1])])
        return image0[0], image1[0], label

    def __len__(self):
        return super().__len__()

```

We train the network on a subset of the data, the  [*Tifinagh*](https://www.omniglot.com/writing/tifinagh.htm) alphabet. Once the model is trained we test it on the [*Inuktitut*](https://www.omniglot.com/writing/inuktitut.htm) alphabet.


```python
def transform(img0, img1, label):
    normalized_img0 = nd.transpose(img0.astype('float32'), (2, 0, 1))/255.0
    normalized_img1 = nd.transpose(img1.astype('float32'), (2, 0, 1))/255.0
    return normalized_img0, normalized_img1, label

training_dir = "images_background/Tifinagh"
testing_dir = "images_background/Inuktitut_(Canadian_Aboriginal_Syllabics)"
train = GetImagePairs(training_dir)
test = GetImagePairs(testing_dir)
train_dataloader = gluon.data.DataLoader(train.transform(transform),
                                        shuffle=True, batch_size=16)
test_dataloader = gluon.data.DataLoader(test.transform(transform),
                                        shuffle=False, batch_size=1)
```

Following code plots some examples from the test dataset.


```python
img1, img2, label = test[0]
print("Same: {}".format(int(label.asscalar()) == 0))
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))
ax0.imshow(img1.asnumpy()[:,:,0], cmap='gray')
ax0.axis('off')
ax1.imshow(img2.asnumpy()[:,:,0], cmap='gray')
ax1.axis("off")
plt.show()

```

![example1](images/inuktitut_1.png)


### Train the Siamese network

Before we can start training, we need to instantiate the custom constrastive loss function and initialize the model.


```python
model = Siamese()
model.initialize(init=mx.init.Xavier())
trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 0.001})
loss = ContrastiveLoss(margin=6.0)
```

Start the training loop:


```python
for epoch in range(10):
    for i, data in enumerate(train_dataloader):
        image1, image2, label = data
        with autograd.record():
            output1, output2 = model(image1, image2)
            loss_contrastive = loss(output1, output2, label)
        loss_contrastive.backward()
        trainer.step(image1.shape[0])
        loss_mean = loss_contrastive.mean().asscalar()
        print("Epoch number {}\n Current loss {}\n".format(epoch, loss_mean))

```

### Test the trained Siamese network
During inference we compute the Euclidean distance between the output vectors of the Siamese network. High distance indicates dissimilarity, low values indicate similarity.


```python
for i, data in enumerate(test_dataloader):
    img1, img2, label = data
    output1, output2 = model(img1, img2)
    dist_sq = mx.ndarray.sum(mx.ndarray.square(output1 - output2))
    dist = mx.ndarray.sqrt(dist_sq).asscalar()
    print("Euclidean Distance:", dist, "Test label", label[0].asscalar())
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))
    ax0.imshow(img1.asnumpy()[0, 0, :, :], cmap='gray')
    ax0.axis('off')
    ax1.imshow(img2.asnumpy()[0, 0, :, :], cmap='gray')
    ax1.axis("off")
    plt.show()

```

![example2](images/inuktitut_2.png)


### Common pitfalls with custom loss functions

When customizing loss functions, we may encounter certain pitfalls. If the loss is not decreasing as expected or if forward/backward pass is crashing, then one should check the following:

#### Activation function in the last layer
Verify whether the last network layer uses the correct activation function: for instance in binary classification tasks we need to apply a sigmoid on the output data. If we use this activation in the last layer and define a loss function like Gluon's SigmoidBinaryCrossEntropy, we would basically apply sigmoid twice and the loss would not converge as expected. If we don't define any activation function, Gluon will per default apply a linear activation.

####  Intermediate loss values
In our example, we computed the square root of squared distances between 2 images: `F.sqrt(distances_squared)`. If images are very similar we take the sqare root of a value close to 0, which can lead to *NaN* values. Adding a small epsilon to `distances_squared` avoids this problem.

#### Shape of intermediate loss vectors
In most cases having the wrong tensor shape will lead to an error, as soon as we compare data with labels. But in some cases, we may be able to normally run the training, but it does not converge. For instance, if we don't set `keepdims=True` in our customized loss function, the shape of the tensor changes. The example still runs fine but does not converge.

If you encounter a similar problem, then it is useful to check the tensor shape after each computation step in the loss function.

#### Differentiable
Backprogration requires the loss function to be differentiable. If the customized loss function cannot be differentiated the backward pass will crash.
