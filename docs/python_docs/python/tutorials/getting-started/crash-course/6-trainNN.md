```python
!pip install --pre --user mxnet -f https://dist.mxnet.io/python/cpu
```

# Training a Neural Network

We have seen all the necessary components for creating a neural network, we are
now ready put all the pieces together and train a model end to end.

## 1. Data preparation

The typical process to create and train a model starts with loading and
preparing the datasets. For this Network we will use a [dataset of leaf
images](https://data.mendeley.com/datasets/hb74ynkjcn/1) that consist of healthy
and diseased leaf images of 12 different plant species. To get this dataset we
first download and extract it

```python
# Download dataset
!wget https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/hb74ynkjcn-1.zip
```

```python
# Extract the dataset in a folder that we create and call plants
!mkdir plants
!unzip hb74ynkjcn-1.zip -d plants
!rm hb74ynkjcn-1.zip
```

#### Data inspection

If we take a look at the dataset we find that the structure of the directories
is as follows

```
plants
|-- Alstonia Scholaris (P2)
|-- Arjun (P1)
|-- Bael (P4)
    |-- diseased
        |-- 0016_0001.JPG
        |-- .
        |-- .
        |-- .
        |-- 0016_0118.JPG
|-- .
|-- .
|-- .
|-- Mango (P0)
    |-- diseased
    |-- healthy

```

So for each plant species we might have examples of diseased leaves or healthy
leaves or both. With this dataset we can formulate different classification
problems, for example we can create a multi-class classifier that determines the
species of a plant based on the leaves, we can also create a binary classifier
that tells you whether the plant is healthy or diseased, finally we could create
a multi-class, multi-label classifier that tells you both: what species is a
plant and whether is diseased or healthy. We will stick to the simplest
classification question, which is whether a plant is healthy or not.

To do this, we need to manipulate the dataset in two ways. First we need to
combine all images of healthy and diseased no matter the species and then we
need to split the data in the train, validation and test sets. We prepared a
small utility code to get our dataset ready. Once we run that utility code on
our data, because our structure will be already organized in folders, we can use
the `ImageFolderDataset` class.

```python
# Import all the necessary libraries to train
import time

import mxnet as mx
from mxnet import np, npx, gluon, init, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

from prepare_dataset import process_dataset #utility code to rearrange the data

mx.random.seed(42)
```

```python
# Call the utility function to rearrange the images
process_dataset('plants')
```

The new dataset will be located in the `datasets` folder and the new structure
looks like this:

```
datasets
|-- test
    |-- diseased
    |-- healthy
|-- train
|-- validation
    |-- diseased
    |-- healthy
        |-- image1.JPG
        |-- image2.JPG
        |-- .
        |-- .
        |-- .
        |-- imagen.JPG

```

Now we just create three different Dataset objects from the `train`,
`validation` and `test` folders and the `ImageFolderDataset` class will take
care of inferring the classes from the directory names.

```python
# Use ImageFolderDataset to create a Dataset object from directory structure
train_dataset = gluon.data.vision.ImageFolderDataset('./datasets/train')
val_dataset = gluon.data.vision.ImageFolderDataset('./datasets/validation')
test_dataset = gluon.data.vision.ImageFolderDataset('./datasets/test')
```

The result from this will be a Dataset object, we can index this object to get
the $i$-th element from the dataset. The $i$-th element is a tuple with two
objects, the first one is the image in array form and the second one the
corresponding label for that training example.

```python
sample_idx = 888 # choose a random sample
sample = train_dataset[sample_idx]
data = sample[0]
label = sample[1]

plt.imshow(data.asnumpy())
print(f"Data type: {data.dtype}")
print(f"Label: {label}")
print(f"Label description: {train_dataset.synsets[label]}")
print(f"Image shape: {data.shape}")
```

As you can see from the plot, the image size is very big 4000 x 6000 pixels.
Usually we downsize images before passing to the neural network because that
greatly speeds up the training time. It is also customary to make slight
modifications to the images to improve generalization. That is why we add
transformations to our data in a process called Data Augmentation.

We can augmented data in MXNet using `transforms`. We compose two different
transformation pipelines, one for training and the other one for validations and
testing. This is because we don't need to randomly flip or do color jitter to
the images we'll be classifying later on.

```python
# Import transforms as compose a series of transformations to the images
from mxnet.gluon.data.vision import transforms

jitter_param = 0.05

# mean and std for normalizing image value in range (0,1)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

training_transformer = transforms.Compose([
    transforms.Resize((350, 500)),
    transforms.CenterCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(contrast=jitter_param),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

validation_transformer = transforms.Compose([
    transforms.Resize((350, 500)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

We now have a our augmentations ready so we can create our `DataLoaders`. To do
this we use the `gluon.data.DataLoader` and we pass the dataset with the applied
transformations (notice the `.transform_first()` method on the datasets). In
this step we also need to decide the batch size, which is how many images we
will be presenting to the network at the same time, and whether we want to
shuffle the data before passing it to the network.

```python
# Create data loaders
batch_size = 4
train_loader = gluon.data.DataLoader(train_dataset.transform_first(training_transformer),batch_size=batch_size, shuffle=True)
validation_loader = gluon.data.DataLoader(val_dataset.transform_first(validation_transformer), batch_size=batch_size)
test_loader = gluon.data.DataLoader(test_dataset.transform_first(validation_transformer), batch_size=batch_size)
```

Now we can see the transformations that we made to the images.

```python
# Function to plot batch
def show_batch(batch, columns=4):
    labels = batch[1].asnumpy()
    batch = batch[0] / 2 + 0.5     # unnormalize
    batch = batch.asnumpy()
    size = batch.shape[0]
    rows = int(size / columns)
    fig, axes = plt.subplots(rows, columns)
    for ax, img, label in zip(axes.flatten(), batch, labels):
        ax.imshow(np.transpose(img, (1, 2, 0)))
        ax.set(title=f"Label: {label}")
    plt.show()
```

```python
for batch in train_loader:
    a = batch
    break
```

```python
show_batch(a)
```

We can see that the original images changed to have different size, variations
in color and lighting and we are now ready to go to the next step: **Create the
architecture**.

## 2. Create Neural Network

Convolutional neural networks are a great tool to capture the spatial
relationship of images, for this reason they have become the gold standard for
computer vision. For this example we will create a small convolutional neural
network. First we create two functions that will create the two type of blocks
we'll use, the convolutional and the dense block and then we will create an
entire network based on these two blocks.

```python
# The convolutional block has a convolution layer, a max pool layer and a batch normalization layer
def conv_block(filters, kernel_size=2, stride=2, batch_norm=True):
    conv_block = nn.HybridSequential()
    conv_block.add(nn.Conv2D(channels=filters, kernel_size=kernel_size, activation='relu'),
              nn.MaxPool2D(pool_size=4, strides=stride))
    if batch_norm:
        conv_block.add(nn.BatchNorm())
    return conv_block

# The dense block consists of a dense layer and a dropout layer
def dense_block(neurons, activation='relu', dropout=0.2):
    dense_block = nn.HybridSequential()
    dense_block.add(nn.Dense(neurons, activation=activation))
    if dropout:
        dense_block.add(nn.Dropout(dropout))
    return dense_block
```

```python
# Create neural network blueprint using the blocks
class LeafNetwork(nn.HybridBlock):
    def __init__(self):
        super(LeafNetwork, self).__init__()
        self.conv1 = conv_block(32)
        self.conv2 = conv_block(64)
        self.conv3 = conv_block(128)
        self.flatten = nn.Flatten()
        self.dense1 = dense_block(512)
        self.dense2 = dense_block(100)
        self.dense3 = nn.Dense(2)
        
    def forward(self, batch):
        batch = self.conv1(batch)
        batch = self.conv2(batch)
        batch = self.conv3(batch)
        batch = self.flatten(batch)
        batch = self.dense1(batch)
        batch = self.dense2(batch)
        batch = self.dense3(batch)
        
        return batch
```

We have concluded the architecting part of the network, so now we can actually
create an instance from the architecture. As we have seen previously in this
course, to be able to use the network we need to initialize the parameters and
hybridize the model.

```python
# Create the model based on the blueprint we built and initialize the parameters
ctx = mx.cpu() 

initializer = mx.initializer.Xavier()

model = LeafNetwork()
model.initialize(initializer, ctx=ctx)
model.summary(mx.nd.random.uniform(shape=(4, 3, 224, 224)))
model.hybridize()
```

## 3. Choose Optimizer and Loss function

With the network created we can move on to choosing an optimizer and a loss
function. We used these to make an informed decision on how to tune the
parameters to fit the final objective better. We use the `gluon.Trainer` to help
us with optimizing the parameters. The `gluon.Trainer` needs two things to work
properly, the parameters that need to be tuned and the optimizer with its
corresponding hyperparameters. The trainer uses the error reported by the loss
function to optimize the parameters.

For this particular dataset we choose Stochastic Gradient Descent as the
optimizer and Cross Entropy as the loss function.

```python
# SGD optimizer
optimizer = 'sgd'

# Set parameters
optimizer_params = {'learning_rate': 0.001}

# Define our trainer for the model
trainer = gluon.Trainer(model.collect_params(), optimizer, optimizer_params)

# Define the loss function
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
```

Finally we create a function to evaluate the performance of the network in the
validation set and then set up the training loop.

```python
# Function to return the accuracy for the validation and test set
def test(ctx, val_data):
    acc = gluon.metric.Accuracy()
    for batch in val_data:
        data = batch[0]
        labels = batch[1]
        outputs = model(data)
        acc.update(labels, outputs)

    _, accuracy = acc.get()
    return accuracy
```

## 4. Training Loop

We have everything set so we can start training our network. This might take
some time to train depending on the hardware and number of layers and images you
use. For this particular case we only choose to train for 2 epochs.

```python
# Start the training loop
epochs = 2
accuracy = gluon.metric.Accuracy()
log_interval = 100

for epoch in range(epochs):
    tic = time.time()
    btic = time.time()
    accuracy.reset()

    for idx, batch in enumerate(train_loader):
        data = batch[0]
        label = batch[1]
        with mx.autograd.record():
            outputs = model(data)
            loss = loss_fn(outputs, label)
        mx.autograd.backward(loss)
        trainer.step(batch_size)
        accuracy.update(label, outputs)
        if log_interval and not (idx + 1) % log_interval:
            _, acc = accuracy.get()
     
            print(f"""Epoch[{epoch + 1}] Batch[{idx + 1}] Speed: {batch_size / (time.time() - btic)} samples/sec \
                  loss = {loss.mean().asscalar()} | accuracy = {acc}""")
            btic = time.time()

    _, acc = accuracy.get()
    
    acc_val = test(ctx, validation_loader)
    print(f"[Epoch {epoch + 1}] training: accuracy={acc}")
    print(f"[Epoch {epoch + 1}] time cost: {time.time()-tic}")
    print(f"[Epoch {epoch + 1}] validation: validation accuracy={acc_val}")
```

## 5. Test on the test set

Now that our network is trained and we reached a decent accuracy we can evaluate
the performance on the test set. To that end we use the `test_loader` and the
test function

```python
test(ctx, test_loader)
```

We have a trained network that can confidently discriminate between plants that
are healthy and the ones that are diseased. We can now start our garden and set
cameras to automatically detect plants in distress!

## 6. Save the parameters

If we want to preserve the trained weights of the network we can save the
parameters in a file. Later when we want to use the network to make predictions
we can load the parameters back

```python
# Save parameters in the 
model.save_parameters('leaf_models.params')
```
