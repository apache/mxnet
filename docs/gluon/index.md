# Gluon

The new Gluon interface is available as a library in Apache MXNet and allows developers of all skill levels to prototype, build, and train deep learning models. This interface greatly simplifies the process of creating deep learning models without sacrificing training speed. You can easily get started with the Gluon interface using [this extensive set of tutorials, documentation, and example](https://virtualenv.pypa.io/en/stable/userguide/).

**Here are Gluon’s four major advantages:**

**1. Simple, Easy-to-Understand Code**

In Gluon, neural networks can be defined using simple, clear, concise code – this is easier to learn and understand. You get a set of plug-and-play neural network building blocks, including predefined layers, optimizers, and initializers. These abstract away many of the complicated underlying implementation details. Below you see that you can define a neural network with just a few lines of code:


```bash
net = gluon.nn.Sequential()
# When instantiated, Sequential stores a chain of neural network layers. 
# Once presented with data, Sequential executes each layer in turn, using 
# the output of one layer as the input for the next
with net.name_scope():
    net.add(gluon.nn.Dense(256, activation="relu")) # 1st layer (256 nodes)
    net.add(gluon.nn.Dense(256, activation="relu")) # 2nd hidden layer
    net.add(gluon.nn.Dense(num_outputs))
```
<br/>

**2. Flexible, Imperative Structure**

The Gluon interface supports a fully imperative way of working with neural networks, offering more familiarity and flexibility. Many familiar programming languages like Python are imperative, and imperative programs execute code line by line versus waiting for a code compilation step prior to executing any code. With Gluon, you can easily experiment and prototype with neural networks. Debugging is also easier with Gluon because you can identify the exact line of code causing an issue. Central to enabling this are the MXNet autograd package and the Gluon trainer method. You can define a model training algorithm that consists of a for loop, using autograd to automatically calculate derivatives of the model’s parameters and trainer to optimize them.


```bash
epochs = 10

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        with autograd.record():
            output = net(data) # the forward iteration
            loss = softmax_cross_entropy(output, label)
            loss.backward()
        trainer.step(data.shape[0])
```

<br/>
**3. Dynamic Graphs**

In certain scenarios, the neural network model may need to change in shape and size during model training. This is necessary in particular when the data fed into the neural network is variable, which is common in Natural Language Processing (NLP) where each sentence inputted can be of different length. With Gluon, neural network definition can be dynamic, meaning you can build it on the fly, with any structure you want, and using any of Python’s native control flow. In the below code snippet, you can see how to incorporate a loop in each forward iteration of model training.


```bash
def forward(self, F, inputs, tree):
    children_outputs = [self.forward(F, inputs, child)
                        for child in tree.children]
    #Recursively builds the neural network based on each input sentence’s
    #syntactic structure during the model definition and training process
    …
```
<br/>
**4. High Performance**

When speed becomes more important than flexibility, the Gluon interface enables you to easily cache the neural network model to achieve high performance. This only requires a small tweak when you set up your neural network – this is after you are done with your prototype and ready to test it on a larger dataset. Instead of using ***Sequential*** (as shown above) to stack the neural network layers, you must use ***HybridSequential***. Its functionality is the same as ***Sequential***, but it lets you call down to the underlying optimized engine to express some or all of your model’s architecture.


```bash
net = nn.HybridSequential()
with net.name_scope():
    net.add(nn.Dense(256, activation="relu"))
    net.add(nn.Dense(128, activation="relu"))
    net.add(nn.Dense(2))
```
Next, to compile and optimize the ***HybridSequential***, we can then call its ***hybridize*** method:

```bash
net.hybridize()
```
