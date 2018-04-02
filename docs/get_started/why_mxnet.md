# Why MXNet?

Probably, if you've stumbled upon this page, you've heard of _deep learning_.
Deep learning denotes the modern incarnation of neural networks,
and it's the technology behind recent breakthroughs
in self-driving cars, machine translation, speech recognition and more.
While widespread interest in deep learning took off in 2012,
deep learning has become an indispensable tool for countless industries.

![alt text](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/get-started/image-classification.png)

It might not come as a surprise that researchers
have investigated neural networks for decades.
Warren McCulloch and Walter Pitts
suggested the forerunner of today's artificial neurons back in 1943.
Each neuron is connected to other neurons along _edges_, analogous to the synapses that connect real neurons. 
And associated with each edge is a _weight_ that indicates whether the connection is excitatory or inhibitatory and the strength of the connection. 

![alt_text](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/get-started/artificial-neuron-2.png)

In the 1980s, the modern version of neural networks took shape.
Researchers arranged artificial neurons into _layers_.
Neurons in any layer get input from the neurons in the layers below them.
And, in turn, their output feeds into the neurons in the layer above.
Typically, the lowest layer represents the _input_ to a neural network.
After computing the values of each layer, the _output_ values are read out from the topmost layer.
The behavior of the network is determined by the setting of the weights. 
And the process of _learning_ in neural networks 
is precisely the process of searching for good settings of these _weights_.

All that we need is an algorithm that tells us how to perform this search.
And since David Rumelhart and colleagues
introduced the _backpropagation_ learning algorithm to train neural networks,
nearly all the major ideas have been in place.
Still, for many years neural networks took a backseat 
to classical statistical methods like logistic regression and support vector machines (SVMs).
So you might reasonably ask, what's changed to garner such interest?

## Scale and Computation
The two biggest factors driving innovation in deep learning now are data and computation.
With distributed cloud computing and parallelism across GPU cores,
we can train models millions of times faster than researchers could in the 1980s.
The availability of large, high-quality datasets is another factor driving the field forward.
In the 1990s, the best datasets in computer vision had thousands of low-resolution images and ground truth assignments to a small number of classes.
Today, researchers cut their teeth on ImageNet, a massive dataset containing millions of high-resolution images from a thousand distinct classes.
The falling price of storage and high network bandwidth
make it affordable to work with big data at will.

In this new world, with bigger datasets and abundant computation, 
neural networks dominate on most pattern recognition problems.
Over the last five years, neural networks have come to dominate on nearly every problem in computer vision,
replacing classical models and hand-engineered features.
Similarly, nearly every production speech recognition system now relies on neural networks, 
where replacing the hidden Markov models that previously held sway.

![alt text](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/get-started/nvidia-gpus.jpg)

While GPUs and clusters present a huge opportunity for accelerating neural network training,
adapting traditional machine learning code
to take advantage of these resources can be challenging.
The familiar scientific computing stacks (Matlab, R, or NumPy & SciPy)
give no straight-forward way to exploit these distributed resources.

Acceleration libraries like _MXNet_ offer powerful tools
to help developers exploit the full capabilities of GPUs and cloud computing.
While these tools are generally useful and applicable to any mathematical computation, _MXNet_ places a special emphasis on speeding up the development and deployment of large-scale deep neural networks. In particular, we offer the following capabilities:
* __Device Placement:__ With _MXNet_, it's easy to specify where each data structures should live.
* __Multi-GPU training__: _MXNet_ makes it easy to scale computation with number of available GPUs.
* __Automatic differentiation__: _MXNet_ automates the derivative calculations that once bogged down neural network research.
* __Optimized Predefined Layers__: While you can code up your own layers in _MXNet_, the predefined layers are optimized for speed, outperforming competing libraries.

## Deep Nets on Fast Computers
While MXNet can accelerate any numerical computation,
we developed the library with neural networks in mind.
However you plan to use MXNet, neural networks make for a powerful motivating example to display MXNet's capabilities.

Neural networks are just functions for transforming input arrays `X` into output arrays `Y`.
In the case of image classification, `X` might represent the pixel values of an image, and `Y` might represent the corresponding probabilities that the image belongs to each of `10` classes.
For language translation, `X` and `Y` both might denote sequences of words. We'll revisit the way you might represent sequences in subsequent tutorials - so for now it's safe to think of `X` and `Y` as fixed length vectors.

To perform this mapping, neural networks stack _layers_ of computation. Each layer consists of a linear function followed by a nonlinear transformation. In _MXNet_ we might express this as:
~~~~
hidden_linear = mx.sym.dot(X, W)
hidden_activation = mx.sym.tanh(hidden_linear)
~~~~
The linear transformations consist of multiplication by parameter arrays (`W` above). 
When we talk about learning we mean finding the right set of values for `W`.
With just one layer, we can implement the familiar family of linear models, 
including linear and logistic regression, linear support vector machines (SVMs), and the perceptron algorithm.
With more layers and a few clever constraints, we can implement all of today's state-of-the-art deep learning techniques.

Of course, tens or hundreds of matrix multiplications can be computationally taxing. 
Generally, these linear operations are the computational bottleneck.
Fortunately, linear operators can be parallelized trivially across the thousands of cores on a GPU.
But low-level GPU programming requires specialized skills that are not common even among leading researchers in the ML community. Moreover, even for CUDA experts, implementing a new neural network architecture shouldn't require weeks of programming to implement low-level linear algebra operations. That's where _MXNet_ comes in.
*  _MXNet_ provides optimized numerical computation for GPUs and distributed ecosystems, from the comfort of high-level environments like Python and R
* _MXNet_ automates common workflows, so standard neural networks can be expressed concisely in just a few lines of code

Now let's take a closer look at the computational demands of neural networks 
and give a sense of how _MXNet_ helps us to write better, faster, code.
Say we have a neural network trained to recognize spam from the content of emails. 
The emails may be streaming from an online service (at inference time),
or from a large offline dataset __D__ (at training time). 
In either case, the dataset typically must be managed by the CPU.

![alt text](https://raw.githubusercontent.com/kevinthesun/web-data/master/mxnet/get-started/architecture.png)

To compute the transformation of a neural network quickly, we need both the parameters and data points to make it into GPU memory. For any example _X_, the parameters _W_ are the same. Moreover the size of the model tends to dwarf the size of an individual example. So we might arrive at the natural insight that parameters should always live on the GPU, even if the dataset itself must live on the CPU or stream in. This prevents IO from becoming the bottleneck during training or inference.

Fortunately, _MXNet_ makes this kind of assignment easy.
~~~~
import mxnet.ndarray as nd

X  = nd.zeros((10000, 40000), mx.cpu(0))           #Allocate an array to store 1000 datapoints (of 40k dimensions) that lives on the CPU
W1 = nd.zeros(shape=(40000, 1024), mx.gpu(0))      #Allocate a 40k x 1024 weight matrix on GPU for the 1st layer of the net
W2 = nd.zeros(shape=(1024, 10), mx.gpu(0))         #Allocate a 1024 x 1024 weight matrix on GPU for the 2nd layer of the net
~~~~

<!-- * __Talk about how mxnet also makes it easy to assign a context (on which device the computation happens__ -->
Similarly, _MXNet_ makes it easy to specify the computing device

~~~~
with mx.Context(mx.gpu()):          # Absent this statement, by default, MXNet will execute on CPU
    h = nd.tanh(nd.dot(X, W1))
    y = nd.sigmoid(nd.dot(h1, W2))
~~~~

Thus, with only a high-level understanding of how our numerical computation maps onto an execution environment, _MXNet_ allows us to exert fine-grained control when needed.

## Nuts and Bolts

MXNet supports two styles of programming: _imperative programming_ (supported by the _NDArray_ API) and _symbolic programming_ (supported by the _Symbol_ API). In short, imperative programming is the style that you're likely to be most familiar with. Here if A and B are variables denoting matrices, then `C = A + B` is a piece of code that _when executed_ sums the values referenced by `A` and `B` and stores their sum `C` in a new variable. Symbolic programming, on the other hand, allows functions to be defined abstractly through computation graphs. In the symbolic style, we first express complex functions in terms of placeholder values. Then, we can execute these functions by _binding them_ to real values.


### Imperative Programming with _NDArray_
If you're familiar with NumPy, then the mechanics of _NDArray_ should be old hat. Like the corresponding `numpy.ndarray`, `mxnet.ndarray` (`mxnet.nd` for short) allows us to represent and manipulate multi-dimensional, homogenous arrays of fixed-size components. Converting between the two is effortless:

~~~~
# Create a numpy array from an mxnet NDArray
A_np = np.array([[0,1,2,3,4],[5,6,7,8,9]])
A_nd = nd.array(A)  

# Convert back to a numpy array
A2_np = A_nd.asnumpy()
~~~~

Other deep learning libraries tend to rely on NumPy exclusively for imperative programming and the syntax.
So you might reasonably wonder, why do we need to bother with _NDArray_?
Put simply, other libraries only reap the advantages of GPU computing when executing symbolic functions. By using _NDArray_, _MXNet_ users can specify device context and run on GPUs. In other words, _MXNet_ gives you access to the high-speed computation for imperative operations that Tensorflow and Theano only give for symbolic operations.


~~~~
X = mx.nd.array([[1,2],[3,4]])
Y = mx.nd.array([[5,6],[7,8]])
result = X + Y
~~~~


### Symbolic Programming in _MXNet_

In addition to providing fast math operations through NDArray, _MXNet_ provides an interface for defining operations abstractly via a computation graph.
With `mxnet.symbol`, we define operations abstractly in terms of place holders. For example, in the following code `a` and `b` stand in for real values that will be supplied at run time.
When we call `c = a+b`, no numerical computation is performed. This operation simply builds a graph that defines the relationship between `a`, `b` and `c`. In order to perform a real calculation, we need to bind `c` to real values.

~~~~
a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
c = a + b
executor = c.bind(mx.cpu(), {'a': X, 'b': Y})
result = executor.forward()
~~~~


Symbolic computation is useful for several reasons. First, because we define a full computation graph before executing it, _MXNet_ can perform sophisticated optimizations to eliminate unnecessary or repeated work. This tends to give better performance than imperative programming. Second, because we store the relationships between different variables in the computation graph, _MXNet_ can then perform efficient auto-differentiation.


## Building Models with _MXNet_ Layers
In addition to providing a general-purpose toolkit for optimizing mathematical operations,
_MXNet_ provides predefined neural network layers.
This higher-level interface has several benefits.

* Predefined layers allow you to express large models concisely. Repetitive work, like allocating parameters and inferring their dimensions are eliminated.
* Layers are written directly in C++, giving even faster performance than equivalent layers implemented manually in Python.


~~~~
data = mx.symbol.Variable('data')
fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=num_classes)
mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
~~~~


## Conclusions
Given its combination of high performance, clean code, access to a high-level API, and low-level control, _MXNet_ stands out as a unique choice among deep learning frameworks.

