# Optimizing Memory Consumption in Deep Learning

In deep learning, we strive to train deeper and larger networks. Although hardware has improved rapidly in recent years, the huge deep network monsters are
always hungry for more GPU RAM. Using less memory for the same network also means we can
use a larger batch size and, usually, a higher GPU utilization rate.

This topic discusses how to optimize memory allocation for deep neural networks, and provides
 candidate solutions for some of the problems. The solutions are by no means complete;
they are examples that we think can be useful in most cases.

## Computation Graph

We'll start the discussion by introducing the computation graph because it's a tool that we will rely on later. A computation graph describes the (data flow) dependencies between the operations in the deep network.
The operations performed in the graph can be either fine-grained or coarse-grained.
The following figure shows two examples of computation graphs.

![Comp Graph Example](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/comp_graph_example.png)

The concept of a computation graph is deeply rooted in packages such as Theano and CGT. Actually, they also exist implicitly
in most libraries as the network configuration. The major difference in these libraries comes down to how they calculate gradient.
There are mainly two ways: doing back-propagation on the same graph or having an explicit backward path that calculates
the gradient needed.

![Backward Graph](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/back_graph.png)

Libraries like Caffe, CXXNet, and Torch use the backprop on the same graph. Libraries like Theano and CGT takes the explicit
backward path approach. In this topic, we adopt the *explicit backward path* approach because it has several advantages
for optimization.

However, we should emphasize that choosing the explicit backward path approach doesn't restrict us
to symbolic libraries, such as Theano and CGT. We can also use the explicit backward path for gradient calculation of
layer-based (which ties forward and backward together) libraries. The following graph shows how to do this.
Basically, we introduce a backward node that links to the forward node of the graph and calls the ```layer.backward```
in the backward operations.

![Backward Layer](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/explicit_back_layer.png)

This discussion applies to almost all existing deep learning libraries.
(There are differences between libraries,  e.g., high-order differentiation, which is beyond the scope of this topic.)

Why is the explicit backward path better? Let's explain it with two examples. The first reason is that the explicit backward path
clearly describes the dependency between computations. Consider the following case, where we want to get
the gradient of A and B. As we can see clearly from the graph, the computation of the ```d(C)``` gradient doesn't depend on F.
This means that we can free the memory of ```F``` right after the forward computation is done. Similarly the memory
of ```C``` can be recycled.

![Backward Prune](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/back_dep_prune.png)

Another advantage of the explicit backward path is the ability to have a different backward path, instead of a mirror of forward one.
A common example is the split connection case, as shown in the following figure.

![Backward Agg](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/back_agg_grad.png)

In this example, the output of B is referenced by two operations. If we want to do the gradient calculation in the same
network, we need to introduce an explicit split layer. This means we need to do the split for the forward pass, too.
In this figure, the forward pass doesn't contain a split layer, but the graph will automatically insert a gradient
aggregation node before passing the gradient back to B. This helps us to save the memory cost of allocating the output of the
split layer, and the operation cost of replicating the data in the forward pass.

If we adopt the explicit backward approach, there's no difference between the forward pass
and the backward pass. We simply proceed in the chronological order of the computation graph, and carry out computations.
This also simplifies our discussions. The problem now is: How do we allocate the memory of each output node of a computation graph?

This seems to have nothing to do with deep learning, but is more about the context of compiling, data flow optimization, etc.
It's really the hungry monster of deep learning that motivates us to attack this problem, and benefits from it.

## What Can Be Optimized?

We hope you're convinced that the computation graph is a good way to discuss memory allocation optimization techniques.
As you can see, we've already saved some memory by using the explicit backward graph. Let's explore
what we can optimize, and determine the baseline.

Assume that we want to build a neural network with ```n``` layers. A typical implementation of a neural network will
need to allocate node space for the output of each layer and gradient values for back-propagation.
This means we need roughly ```2 n``` memory cells. This is the same in the explicit backward graph approach because
the number of nodes in a backward pass is roughly the same as in a forward pass.

### In-place Operations
One of the very first things that we can do is in-place memory sharing of operations. This is usually done for
simple operations such as activation functions. Consider the following case, where we want to
compute the value of three chained sigmoid functions.

![Inplace op](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/alloc_inline.png)

Because we can compute sigmoid ```in-place```, that is, use the same memory for input and output, we can simply allocate one copy of memory, and use it to compute an arbitrary length of sigmoid chain.

However, in-place optimization sometimes can be done incorrectly, especially when the package tries
to be a bit general. Consider the following case, where the value of B is not only used by C, but also by F.

![In-place trap](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/alloc_inline_trap.png)

We can't perform in-place optimization because the value of B is still needed after ```C=sigmoid(B)``` is computed.
An algorithm that simply does in-place optimization for every sigmoid operation might fall into such trap,
so we need to be careful about when we can use it.

### Standard Memory Sharing
Memory can be shared in other ways than in in-place operations. In the following case, because the
value of B is no longer needed when we compute E, we can reuse the memory to hold the result of E.

![Normal Sharing](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/alloc_normal.png)

*Memory sharing doesn't necessarily require the same data shape*.
In the preceding example, the shape of ```B``` and ```E``` can differ. We can simply allocate a
memory region that is the maximum of the two sizes and shares it between them.

### Example of Real Neural Network Allocation
The preceding examples, which contain only the computation of the forward pass, are made up.
But, the same idea holds for real neural networks. The following figure shows an allocation
plan for a two-layer perceptron.

![Net Alloc](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/alloc_mlp.png)

In this example:

- In-place optimization is applied on computing ```act1```, ```d(fc1)```, ```out``` and ```d(fc2)```.
- Memory is shared between ```d(act1)``` and ```d(A)```.

## Memory Allocation Algorithm

We've discussed general techniques for optimizing memory allocation.
We've also seen that there are traps to avoid, like the one we saw for in-place memory optimization.
So, how can we allocate memory correctly? This is not a new problem. For example, it is very similar
to the problem with register allocation in compilers. There might be techniques that we can borrow. We're not attempting to give
a comprehensive review of techniques here, but rather to introduce some simple, but useful tricks to attack
the problem.

The key problem is that we need to place resources so that they don't conflict with each other.
More specifically, each variable has a ```life time``` between the time it gets computed until the last time it is used.
In the case of the multi-layer perceptron, the ```life time``` of ```fc1``` ends after ```act1``` get computed.

![Net Alloc](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/alloc_mlp.png)

The principle is *to allow memory sharing only between variables whose lifetimes don't overlap*. There are multiple
ways to do this. You can construct the conflicting graph with each variable as a node and link the edge
between variables with overlapping lifespans, and then run a graph-coloring algorithm. This likely has ```$O(n^2)$```
complexity, where ```n``` is the number of nodes in the graph. This might be too costly.

Let's consider another simple heuristic. The idea is to simulate the procedure of traversing the graph,
and keep a counter of future operations that depends on the node.

![Alloc](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/alloc_step.png)

- An in-place optimization can be performed when only the current operation depends on the source (i.e., counter=1).
- Memory can be recycled into the box on the upper right corner when the counter goes to 0.
- When we need new memory, we can either get it from the box or allocate a new one.

***Note:*** During the simulation, no memory is allocated. Instead, we keep a record of how much memory each node needs,
and allocate the maximum of the shared parts in the final memory plan.

## Static vs. Dynamic Allocation

The preceding strategy exactly simulates the dynamic memory allocation procedure in imperative
languages, such as Python. The counter is the reference counter of each memory object, and the object gets garbage collected when
the reference counter goes to 0. In that sense, we are simulating dynamic memory allocation once to create a static allocation plan.
Can we simply use an imperative language that dynamically allocates and deallocates memory?

The major difference is that static allocation is only done once, so we can afford to use more complicated algorithms. For example, we can search for memory sizes that are similar to the required memory block. The Allocation can also be made graph aware. We'll talk about that in the next section. Dynamic allocation puts more pressure on fast memory allocation and garbage collection.

There is also one takeaway for users who want to rely on dynamic memory allocations: *do not unnecessarily reference objects*. For example, if we organize all of the nodes in
a list and store then in a Net object, these nodes will never get dereferenced, and we gain no space.
Unfortunately, this is a common way to organize code.


## Memory Allocation for Parallel Operations

In the previous section, we discussed how we can *simulate* running the procedure for a computation graph to get a static allocation plan. However, optimizing for parallel computation presents other challenges
because resource sharing and parallelization are on the two ends of a balance.
Let's look at the following two allocation plans for the same graph:

![Parallel Alloc](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/parallel_alloc.png)

Both allocation plans are valid if we run the computation in serially, from ```A[1]``` to ```A[8]```.
However, the allocation plan on the left introduces additional dependencies, which means we can't
run computation of ```A[2]``` and ```A[5]``` in parallel. The plan on the right can.

To parallelize computation, we need to take greater care.

### Be Correct and Safe First
Being correct is our first principle. This means to execute in a way that takes implicit dependency
memory sharing into consideration. You can do this by adding the implicit dependency edge to the execution graph.
Or, even simpler, if the execution engine is mutated aware, as described in the
[dependency engine topic](http://mxnet.io/architecture/note_engine.html), push the operation
in sequence and write to the same variable tag that represents the same memory region.

Always produce a safe memory allocation plan. This means never allocate the same memory to nodes that can
be parallelized. This might not be ideal because sometimes memory reduction is more desirable, and we don't gain too
much when we can get multiple computing stream execution on the same GPU.

### Try to Allow More Parallelization
Now we can safely perform some optimizations. The general idea is to try to
encourage memory sharing between nodes that can't be parallelized. You can do this by creating an ancestor relationship
graph and querying it during allocation, which costs approximately ```$O(n^2)$``` in time to construct. We can also use a heuristic here,
for example, color the path in the graph.
As shown in the following figure, when you try to find the longest paths in the graph, color them the same color,
and continue.

![Path Color](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/graph_color.png)

After you get the color of the node, you allow sharing (or encourage sharing) only between nodes of the same color.
This is a stricter version of the ancestor relationship, but it costs only ```$O(n)$``` of time if you search for only the first ```k``` path.

This is by no means the only solution. More sophisticated approaches might exist

## How Much Can you Save?

We've discussed the techniques and algorithms you can use to squeeze memory usage for deep learning.
How much can you really save by using these techniques?

On coarse-grained operation graphs that are already optimized for big operations, you can reduce memory consumption roughly *by half*. You can reduce memory usage even more if you are optimizing a fine-grained computation network used by symbolic libraries, such as Theano.

Most of the ideas in this article inspired the design of MXNet.
We've also provided a [Memory Cost Estimation Script](https://github.com/dmlc/mxnet/tree/master/example/memcost),
which you can use to see how much memory you need under different scenarios.

The script has an option called ```forward_only```, which shows the cost of running only the forward pass.
You will find that cost when using this option is extremely low compared to others. This is simply because there's  more memory reuse if you run only the forward pass. So here are two takeaways:

- Use a computation graph to allocate memory.
- Deep learning prediction consumes much less memory than deep learning training.

## Contributing to This Topic

This topic is part of our effort to provide [open-source system design notes](index.md)
 for deep learning libraries. We welcome your contributions.
## Next Steps

* [Efficient Data Loading Module for Deep Learning](http://mxnet.io/architecture/note_data_loading.html)
* [Survey of RNN Interface](http://mxnet.io/architecture/rnn_interface.html)
