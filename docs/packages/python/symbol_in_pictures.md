Symbolic Configuration and Execution in Pictures
================================================
This is a self-contained tutorial that explains the Symbolic construction and execution in pictures.
You are recommended to read this together with [Symbolic API](symbol.md).

Compose Symbols
---------------
Symbols are a description of computation we want to do. The symbolic construction API generates the computation
graph that describes what computation is needed. The following picture shows how we compose symbols to describe basic computations.

![Symbol Compose](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/symbol/compose_basic.png)

- The ```mxnet.symbol.Variable``` function creates argument nodes that represent input to the computation.
- The Symbol is overloaded with basic element-wise arithmetic operations. 

Configure Neural Nets
---------------------
Besides fine-grained operations, mxnet also provide a way to perform big operations that is analogous to layers in neural nets.
We can use these operators to describe a neural net configuration.

![Net Compose](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/symbol/compose_net.png)


Example of Multi-Input Net
--------------------------
The following is an example of configuring multiple input neural nets.

![Multi Input](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/symbol/compose_multi_in.png)


Bind and Execute Symbol 
-----------------------
When we need to execute a symbol graph, we call the bind function to bind ```NDArrays``` to the argument nodes
to obtain an ```Executor```.

![Bind](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/symbol/bind_basic.png)

You can call ```Executor.Forward``` to get the output results, given the bound NDArrays as input.

![Forward](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/symbol/executor_forward.png)


Bind Multiple Outputs
---------------------
You can use ```mx.symbol.Group``` to group symbols together then bind them to 
get outputs of both.

![MultiOut](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/symbol/executor_multi_out.png)

But remember: only bind what you need, so that the system can do more optimizations for you.


Calculate the Gradient
------------------
In the bind function, you can specify NDArrays that will hold gradients. Calling ```Executor.backward``` after ```Executor.forward```
will give you the corresponding gradients.

![Gradient](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/symbol/executor_backward.png)


Simple Bind Interface for Neural Nets
-------------------------------------
Sometimes it is tedious to pass the argument NDArrays to the bind function, especially when you are binding a big
graph. ```Symbol.simple_bind``` provides a way to simplify
the procedure. You only need to specify input data shapes, and the function will allocate the arguments, and bind
the Executor for you.

![SimpleBind](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/symbol/executor_simple_bind.png)

Auxiliary States
----------------
Auxiliary states are just like arguments, except that you cannot take the gradient of them. These are states that may 
not be part of the computation, yet can be helpful to track. You can pass the auxiliary state in the same way as arguments.

![SimpleBind](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/symbol/executor_aux_state.png)

More Information
----------------
Please refer to [Symbolic API](symbol.md) and [Python Documentation](index.md).
