Execution Engine
================
MXNet's engine is not only for deep learning or any domain-specific problem. Rather, it is designed to face a general problem: execute a bunch of functions following their dependencies. Execution of any two functions with dependencies should be serialized.
Functions with no dependencies *may* be executed in parallel to boost performance.
See also [Note on Dependency Engine](note_engine.md) for general discussions on the topic.

Interface
---------
The core interface of execution engine is:
```c++
virtual void PushSync(Fn exec_fun, Context exec_ctx,
                      std::vector<VarHandle> const& const_vars,
                      std::vector<VarHandle> const& mutate_vars) = 0;
```
This API allows users to push a function (`exec_fun`), along with its context information and dependencies to the engine. The `exec_ctx` is the context information in which the `exec_fun` should be executed. `const_vars` denotes the variables that the function would read from while `mutate_vars` are the variables that to be modified. Regardless of the details that would be explained later, the engine guarantees following order:

>*The execution of any two functions that any one of them modifies at least one common variable would be serialized in their push order.*

Function
--------
The function type of the engine is:
```c++
using Fn = std::function<void(RunContext)>;
```
The `RunContext` contains runtime information which is determined by the engine:
```c++
struct RunContext {
    // stream pointer which could be safely cast to
    // cudaStream_t* type
	void *stream;
};
```
Alternatively, one could use `mxnet::engine::DAGEngine::Fn` which is the same type defination.

All the functions will be executed by the internal threads of the engine. In such model, it is usually not suggested to push *blocking* functions to the engine (usually for dealing with I/O tasks like disk, web service, UI, etc.) since it will occupy the execution thread and reduce the total throughput. In such case, we provide another *asynchronous* function type:
```c++
using Callback = std::function<void()>;
using AsyncFn = std::function<void(RunContext, Callback)>;
```
In the `AsyncFn` function, user could pass the heavy part to their own threads and safely exit the function body. The engine will not consider the function to be finished until the `Callback` function is called.

Context
--------
User could specify the `Context` of the function to be executed within. This usually includes whether the function should be run on CPU or GPU, and if GPU, which GPU to use. `Context` is different from `RunContext`. `Context` contains device type (gpu/cpu) and device id while `RunContext` contains information that could only be decided during runtime like on which stream the function should be executed.

VarHandle
--------
`VarHandle` is used to specify the dependencies of functions. The design of MXNet engine is to decouple it with other modules in MXNet. So `VarHandle` is like an engine-given token for user to represent the external resources the functions may use or modified. It is designed to be light, so create, delete or copy a variable will incur little overhead. Upon pushing functions, users need to specify the variables that will be used (immutable) in `const_vars` vector and the variables to be modified (mutable) in `mutate_vars` vector. The only rule for the engine to resolve the dependencies among functions pushed is:

>*The execution of any two functions that any one of them modifies at least one common variable would be serialized in their push order.*

For example, if `Fn1`, `Fn2` both mutate `V2`, `Fn2` is guaranteed to be executed after `Fn1` if `Fn2` is pushed after `Fn1`. On the other hand, if `Fn1` and `Fn2` both use `V2`, their actual execution order could be any kind.

This design allows the engine to schedule *state-mutating* operations. For example, the weight update function in DNN can now use `+=` operator to update the weights in place, rather than generating a new weight array each time.

To create a variable, use `NewVar()` API. To delete a variable, use `PushDelete` API.

Push & Wait
-----------
**All `Push` APIs are asynchronous.** The API call will return immediately no matter the pushed `Fn` is finished or not. This allows engine to start computing at the same time user thread is pushing functions. All `Push` APIs are not thread-safe. To be specific, only one thread should make engine API calls at one time.

If you want to wait for a specific `Fn` to be finished, include a callback function in the closure and call the function at the end of your `Fn`.

If you want to wait for all `Fn` that involves (use/mutate) a certain variable to be finished, use `WaitForVar(var)` API.

If you want to wait for all pushed `Fn` to be finished, use `WaitForAll()` API.

Save Object Creation Cost
-------------------------
In some cases, you need to push several functions to the engine but for tons of times. If the computation of these functions are light, the overhead of copying lambdas and creating use/mutate variable lists would become relatively high. We provide an API to create an `OprHandle` beforehand:
```c++
virtual OprHandle NewOperator(AsyncFn fn,
                              std::vector<VarHandle> const& const_vars,
                              std::vector<VarHandle> const& mutate_vars) = 0;
```
So you could keep pushing the `OprHandle` without repeatedly creating them:
```c++
virtual void Push(OprHandle op, Context exec_ctx) = 0;
```
To delete it, simply call `DeleteOperator(OprHandle op)`. But please make sure the operator has finished computing.


API Reference
-------------
```eval_rst
.. doxygenclass:: mxnet::Engine
   :members:
```
