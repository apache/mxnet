Execution Engine
================

MXNet 的执行引擎不仅仅是为了深度学习和其他任何特定的领域问题. 相反地, 它设计用来解决通用问题: 根据依赖关系来执行一系列的功能操作. 有依赖关系的任意两个功能需要被序列化. 没有依赖的功能 *可以* 并发执行来提升系统性能. 也可以参考 [Note on Dependency Engine](note_engine.md).

Interface
---------
执行引擎的核心接口如下:

```c++
virtual void PushSync(Fn exec_fun, Context exec_ctx,
                      std::vector<VarHandle> const& const_vars,
                      std::vector<VarHandle> const& mutate_vars) = 0;
```
这个 API 用户将一个 函数 (`exec_fun`)连同它的上下文信息以及依赖关系 push 到执行引擎. `exec_ctx` 是 `exec_fun` 执行的上下文环境. `const_vars` 代表的是函数只有读取权限的变量, `mutate_vars` 表示的是函数可以修改的变量. 先不考虑具体的细节, 执行引擎保证下面的规则:

>*任意两个会修改同一个变量的函数,会根据它们 push 到引擎的顺序进行序列化.*

Function
--------

执行引擎需要的函数类型按照下面方式来定义:
```c++
using Fn = std::function<void(RunContext)>;
```
`RunContext` 包含了引擎确定的运行时信息:

```c++
struct RunContext {
    // stream pointer which could be safely cast to
    // cudaStream_t* type
	void *stream;
};
```
用户也可以使用 `mxnet::engine::DAGEngine::Fn` 定义作为第二种选择, 它们的类型是一样的.


所有的函数都会被 engine 内部的线程来执行. 在这个模型中, 我们不鼓励用户将 *阻塞 (blocking)* 函数 push 到引擎 ( 通常是处理 I/O 任务的函数, 比如读取硬盘, web 服务, UI, 等等). 因为阻塞函数会占用执行线程, 同时降低了这个系统的吞吐量. 这种情况下, 我们提供了另外的一种 *asynchronous* 函数类型:

```c++
using Callback = std::function<void()>;
using AsyncFn = std::function<void(RunContext, Callback)>;
```

在 `AsyncFn` 函数中, 用户可以将重要的计算交由自己的线程来执行, 同时不用等待函数执行结束. 除非异步函数的 `Callback` 被调用, 否则引擎不会考虑函数是否已经结束的事情.

Context
--------

用户可以指定函数执行的需要的 `Context`. 这个 `Context` 一般包括函数是否执行在 CPU 或者 GPU 上, 如果是 GPU, 那么具体是哪个 GPU. `Context` 和 `RunContext` 是不一样的. `Context` 包括设备类型 (gpu/cpu) 和设备 id,  而 `RunContext` 包含的是只有在运行时才可以确定的信息, 比如说函数要在哪个 stream 上执行.


VarHandle
--------

`VarHandle` 是 用来指定函数的依赖关系的. MXNet 执行引擎的设计目的是为了接口和 MXNet 的其他模块解耦合. 所以 `VarHandle` 类似引擎为用户提供用来代表函数需要或者会修改的外部资源的一个令牌. 它被设计成轻量级的, 所以创建, 删除或者拷贝一个变量只需要一点点开销. 对于正在推送到引擎的函数, 用户需要在 `const_vars` vector 里指定需要的不可变变量,  在 `mutate_vars` vector 里指定会被修改的变量. 执行引擎解析函数之间的依赖关系唯一的规则是:


>**任意两个会修改同一个变量的函数,会根据它们 push 到引擎的顺序进行序列化.*

举个例子, 如果 `Fn1`和`Fn2`都要修改 `V2`, 那么如果 `Fn2` 后于 `Fn1` push 到引擎, 那么引擎会保证 `Fn2` 会在 `Fn1` 之后执行. 另一方面, 如果 `Fn1` 和 `Fn2` 都使用但不修改 `V2`, 那么它们具体的执行顺序是任意的.

这种设计方式可以允许引擎可以调度 *状态改变(state-mutating)* 操作. 比如说, 在 DNN 中权重更新的函数可以用 `+=` 操作来原地更新权重, 而不是每次都产生一个新的权重数组.

如果要创建一个变量, 使用 `NewVar()` API. 如果要删除一个变量, 使用 `PushDelete` API.



Push & Wait
-----------

**所有的 `Push` API 是异步的.**  API 调用会在调用之后马上返回, 而不管 `Fn` 是否执行完与否. 这允许执行引擎可以在用户的线程推送函数到引擎的时候马上开始计算. 所有的 `Push` API 不是 thread-safe 的. 具体来说, 就是应该只有一个线程来调用 API.

如果你想要等待一个具体的 `Fn` 完成,  需要包含一个 callback 函数, 然后在你的 `Fn` 的最后调用前面的回调函数.

如果你要等待所有的对一个确定的变量的使用 (读取/修改) 的 `Fn`, 那么你应该调用 `WaitForVar(var)`  API.

如果你要等待所有推送到引擎的 `Fn` 都结束, 那么需要调用  `WaitForAll()` API.


Save Object Creation Cost
-------------------------

有些情况下, 你需要推送几个函数到引擎很多次. 如果这些函数的计算是轻量级的, 那么拷贝 Lambda 表达式和创建读/写变量的列表的开销的比例就会相当高. 我们提供了一个 API 来提前创建 `OprHandle`:

```c++
virtual OprHandle NewOperator(AsyncFn fn,
                              std::vector<VarHandle> const& const_vars,
                              std::vector<VarHandle> const& mutate_vars) = 0;
```
这样你就可以一直推送 `OprHandle`, 而不用每次都要创建:

```c++
virtual void Push(OprHandle op, Context exec_ctx) = 0;
```
可以通过调用 `DeleteOperator(OprHandle op)` 来删除它. 不过一定要确保它们已经执行完了.

API Reference
-------------
```eval_rst
.. doxygenclass:: mxnet::Engine
   :members:
```
