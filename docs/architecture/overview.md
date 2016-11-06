# MXNet System Architecture

![System Overview](https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/mxnet/system/overview.png)

Above image shows major modules/components of the MXNet system and their interaction. The modules are:
- Runtime Dependency Engine: Schedules and executes the
  operations according to their read/write dependency.
- Storage Allocator: Efficiently allocate and recycles memory blocks for GPU and
  CPU.
- Resource Manager: Manage global resources such as random number generator, temporal space.
- NDArray: Dynamic asynchronous n-dimensional arrays, provide flexible
  imperative programs for MXNet.
- Symbolic Execution: Static symbolic graph executor, provide efficient symbolic
  graph execution and optimization.
- Operator: Operators that defines static forward and gradient
  calculation(backprop).
- SimpleOp: Operators that extend to NDArray operators and symbolic operators
  in a unified fashion.
- Symbol Construction: Symbolic construction, provide a way to construct
  computation graph(net configuration)
- KVStore: Key-value store interface for easy parameter synchronizations.
- Data Loading(IO): Efficient distributed data loading and augmentation.

# MXNet System Components

## Execution Engine

MXNet's engine is not only for deep learning or any domain-specific problem. Rather, it is designed to face a general problem: execute a bunch of functions following their dependencies. Execution of any two functions with dependencies should be serialized.
Functions with no dependencies *may* be executed in parallel to boost performance.
See also [Note on Dependency Engine](note_engine.md) for general discussions on the topic.

### Interface

The core interface of execution engine is:
```c++
virtual void PushSync(Fn exec_fun, Context exec_ctx,
                      std::vector<VarHandle> const& const_vars,
                      std::vector<VarHandle> const& mutate_vars) = 0;
```
This API allows users to push a function (`exec_fun`), along with its context information and dependencies to the engine. The `exec_ctx` is the context information in which the `exec_fun` should be executed. `const_vars` denotes the variables that the function would read from while `mutate_vars` are the variables that to be modified. Regardless of the details that would be explained later, the engine guarantees following order:

>*The execution of any two functions that any one of them modifies at least one common variable would be serialized in their push order.*

### Function

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
Alternatively, one could use `mxnet::engine::DAGEngine::Fn` which is the same type definition.

All the functions will be executed by the internal threads of the engine. In such model, it is usually not suggested to push *blocking* functions to the engine (usually for dealing with I/O tasks like disk, web service, UI, etc.) since it will occupy the execution thread and reduce the total throughput. In such case, we provide another *asynchronous* function type:
```c++
using Callback = std::function<void()>;
using AsyncFn = std::function<void(RunContext, Callback)>;
```
In the `AsyncFn` function, user could pass the heavy part to their own threads and safely exit the function body. The engine will not consider the function to be finished until the `Callback` function is called.

### Context

User could specify the `Context` of the function to be executed within. This usually includes whether the function should be run on CPU or GPU, and if GPU, which GPU to use. `Context` is different from `RunContext`. `Context` contains device type (gpu/cpu) and device id while `RunContext` contains information that could only be decided during runtime like on which stream the function should be executed.

### VarHandle

`VarHandle` is used to specify the dependencies of functions. The design of MXNet engine is to decouple it with other modules in MXNet. So `VarHandle` is like an engine-given token for user to represent the external resources the functions may use or modified. It is designed to be light, so create, delete or copy a variable will incur little overhead. Upon pushing functions, users need to specify the variables that will be used (immutable) in `const_vars` vector and the variables to be modified (mutable) in `mutate_vars` vector. The only rule for the engine to resolve the dependencies among functions pushed is:

>*The execution of any two functions that any one of them modifies at least one common variable would be serialized in their push order.*

For example, if `Fn1`, `Fn2` both mutate `V2`, `Fn2` is guaranteed to be executed after `Fn1` if `Fn2` is pushed after `Fn1`. On the other hand, if `Fn1` and `Fn2` both use `V2`, their actual execution order could be any kind.

This design allows the engine to schedule *state-mutating* operations. For example, the weight update function in DNN can now use `+=` operator to update the weights in place, rather than generating a new weight array each time.

To create a variable, use `NewVar()` API. To delete a variable, use `PushDelete` API.

### Push & Wait

**All `Push` APIs are asynchronous.** The API call will return immediately no matter the pushed `Fn` is finished or not. This allows engine to start computing at the same time user thread is pushing functions. All `Push` APIs are not thread-safe. To be specific, only one thread should make engine API calls at one time.

If you want to wait for a specific `Fn` to be finished, include a callback function in the closure and call the function at the end of your `Fn`.

If you want to wait for all `Fn` that involves (use/mutate) a certain variable to be finished, use `WaitForVar(var)` API.

If you want to wait for all pushed `Fn` to be finished, use `WaitForAll()` API.

### Save Object Creation Cost

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


### API Reference

```eval_rst
.. doxygenclass:: mxnet::Engine
   :members:
```

## Operators in MXNet

An operator in MXNet is a class that contains both actual computation logic and auxiliary informations that could aid our system to perform optimizations like in-place updates and auto-derivative. Before continue on this document, it is strongly recommended for you to first understand `mshadow` library, since all operators compute on tensor-like structure `mshadow::TBlob` provided by the system during runtime. MXNet's operator interface tries its best to offer users flexibility including:
* Save memory allocation cost by specifying in-place updates.
* Hide some internal arguments from python side to make it cleaner.
* Define the relationships among input tensors and output tensors which allows system to perform shape check for you.
* Acquire additional temporary spaces from system to perform computation (e.g. calling `cudnn` routines).

### Operator Interface

The core interface of operator is `Forward`:
```c++
virtual void Forward(const OpContext &ctx,
                     const std::vector<TBlob> &in_data,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &out_data,
                     const std::vector<TBlob> &aux_states) = 0;
```
* The `OpContext` structure is like follows:
  ```c++
  struct OpContext {
    int is_train;
    RunContext run_ctx;
    std::vector<Resource> requested;
  }
  ```
  
  , where you could get whether the operator is in train/test phase; which device the operator should be run on (in `run_ctx`) and requested resources (covered in the following sections).
* `in_data` and `out_data` represent the input and output tensors respectively. All the tensor spaces have been allocated by the system.
* `req` denotes how the computation results are written into the `out_data`. In other word, `req.size() == out_data.size()` and `req[i]` corresponds to the write type of `out_data[i]`. The `OpReqType` is defined as:

  ```c++
  enum OpReqType {
    kNullOp,
    kWriteTo,
    kWriteInplace,
    kAddTo
  };
  ```
  Normally, the types of all `out_data` should be `kWriteTo`, meaning the provided `out_data` tensor is a *raw* memory block so the operator should directly write results into it. In some cases, for example when calculating the `gradient` tensor, it would be great if we could accumulate the result rather than directly overwrite the tensor contents so no extra space need to be created each time. In such case, the corresponding `req` type will be set as `kAddTo`, indicating a `+=` should be called.
* `aux_states` is intentionally designed for auxiliary tensors used to help computation. It is currently useless.

Apart from `Forward` operator, user could also optionally implement `Backward` interface defined as follows:
```c++
virtual void Backward(const OpContext &ctx,
                      const std::vector<TBlob> &out_grad,
                      const std::vector<TBlob> &in_data,
                      const std::vector<TBlob> &out_data,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &in_grad,
                      const std::vector<TBlob> &aux_states);
```
The interface follows the design principle as `Forward` interface, except that `out_grad`, `in_data` and `out_data` are given and the operator should computes `in_grad` as results. The name strategy is similar to torch's convention and could be summarized in following figure:

[input/output semantics figure]

Some operator may not need all the `out_grad`, `in_data` and `out_data`. This could be specified by the `DeclareBackwardDependency` interface in `OperatorProperty`.

### Operator Property

It is possible that one convolution has several implementations and users want to switch among them to achieve best performance. Therefore, we separate the operator *semantic* interfaces from the implementation interface (`Operator` class) into `OperatorProperty` class. The `OperatorProperty` interface consists of:

* **InferShape:**
  ```c++
  virtual bool InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape,
                          std::vector<TShape> *aux_shape) const = 0;
  ```
  There are two purposes of this interface: (1) tell system the size of each input and output tensor, so it could allocate them before the `Forward` and `Backward` call; (2) do size check to make sure there is no obvious error before running. The shape in `in_shape` would be set by the system (from the `out_shape` of the previous operators). Return `false` when there is not enough information to interence shapes or throw error when the shape is inconsistent.

* **Request Resources:** Operation like `cudnnConvolutionForward` need a workspace to help computation. It is would be nice if the system could manage that since the system then could do optimizations like reuse the space and so on. MXNet defines two interfaces to achieve this:
  ```c++
  virtual std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const;
  virtual std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const;
  ```
  The `ResourceRequest` structure (in `resource.h`) currently contains only a type flag:
  ```c++
  struct ResourceRequest {
    enum Type {
      kRandom,  // get an mshadow::Random<xpu> object
      kTempSpace,  // request temporay space
    };
    Type type;
  };
  ```
  If the `ForwardResource` and `BackwardResource` return non-empty arrays, system will offer the corresponding resources through the `ctx` parameter in the `Forward` and `Backward` interface of `Operator`. Basically, to access those resources, simply write:
  ```c++
  auto tmp_space_res = ctx.requested[kTempSpace].get_space(some_shape, some_stream);
  auto rand_res = ctx.requested[kRandom].get_random(some_stream);
  ``` 
  Please refer to `src/operator/cudnn_convolution-inl.h` for a concrete example.

* **Backward dependency:** Let us first see two different operator signatures (we name all the arguments for demonstration purpose):
  ```c++
  void FullyConnectedForward(TBlob weight, TBlob in_data, TBlob out_data);
  void FullyConnectedBackward(TBlob weight, TBlob in_data, TBlob out_grad, TBlob in_grad);

  void PoolingForward(TBlob in_data, TBlob out_data);
  void PoolingBackward(TBlob in_data, TBlob out_data, TBlob out_grad, TBlob in_grad);
  ```
  Note that the `out_data` in `FullyConnectedForward` is not used by `FullyConnectedBackward` while `PoolingBackward` requires all the arguments of `PoolingForward`. Therefore, for `FullyConnectedForward`, the `out_data` tensor once consumed by its consumers could be safely freed since backward function will not need it. This provides a chance for system to garbage collect some tensors as soon as possible. To specify this situation, we provide an interface:
  ```c++
  virtual std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const;
  ```
  The `int` element of the argument vector is an id to distinguish different arrays. Let us see how this interface specifies different dependencies for `FullyConnected` and `Pooling`:
  ```c++
  std::vector<int> FullyConnectedProperty::DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const {
    return {out_grad[0], in_data[0]};  // NOTE: out_data[0] is NOT included
  }
  std::vector<int> PoolingProperty::DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const {
    return {out_grad[0], in_data[0], out_data[0]};
  }
  ```

* **Inplace Option:** To further save memory allocation cost, in-place update are welcomed. This usually happens for element-wise operations when input tensor and output tensor are of the same shape. This could be specified by the following interface:
  ```c++
  virtual std::vector<std::pair<int, void*>> ElewiseOpProperty::ForwardInplaceOption(
      const std::vector<int> &in_data,
      const std::vector<void*> &out_data) const {
    return { {in_data[0], out_data[0]} };
  }
  virtual std::vector<std::pair<int, void*>> ElewiseOpProperty::BackwardInplaceOption(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data,
      const std::vector<void*> &in_grad) const {
    return { {out_grad[0], in_grad[0]} }
  }
  ```
  The above tells the system the `in_data[0]` and `out_data[0]` tensors could share the same memory spaces during `Forward`, and so do `out_grad[0]` and `in_grad[0]` during `Backward`.
  
  >**ATTENTION:** Even with the above specification, it is *not* guaranteed that input and output tensors will share the same space. In fact, this is only a hint for the system for the final decision. However, in either case, such decision is completely transparent to user, so the actual `Forward` and `Backward` implementation does not need to consider that.

* **Expose Operator to Python:** Due to the restriction of c++ language, we need user to implement following interfaces:
  ```c++
  // initial the property class from a list of key-value string pairs
  virtual void Init(const vector<pair<string, string>> &kwargs) = 0;
  // return the parameters in a key-value string map
  virtual map<string, string> GetParams() const = 0;
  // return the name of arguments (for generating signature in python)
  virtual vector<string> ListArguments() const;
  // return the name of output values
  virtual vector<string> ListOutputs() const;
  // return the name of auxiliary states
  virtual vector<string> ListAuxiliaryStates() const;
  // return the number of output values
  virtual int NumOutputs() const;
  // return the number of visible outputs
  virtual int NumVisibleOutputs() const;
  ```

### Create Operator from Operator Property

As mentioned above `OperatorProperty` includes all *semantical* attributes of an operation. It is also in charge of creating `Operator` pointer for actual computation.

#### Create Operator
Implement following interface in `OperatorProperty`:
```c++
virtual Operator* CreateOperator(Context ctx) const = 0;
```
For example:
```c++
class ConvolutionOp {
 public:
  void Forward( ... ) { ... }
  void Backward( ... ) { ... }
};
class ConvolutionOpProperty : public OperatorProperty {
 public:
  Operator* CreateOperator(Context ctx) const {
    return new ConvolutionOp;
  }
};
```

#### Parametrize Operator
When implementing convolution operator, we need to know the kernal size, the stride size, padding size and so on. These parameters should be passed to the operator before any `Forward` or `Backward` interface is called. To do so, user could define a `ConvolutionParam` structure:
```c++
#include <dmlc/parameter.h>
struct ConvolutionParam : public dmlc::Parameter<ConvolutionParam> {
  TShape kernel, stride, pad;
  uint32_t num_filter, num_group, workspace;
  bool no_bias;
};
```
Put it in `ConvolutionOpProperty` and pass it to the operator class during construction:
```c++
class ConvolutionOp {
 public:
  ConvolutionOp(ConvolutionParam p): param_(p) {}
  void Forward( ... ) { ... }
  void Backward( ... ) { ... }
 private:
  ConvolutionParam param_;
};
class ConvolutionOpProperty : public OperatorProperty {
 public:
  void Init(const vector<pair<string, string>& kwargs) {
    // initialize param_ using kwargs
  }
  Operator* CreateOperator(Context ctx) const {
    return new ConvolutionOp(param_);
  }
 private:
  ConvolutionParam param_;
};
```

#### Register Operator to MXNet
Use following macros to register the parameter structure and the operator property class to MXNet system:
```c++
DMLC_REGISTER_PARAMETER(ConvolutionParam);
MXNET_REGISTER_OP_PROPERTY(Convolution, ConvolutionOpProperty);
```
The first argument to the macro is the name string, the second is the property class name.

### All in a list

Finally! We almost covered the interface we needed to define a new operator. Let's do a recap in a list:
* Use `Operator` interface to write your actual computation logic (`Forward` and `Backward`).
* Use `OperatorProperty` interface to:
  - Pass parameter to operator class (may use `Init` interface).
  - Create operator using `CreateOperator` interface.
  - Correctly implement the operator description interface such as the names of arguments, etc.
  - Correctly implement the `InferShape` interface to set the output tensor shape.
  - [Optional] If additional resources are needed, check `ForwardResource` and `BackwardResource`.
  - [Optional] If `Backward` does not need all the input and output of `Forward`, check `DeclareBackwardDependency`.
  - [Optional] If inplace update is supported, check `ForwardInplaceOption` and `BackwardInplaceOption`.
* Register the `OperatorProperty` class and the parameter class.

## Unifying NDArray Operator and Symbolic Operator : How does it work
NDArray operations are similar to symbolic operations except the fact that sometimes we 
cannot write in place to the operands without a complete dependency graph. However, the 
logics underlying NDArray and Symbolic operation are almost the same. Unifying different 
invoking process and returning to the fundamental elements of operators are the purpose of 
**SimpleOp**, a new unified operator API. Because most mathematical operators attend to one or two 
operands and more operands make dependency-related optimization useful, the unified operator 
are specially designed for unary and binary operations.

Consider elements of an operation. Ideally, functions and derivatives are all we need to 
describe an operation. Let us restrict that to the space of unary and binary operations. How 
do we classify all operations to maximize the possibility of inplace write optimization? Note 
that functions can be separate out by the number of operands. Derivatives are a bit more 
complex. Whether output value, input data or neither are needed alongside head gradient is 
crucial to construct a dependency graph. Gradient functions in the unified API is thus 
differentiated through the types of operands it takes for calculation.

Before we continue on the SimpleOp interface, it is recommend to take a look at the [mshadow
library guide](https://github.com/dmlc/mshadow/tree/master/guide) since actual calculations 
will be done in `mshadow::TBlob` structure.

In this example, we will create a operator functioning as smooth l1 loss, which is a mixture 
of l1 loss and l2 loss. The loss itself can be written as:
```
loss = outside_weight .* f(inside_weight .* (data - label))
grad = outside_weight .* inside_weight .* f'(inside_weight .* (data - label))
```
where `.*` stands for element wise multiplication and `f`, `f'` is the smooth l1 loss function, 
which we suppose we have in `mshadow` for now. At first glance, it is impossible to implement 
this particular loss as an unary or binary operator. But we have automatic differentiation in 
the symbolic execution. That would simplify the loss to `f` and `f'` directly. In this way, this 
loss is no more complex than a `sin` or a `abs` function and can certainly be implemented as a 
unary operator.

## SimpleOp: the Unified Operator API
### Define Shapes
`mshadow` library require explicit memory allocation. As a consequence, all data shape
must be provided before any calculation. Before we proceed to define functions and gradient, 
we would like to check input data shape consistency and provide output shape.
```cpp
typedef TShape (*UnaryShapeFunction)(const TShape& src,
                                     const EnvArguments& env);
typedef TShape (*BinaryShapeFunction)(const TShape& lhs,
                                      const TShape& rhs,
                                      const EnvArguments& env);
```
We can use `mshadow::TShape` to check input data shape and designate the output data shape.
When this function is not defined, the default output shape will be the same as input shape.
In the case of binary operator, the shape of `lhs` and `rhs` is checked to be the same by default.

Shape functions can also be used to check if any additional arguments and resources are present.
Please refer to additional usages on `EnvArguments` to achieve this aim.

Before we start on our smooth l1 loss example, we define a `XPU` to `cpu` or `gpu` in the header 
`smooth_l1_unary-inl.h` implementation so that we reuse the same code in `smooth_l1_unary.cc` and 
`smooth_l1_unary.cu`.
```cpp
#include <mxnet/operator_util.h>
#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif
```

In our smooth l1 loss example, it is okay for the default behavior of same output shape as source. 
Written explicitly, it is 
```cpp
inline TShape SmoothL1Shape_(const TShape& src,
                             const EnvArguments& env) {
  return TShape(src);
```

### Define Functions
Create an unary or binary function with one output `mshadow::TBlob`.
```cpp
typedef void (*UnaryFunction)(const TBlob& src,
                              const EnvArguments& env,
                              TBlob* ret,
                              OpReqType req,
                              RunContext ctx);
typedef void (*BinaryFunction)(const TBlob& lhs,
                               const TBlob& rhs,
                               const EnvArguments& env,
                               TBlob* ret,
                               OpReqType req,
                               RunContext ctx);
```
* Functions are differentiated by the types of input arguments.
* `RunContext ctx` contains information needed in runtime for actual execution.

  ```cpp
  struct RunContext {
    void *stream;  // the stream of the device, can be NULL or Stream<gpu>* in GPU mode
    template<typename xpu> inline mshadow::Stream<xpu>* get_stream() // get mshadow stream from Context
  }  // namespace mxnet
  ```
  `mshadow::stream<xpu> *s = ctx.get_stream<xpu>();` is an example of obtaining a stream from `ctx`.
* `OpReqType req` denotes how computation results are written into `ret`.

  ```cpp
  enum OpReqType {
    kNullOp,  // no operation, do not write anything
    kWriteTo,  // write gradient to provided space
    kWriteInplace,  // perform an inplace write
    kAddTo  // add to the provided space
  };
  ```
  There is a macro defined in `operator_util.h` for a simplified use of `OpReqType`. 
  `ASSIGN_DISPATCH(out, req, exp)` will check `req` and perform an assignment.

In our smooth l1 loss example, we use `UnaryFunction` to define the function of this operator.
```cpp
template<typename xpu>
void SmoothL1Forward_(const TBlob& src,
                      const EnvArguments& env,
                      TBlob *ret,
                      OpReqType req,
                      RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  real_t sigma2 = env.scalar * env.scalar;
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    mshadow::Tensor<xpu, 2, DType> out = ret->get<xpu, 2, DType>(s);
    mshadow::Tensor<xpu, 2, DType> in = src.get<xpu, 2, DType>(s);
    ASSIGN_DISPATCH(out, req,
                    F<mshadow_op::smooth_l1_loss>(in, ScalarExp<DType>(sigma2)));
  });
}
```
After obtaining `mshadow::Stream` from `RunContext`, we get `mshadow::Tensor` from `mshadow::TBlob`. 
`mshadow::F` is a shortcut to initiate a `mshadow` expression. The macro `MSHADOW_TYPE_SWITCH(type, DType, ...)` 
handles details on different types and the macro `ASSIGN_DISPATCH(out, req, exp)` checks `OpReqType` and 
performs actions accordingly. `sigma2` is a special parameter in this loss, which we will cover in addtional usages. 

### Define Gradients (optional)
Create a gradient function with various types of inputs.
```cpp
// depending only on out_grad
typedef void (*UnaryGradFunctionT0)(const OutputGrad& out_grad,
                                    const EnvArguments& env,
                                    TBlob* in_grad,
                                    OpReqType req,
                                    RunContext ctx);
// depending only on out_value
typedef void (*UnaryGradFunctionT1)(const OutputGrad& out_grad,
                                    const OutputValue& out_value,
                                    const EnvArguments& env,
                                    TBlob* in_grad,
                                    OpReqType req,
                                    RunContext ctx);
// depending only on in_data
typedef void (*UnaryGradFunctionT2)(const OutputGrad& out_grad,
                                    const Input0& in_data0,
                                    const EnvArguments& env,
                                    TBlob* in_grad,
                                    OpReqType req,
                                    RunContext ctx);
```
Gradient functions of binary operator have similar structures except `Input`, `TBlob`, `OpReqType`
are doubled.
* `GradFunctionArgument`
  The `Input0`, `Input`, `OutputValue` and `OutputGrad` all share the structure of `GradFunctionArgument`, 
  which is defined as:
  ```cpp
  struct GradFunctionArgument {
      TBlob data;
  }
  ```

In our smooth l1 loss example, note that it is a `f'(x)`, which utilize input for gradient calculation, 
so the `UnaryGradFunctionT2` is suitable. To enable chain rule of gradient, we also need to multiply 
`out_grad` from top to the result of `in_grad`. 
```cpp
template<typename xpu>
void SmoothL1BackwardUseIn_(const OutputGrad& out_grad,
                            const Input0& in_data0,
                            const EnvArguments& env,
                            TBlob *in_grad,
                            OpReqType req,
                            RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  real_t sigma2 = env.scalar * env.scalar;
  MSHADOW_TYPE_SWITCH(in_grad->type_flag_, DType, {
    mshadow::Tensor<xpu, 2, DType> src = in_data0.data.get<xpu, 2, DType>(s);
    mshadow::Tensor<xpu, 2, DType> ograd = out_grad.data.get<xpu, 2, DType>(s);
    mshadow::Tensor<xpu, 2, DType> igrad = in_grad->get<xpu, 2, DType>(s);
    ASSIGN_DISPATCH(igrad, req,
                    ograd * F<mshadow_op::smooth_l1_gradient>(src, ScalarExp<DType>(sigma2)));
  });
}
```

### Register SimpleOp to MXNet
After creating shape, function and gradient, it is sufficient to restore them into both NDArray operator and 
Symbolic operator. There is a registration macro defined in `operator_util.h` to simplify this process.
```cpp
MXNET_REGISTER_SIMPLE_OP(Name, DEV)
.set_shape_function(Shape)
.set_function(DEV::kDevMask, Function<XPU>, SimpleOpInplaceOption)
.set_gradient(DEV::kDevMask, Gradient<XPU>, SimpleOpInplaceOption)
.describe("description");
```
`SimpleOpInplaceOption` is defined as:
```cpp
enum SimpleOpInplaceOption {
  kNoInplace,  // do not allow inplace in arguments
  kInplaceInOut,  // allow inplace in with out (unary)
  kInplaceOutIn,  // allow inplace out_grad with in_grad (unary)
  kInplaceLhsOut,  // allow inplace left operand with out (binary)
  kInplaceOutLhs  // allow inplace out_grad with lhs_grad (binary)
};
```

In our example, we have a gradient function that relies on input data, so the function can not be written in 
place. The output gradient is useless after gradient computation, so the gradient can be written inplace. 
```cpp
MXNET_REGISTER_SIMPLE_OP(smooth_l1, XPU)
.set_function(XPU::kDevMask, SmoothL1Forward_<XPU>, kNoInplace)
.set_gradient(XPU::kDevMask, SmoothL1BackwardUseIn_<XPU>, kInplaceOutIn)
.set_enable_scalar(true)
.describe("Calculate Smooth L1 Loss(lhs, scalar)");
```
Remember from shape functions that a default behavior without `set_shape_function` will be forcing the inputs 
(if binary) to be of the same shape and yield the same shape for output. The `set_enable_scalar` will be 
discussed in additional information.

### All in a List
* Create a shape function for determining the output shape
* Create a function as the forward routine by choosing a suitable function type
* Create a gradient as the backward routine by choosing a suitable gradient type
* Register the operator using registration process

## Additional Information on SimpleOp
### Usage on `EnvArguments`
Some operations may need a scalar as input, such as gradient scale, a set of keyword arguments 
controlling behavior or a temporary space to speed up calculations.
`EnvArguments` provide additional arguments and resources to make calculations more scalable 
and efficient.
```cpp
struct EnvArguments {
  real_t scalar;  // scalar argument, if enabled
  std::vector<std::pair<std::string, std::string> > kwargs;  // keyword arguments
  std::vector<Resource> resource;  // pointer to the resources requested
};
```

More registration parameters are required to enable these additional features. `scalar` and `kwargs` 
can not be present at the same time to prevent confusions on parameters. To enable `scalar`, use 
`set_enable_scalar(bool enable_scalar)` in registration. Then in forward function and gradients, 
this `scalar` can be accessed from `env.scalar` as in function parameter `EnvArguments env`.

To enable `kwargs`, use `set_enable_kwargs(bool enable_kwargs)` in registration. Then in forward 
functions and gradients, additional arguments are contained in `env.kwarg`, which is defined as 
`std::vector<std::pair<std::string, std::string> >`. The DMLC parameter structure can be used to 
simplify parsing keyword arguments. Refer to the [guide on parameter structure](https://github.com/dmlc/dmlc-core/blob/master/doc/parameter.md)
for more details.

Additional resources like `mshadow::Random<xpu>` and temporary memory space can also be requested and 
accessed from `EnvArguments.resource`. The registration routine is `set_resource_request(ResourceRequest req)` 
or `set_resource_request(const std::vector<ResourceRequest>)`, where `mxnet::ResourceRequest` is defined as in:
```cpp
struct ResourceRequest {
  enum Type {  // Resource type, indicating what the pointer type is
    kRandom,  // mshadow::Random<xpu> object
    kTempSpace  // A dynamic temp space that can be arbitrary size
  };
  Type type;  // type of resources
};
```
The registration will request the declared resource requests from `mxnet::ResourceManager` and place resources 
in `std::vector<Resource> resource` in `EnvArguments`. To access resources, write:
```cpp
auto tmp_space_res = env.resources[0].get_space(some_shape, some_stream);
auto rand_res = env.resources[0].get_random(some_stream);
```
Refer to `src/operator/loss_binary_op-inl.h` for a concrete example.

In our smooth l1 loss example, a scalar input is needed to mark the turning point of loss function. Therefore 
in the registration process, we use `set_enable_scalar(true)` and use `env.scalar` in function and gradient 
declarations. 

### Crafting a Tensor Operation
Since actual computation utilize `mshadow` library and sometimes we don't have functions readily available, it is 
possible to craft such tensor operations in operator implementations. If such functions are elementwise defined, we 
can implement them as a `mxnet::op::mshadow_op`. `src/operator/mshadow_op.h` contains a lot of `mshadow_op`, serving 
as a good example. `mshadow_op` are expression mappers and deal with the scalar case of desired functions. Refer to 
[mshadow expression API guide](https://github.com/dmlc/mshadow/tree/master/doc) for details.

It could also be possible that the operation cannot be done in an element wise way, like the softmax loss and gradient. 
Then there is a need to create a new tensor operation. Then we need to create a `mshadow` function and a `mshadow::cuda`
function directly. Please refer to `mshadow` library for details or `src/operator/roi_pooling.cc` for an example.

In our smooth l1 loss example, we create two mappers, namely the scalar cases of smooth l1 loss and gradient.
```cpp
namespace mshadow_op {
struct smooth_l1_loss {
  // a is x, b is sigma2
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    if (a > 1.0f / b) {
      return a - 0.5f / b;
    } else if (a < -1.0f / b) {
      return -a - 0.5f / b;
    } else {
      return 0.5f * a * a * b;
    }
  }
};
}
```
The gradient is similar, which can be found in `src/operator/smooth_l1_unary-inl.h`.

### Beyond Two Operands
This new unified API is designed to fulfil the fundamentals of an operation. For operators with more than two inputs, 
more than one outputs, or in need of more features, please refer to the original [Operator API](operator.md).

## KVStore: Multi-devices and multi-machines

### Introduction

MXNet uses a two-level *parameter server* for data synchronization.

<img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/multi-node/ps_arch.png width=400/>

- On the first layer, data are synchronized over multiple devices within a
  single worker machine. A device could be a GPU card, CPU, or other computational
  units. We often use sequential consistency model, also known as BSP, on this
  level.

- On the second layer, data are synchronize over multiple workers via server
  machines. We can either use a sequential consistency model for guaranteed
  convergence or an (partial)-asynchronous model for better system performance.

### KVStore

MXNet implemented the two-level parameter server in class *KVStore*. We
currently provide the following three types. Given the batch size *b*:

```eval_rst
============  ======== ======== ============== ============== =========
kvstore type  #devices #workers #ex per device #ex per update max delay
============  ======== ======== ============== ============== =========
`local`       *k*       1           *b / k*        *b*         *0*
`dist_sync`   *k*       *n*       *b / k*       *b × n*       *0*
`dist_async`  *k*       *n*       *b / k*        *b*           inf
============  ======== ======== ============== ============== =========
```

where the number of devices *k* used on a worker could vary for different
workers. And

- **number examples per update** : for each update, the number of examples used to
  calculate the averaged gradients. Often the larger, the slower the convergence.
- **number examples per device** : the number of examples batched to one device
  each time. Often the larger, the better the performance.
- **max delay** : The maximal delay of the weight a worker can get. Given a worker,
  a delay *d* for weight *w* means when this worker uses *w* (to calculate the
  gradient), *w* have been already updated by *d* times on some other places. A
  larger delay often improves the performance, but may slows down the
  convergence.

### Multiple devices on a single machine

KV store `local` synchronizes data over multiple devices on a single machine.
It gives the same results (e.g. model accuracy) as the single device case. But
comparing to the latter, assume there are *k* devices, then each device only
processes *1 / k* examples each time (also consumes *1 / k* device memory). We
often increase the batch size *b* for better system performance.

When using `local`, the system will automatically chooses one of the following
three types. Their differences are on where to average
the gradients over all devices, and where to update the weight.

```eval_rst
=======================  ================   ==============
 kvstore type            average gradient   perform update
=======================  ================   ==============
`local_update_cpu`       CPU                 CPU
`local_allreduce_cpu`    CPU                 all devices
`local_allreduce_device` a device            all devices
=======================  ================   ==============
```

They produce (almost) the same results, but may vary on speed.

- `local_update_cpu`, gradients are first copied to main memory, next averaged on CPU,
  and then update the weight on CPU. It is suitable when the average size of
  weights are not large and there are a large number of weight. For example the
  GOOGLE Inception network.

- `local_allreduce_cpu` is similar to `local_update_cpu` except that the
  averaged gradients are copied back to the devices, and then weights are
  updated on devices. It is faster than 1 when the weight size is large so we
  can use the device to accelerate the computation (but we increase the workload
  by *k* times). Examples are AlexNet on Imagenet.

- `local_allreduce_device` is similar to `local_allreduce_cpu` except that the
  gradient are averaged on a chosen device. It may take advantage of the
  possible device-to-device communication, and may accelerate the averaging
  step. It is faster than 2 when the gradients are huge. But it requires more
  device memory.

### Multiple machines

Both `dist_async` and `dist_sync` can handle the multiple machines
situation. But they are different on both semantic and performance.

- `dist_sync`: the gradients are first averaged on the servers, and then send to
  back to workers for updating the weight. It is similar to `local` and
  `update_on_kvstore=false` if we treat a machine as a device.  It guarantees
  almost identical convergence with the single machine single device situation
  if reduces the batch size to *b / n*. However, it requires synchronization
  between all workers, and therefore may harm the system performance.

- `dist_async`: the gradient is sent to the servers, and the weight is updated
  there. The weights a worker has may be stale. This loose data consistency
  model reduces the machine synchronization cost and therefore could improve the
  system performance. But it may harm the convergence speed.

# Recommended Next Steps

* [Analogy to other DL systems](http://mxnet.io/architecture/analogy.html)
* [How to read MXNet code](http://mxnet.io/architecture/read_code.html)
* [Develop and hack MXNet](http://mxnet.io/how_to/develop_and_hack.html)