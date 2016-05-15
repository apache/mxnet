# Unifying NDArray Operator and Symbolic Operator : How does it work
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
where `.*` stands for elementwise multiplication and `f`, `f'` is the smooth l1 loss function, 
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
discussed in addtional information.

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

Addtional resources like `mshadow::Random<xpu>` and temporary memory space can also be requested and 
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

It could also be possible that the operation cannot be done in an elementwise way, like the softmax loss and gradient. 
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
This new unified API is designed to fulfill the fundamentals of an operation. For operators with more than two inputs, 
more than one outputs, or in need of more features, please refer to the original [Operator API](operator.md).
