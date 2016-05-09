# Unifying NDArray Operator and Symbolic Operator : How does it work
NDArray operations are similar to symbolic operations except the fact that sometimes we 
cannot write in place to the operands without a complete dependency graph. However, the 
logics underlying NDArray and Symbolic operation are almost the same. Unifying different 
invoking process and returning to the fundamental elements of operators are the purpose of 
this new unified operator API. Because most mathematical operators attend to one or two 
operands and more operands make dependency-related optimization useful, the unified operator 
are specially designed for unary and binary operations.

Consider elements of an operation. Ideally, functions and derivatives are all we need to 
describe an operation. Let us restrict that to the space of unary and binary operations. How 
do we classify all operations to maximize the possibility of inplace write optimization? Note 
that functions can be separate out by the number of operands. Derivatives are a bit more 
complex. Whether output value, input data or neither are needed alongside head gradient is 
crucial to construct a dependency graph. Gradient functions in the unified API is thus 
differentiated through the types of operands it takes for calculation.

Before we continue on the operator interface, it is recommend to take a look at the [mshadow
library guide](https://github.com/dmlc/mshadow/tree/master/guide) since actual calculations 
will be done in `mshadow::TBlob` structure.

# Unified Operator API
## Define Shapes
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
Please refer to additional usages.

## Define Functions
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

## Define Gradients (optional)
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

## Register Operator
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

## All in a list
* Create a shape function for determining the output shape
* Create a function as the forward routine by choosing a suitable function type
* Create a gradient as the backward routine by choosing a suitable gradient type
* Register the operator using registration process

# Additional Information on the Unified Operator API
## Usage on `EnvArguments`
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
