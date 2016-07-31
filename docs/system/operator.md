Operators in MXNet
==================
An operator in MXNet is a class that contains both actual computation logic and auxiliary informations that could aid our system to perform optimizations like in-place updates and auto-derivative. Before continue on this document, it is strongly recommended for you to first understand `mshadow` library, since all operators compute on tensor-like structure `mshadow::TBlob` provided by the system during runtime. MXNet's operator interface tries its best to offer users flexibility including:
* Save memory allocation cost by specifying in-place updates.
* Hide some internal arguments from python side to make it cleaner.
* Define the relationships among input tensors and output tensors which allows system to perform shape check for you.
* Acquire additional temporary spaces from system to perform computation (e.g. calling `cudnn` routines).

Operator Interface
------------------
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

Operator Property
-----------------
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

* **Inplace Option:** To further save memory allocation cost, inplace update are welcomed. This usually happens for element-wise operations when input tensor and output tensor are of the same shape. This could be specified by the following interface:
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

Create Operator from Operator Property
--------------------------------------
As mentioned above `OperatorProperty` includes all *semantical* attributes of an operation. It is also in charge of creating `Operator` pointer for actual computation.

### Create Operator
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

### Parameterize Operator
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

### Register Operator to MXNet
Use following macros to register the parameter structure and the operator property class to MXNet system:
```c++
DMLC_REGISTER_PARAMETER(ConvolutionParam);
MXNET_REGISTER_OP_PROPERTY(Convolution, ConvolutionOpProperty);
```
The first argument to the macro is the name string, the second is the property class name.

All in a list
-------------
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

Enjoy your MXNet trip.
