Operators in MXNet
==================
MXNet 中的 operator 是一个既包含了实际的计算也包含了一些附加信息的 class, 这些附加信息可以帮助我们的系统来实现原地更新和自动微分等优化.  在在继续这篇文档之前, 我们强烈的建议您首先要搞明白 `mshadow` 库, 因为所有 operator 的计算都是基于系统提供的数据结构 `mshadow::TBob`, 而该数据结构类似于张量(Tensor). MXNet 的 operator 的接口致力于灵活性, 现在提供的灵活性包括以下几点:

* 通过特定的原地更新数据来节省内存.
* 在 python 端掩盖一些内部的参数, 是代码整洁.
* 定义输入 tensor 和输出 tensor 的关系,这样系统可以帮助我们进行类型检查.
* 从系统中申请额外的空间来进行计算 (e.g. calling `cudnn` routines).

Operator Interface
------------------
operator 核心的接口是 `Forward`:
```c++
virtual void Forward(const OpContext &ctx,
                     const std::vector<TBlob> &in_data,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &out_data,
                     const std::vector<TBlob> &aux_states) = 0;
```
* `OpContext` 的数据结构是下面的代码:
  ```c++
  struct OpContext {
    int is_train;
    RunContext run_ctx;
    std::vector<Resource> requested;
  }
  ```
  
  ,  你可以知道 operator 是在进行 train 还是 test (`is_train`); operator 应该在哪个device ( `run_ctx` ) 上运行以及通过下面的参数来确定是否要申请额外的资源.
* `in_data` 和 `out_data` 分别代表输入 tensor 和输出 tensor. 所有的 tensor 需要的空间都是系统进行申请和管理.
* `req` 表示计算的结构是如何写入到 `out_data` 中的. 换句话说, `req.size() == out_data.size()` 和 `req[i]` 与如何写入 `out_data[i]` 是相关的. `OpReqType` 是下面这样定义的:

  ```c++
  enum OpReqType {
    kNullOp,
    kWriteTo,
    kWriteInplace,
    kAddTo
  };
  ```
 , 一般情况下, 所有的 `out_data` 的类型应该是 `kWriteTo`, 表示`out_data` 代表的 tensor 提供的是可以直接写入的 *原始的* 内存块 . 在有些情况下, 比如说在计算 表示 `gradient` 的 tensor 的时候, 我们最好是将梯度累加起来, 而不是直接覆盖掉原来的结果, 这样我们就不需要每次计算的时候申请额外需要的内存空间. 在这种情况下, `req` 的类型应该是 `kAddTo`, 表示应该调用 `+=` 操作.
* `aux_states` 表示的是为了方便计算而需要的附加的 tensor,  现在是没有用到的.

除了 `Forward` operator, 用户有时候也需要实现 `Backward` 接口, 定义如下:
```c++
virtual void Backward(const OpContext &ctx,
                      const std::vector<TBlob> &out_grad,
                      const std::vector<TBlob> &in_data,
                      const std::vector<TBlob> &out_data,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &in_grad,
                      const std::vector<TBlob> &aux_states);
```
Backward 的接口遵循 `Forward` 一样的设计原则, 除了 `out_grad`, `in_data` 和 `out_data` 是 operator 计算  `in_grad` 必须的以外, 其他的输入和 `Forward` 是一样的. 命名策略和 torch 的约定很类似, 可以用下面的图示来总结:

[input/output semantics figure]

有些 operator 并不是都需要的`out_grad`,`in_data`和`out_data` 参数, 这个需求可以通过`OperatorProperty`中的接口`DeclareBackwardDependency`来实现.

Operator Property
-----------------

有这么一种可能, convolution 有好几种不同的实现, 用户可能想要在这些算法中选择能够获得最高性能的算法. 为了实现这个目的, 我们将 operator 的 *sematic* 接口从具体的实现 (`Operator` 类) 中分离出来, 独立为`OperatorProperty` 类.  `OperatorProperty`的接口包括以下内容:

* **InferShape:**
  ```c++
  virtual bool InferShape(std::vector<TShape> *in_shape,
                          std::vector<TShape> *out_shape,
                          std::vector<TShape> *aux_shape) const = 0;
  ```
这个接口有两个目的: (1) 向系统提供每个输入和输出 Tensor 的大小, 这样系统可以在进行 `Forward` 和 `Backward` 之前提前申请好相应的内存; (2) 进行类型检查, 在运行前确保没有明显的错误. `in_shape` 中的 shape 是有系统自动设置 (依据是依赖的上个 Operator 的 `out_shape` ). 这个接口会返回`false`如果系统认为提供的信息不足以完成shape的推断, 或者在shape 不一致的时候抛出异常.

* **Request Resources:**  有些操作需要额外的内存作为工作空间来进行计算, 比如说`cudnnConvolutionForward`. 这种情况下, 系统最好可以对这部分内存进行管理, 这样系统可以做一些优化, 比如说内存的重复利用. MXNet 定义了两个接口来达到目的:

  ```c++
  virtual std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const;
  virtual std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const;
  ```
  `ResourceRequest` 数据结构 (在 `resource.h` 中) 现在只包含了一个 type flag:
  ```c++
  struct ResourceRequest {
    enum Type {
      kRandom,  // get an mshadow::Random<xpu> object
      kTempSpace,  // request temporay space
    };
    Type type;
  };
  ```
 如果 `ForwardResource` 和 `BackwardResource` 返回的数组是非空的, 那么系统会通过`Operator` 的`Foward` 和`Backward` 接口中的 `ctx` 参数来提供相应的资源. 简单的举个例子, 如果要获取这些资源, 可以按照下面的写法来做:
  ```c++
  auto tmp_space_res = ctx.requested[kTempSpace].get_space(some_shape, some_stream);
  auto rand_res = ctx.requested[kRandom].get_random(some_stream);
  ``` 
  具体的例子可以参考 `src/operator/cudnn_convolution-inl.h`.

* **Backward dependency:** 让我们来看两个不同的 operator signature ( 为了演示的目的，我们命名了所有的变量):

  ```c++
  void FullyConnectedForward(TBlob weight, TBlob in_data, TBlob out_data);
  void FullyConnectedBackward(TBlob weight, TBlob in_data, TBlob out_grad, TBlob in_grad);

  void PoolingForward(TBlob in_data, TBlob out_data);
  void PoolingBackward(TBlob in_data, TBlob out_data, TBlob out_grad, TBlob in_grad);
  ```
 我们注意到, 在 `FullyConnectedForward` 中使用的 `out_data`变量, 在 `FullyConnectedBackward` 没有被用到. 与此同时, `PoolingBackward` 使用了 `PoolingForward` 中所有的变量. 因此,  这个`out_data` tensor 在做backward 不需要的时候, 需要释放. 这里有个挑战就是如何进行垃圾回收 (GC) 尽可能的快. 对于这种情况, 我们提供了一个接口:

  ```c++
  virtual std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const;
  ```

这里的 vector 中的 `int` 元素一个区分不同的 arrays 的 id. 让我们来看看这个接口如何定义`FullyConnected`和`Poolling`的不同的依赖关系的:


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



* **Inplace Option:** 为了进一步的节省内存的申请开销, 我们倾向于是用原地更新(inplace update). 这个主要用在 element-wise 操作上, 因为这种情况下输入 tensor 和输出 tensor 的 shape 是一致的. 针对这种情况下, 我们提供了下面的接口:

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

这个接口告诉系统 `in_data[0]` 和 `out_data[0]` tensors 应该在`Forward` 的计算过程中使用同样的内存空间, 同样地,  `out_grad[0]` 和 `in_grad[0]` 分享同样的内存空间在 `Backward`计算过程中.

  
  >**ATTENTION:** Even with the above specification, it is *not* guaranteed that input and output tensors will share the same space. In fact, this is only a hint for the system for the final decision. However, in either case, such decision is completely transparent to user, so the actual `Forward` and `Backward` implementation does not need to consider that.

 

* **Expose Operator to Python:** 由于 c++ 编程语言的限制, 我们需要用户实现下面的接口:

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
我们在上面的内容中提到过 `OperatorProperty` 包括所有的一个操作的 *semantical*attributes. 它也包括需要创建一个`Operator` 指针指向真正的计算操作.

### Create Operator
实现 `OperatorProperty`中下面的接口:
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

当我们实现一个卷积operator 的时候, 我们需要知道kernal 大小, stride 大小,  padding 大小等等信息. 这些需要作为参数传递给 operator, `Forward` 和`Backward`计算过程需要这些参数. 为了传递参数, 用户需要定义 `ConvolutionParam` 数据结构:

```c++
#include <dmlc/parameter.h>
struct ConvolutionParam : public dmlc::Parameter<ConvolutionParam> {
  TShape kernel, stride, pad;
  uint32_t num_filter, num_group, workspace;
  bool no_bias;
};
```
把它放到`ConvolutionOpProperty`里, 然后在类初始化的时候,  将这个数据结构传递给operator 类中.

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
使用下面的宏定义来将 parameter 结构和 OperatorProperty 类注册到 MXNet 的系统中:

```c++
DMLC_REGISTER_PARAMETER(ConvolutionParam);
MXNET_REGISTER_OP_PROPERTY(Convolution, ConvolutionOpProperty);
```
这个宏定义的第一个参数是 name, 第二个参数是 Property 类的名字.


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


祝您 MXNet 之旅愉快.
