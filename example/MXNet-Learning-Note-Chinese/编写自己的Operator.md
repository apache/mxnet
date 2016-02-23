本文件根据http://mxnet.readthedocs.org/en/latest/developer-guide/operator.html 完成
可以和[卷积操作函数分析文档](./convolution-inl.h分析.md)一起看
#####操作接口
有Forward和Backward
```cpp
virtual void Forward(const OpContext &ctx,
                     const std::vector<TBlob> &in_data,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &out_data,
                     const std::vector<TBlob> &aux_states) = 0;
```
```cpp
struct OpContext {
  int is_train;
  RunContext run_ctx;
  std::vector<Resource> requested;
}
```
```cpp
virtual void Backward(const OpContext &ctx,
                      const std::vector<TBlob> &out_grad,
                      const std::vector<TBlob> &in_data,
                      const std::vector<TBlob> &out_data,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &in_grad,
                      const std::vector<TBlob> &aux_states);
```
#####操作资源
-	Infershape
```cpp
virtual bool InferShape(std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        std::vector<TShape> *aux_shape) const = 0;
```
返回false当没有足够的输入来推断大小，返回error当数据大小不一致
-	所需资源
求解执行操作所需的前向和后向资源
```cpp
	virtual std::vector<ResourceRequest> ForwardResource(
    const std::vector<TShape> &in_shape) const;
virtual std::vector<ResourceRequest> BackwardResource(
    const std::vector<TShape> &in_shape) const;
```
其中ResourceRequest是一个表示所需资源的结构体

需要申请资源时，只需执行
```cpp
auto tmp_space_res = ctx.requested[kTempSpace].get_space(some_shape, some_stream);
auto rand_res = ctx.requested[kRandom].get_random(some_stream);
```
-	反向依赖
```cpp
virtual std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const;
```
利用此函数来声明反向传播所需要的参数，可以方便系统将不需要的内存释放掉
-	原址操作
当输入和输出大小相同时，通过声明表示输出可以覆盖输入的位置

#####生成操作
```cpp
\\OperatorProperty 中
virtual Operator* CreateOperator(Context ctx) const = 0;
```
例子
```cpp
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
#####操作参数
首先定义一个ConvolutionParam结构体
```cpp
#include <dmlc/parameter.h>
struct ConvolutionParam : public dmlc::Parameter<ConvolutionParam> {
  TShape kernel, stride, pad;
  uint32_t num_filter, num_group, workspace;
  bool no_bias;
};
```
将上述结构体放入ConvolutionOpProperty中并传递给operator类
```cpp
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
#####在MXNet中注册操作
```cpp
DMLC_REGISTER_PARAMETER(ConvolutionParam);
MXNET_REGISTER_OP_PROPERTY(Convolution, ConvolutionOpProperty);
```
其中第一个参数是名称，第二个是执行类
#####汇总
-	使用Operator 中的Forward和Backward写自己需要的操作
-	使用OperatorProperty的接口：
      -	将参数传递给操作类(可以使用Init接口)
      -	使用CreateOperator接口创造操作
      -	正确实现操作接口描述，例如参数名称
      -	正确实现InferShape设置输出张量大小
      -	[可选]如果需要其他资源，检查ForwardResource和BackwardResource
      -	[可选]如果Backward不需要Forward的所有输入和输出，检查DeclareBackwardDependency
      -	[可选]如果支持原址操作，检查ForwardInplaceOption和BackwardInplaceOption
-	在OperatorProperty中注册