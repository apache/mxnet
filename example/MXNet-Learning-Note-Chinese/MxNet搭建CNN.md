####MxNet搭建CNN的主要步骤
1.	搭建网络结构
2.	确定网络结构中的变量，以及变量的处理方式
3.	通过mxnet::Executor::Bind函数，将网络结构和变量参数组合成一个可进行的计算的图
4.	进行前向计算，反向求导计算等

####搭建网络结构

```cpp
Symbol data = Symbol::CreateVariable("data");
		Symbol conv1 = OperatorSymbol("Convolution", data, "conv1","kernel", mshadow::Shape2(5, 5),
			"num_filter", 20);
		Symbol tanh1 = OperatorSymbol("Activation", conv1, "tanh1","act_type", "tanh");
		Symbol pool1 = OperatorSymbol("Pooling", tanh1, "pool1",
			"pool_type", "max",
			"kernel", mshadow::Shape2(2, 2),
			"stride", mshadow::Shape2(2, 2));

		Symbol fc1 = OperatorSymbol("FullyConnected",pool1, "fc2",
			"num_hidden", 10);
		Symbol lenet = OperatorSymbol("SoftmaxOutput", fc1, "softmax");
//本例中OperatorSymbol并非MxNet自带的函数，需要自己进行实现，可以在cpp_net.hpp中找到。
```
####确定网络结构中的变量
1.	定义前向通道使用的存储变量（输入值输出值和每层网络的参数等），统一压入`vector<mxnet::NDArray> in_args`
2.	定义反向计算通道的存储变量，统一压入`vector<mxnet::NDArray> arg_grad_store`
3.	每个变量的操作类型（write，add，noop），统一压入`vector<mxnet::OpReqType> grad_req_type`;
4.	定义aux_states（暂时不明白是什么作用，used as internal state in op），统一压入`vector<mxnet::NDArray>aux_states`

上述类型属于NDArray的可以只定义维数，不初始化值，维数类型可以根据InferShape（MXNet内部函数）得出，具体用法见cpp_net.hpp中的InitArgArrays函数

####Bind函数分析
`exe = mxnet::Executor::Bind(net, ctx_dev, g2c, in_args, arg_grad_store,
						grad_req_type, aux_states);`
![ind](.\pic\Bind.png)
-	symbol 网络的名称
-	default_ctx 网络默认的训练设备(CPU,GPU)
-	group2ctx  没有设置过，Context mapping group to context
-	in_args 输入值
-	arg_grad_store 梯度计算值
-	grad_req_type，每个梯度的保存方式,{kNullOp, kAddTo, kWriteTo}
-	aux_states ,  NDArray that is used as internal state in op

####执行训练过程
	exe->Forward(true);
	exe->Backward(std::vector<mxnet::NDArray>());
    optimizer->Update(i, &in_args[i], &arg_grad_store[i], learning_rate);