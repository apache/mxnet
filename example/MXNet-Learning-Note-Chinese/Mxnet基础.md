#NDArray
MXNet的基本计算单元，用于矩阵和张量计算
-	多设备，可用于CPU和GPU
-	自动并行运算
-	提供了元素级的计算（类似于Matlab中矩阵点乘），在不同设备间计算需要先复制到同一设备中
-	可以直接保存为二进制文件到硬盘

尽量推迟导出计算结果，可以加速计算
构建符号图
![例子](.\pic\compose_multi_in.png "构建符号图")

利用符号图进行计算
![compute](.\pic\executor_forward.png)
一个构建神经网络的例子
```python
import mxnet as mx
net = mx.symbol.Variable('data')  #Variable通常定义输入
net = mx.symbol.FullyConnected(data=net, name='fc1', num_hidden=128)
net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
net = mx.symbol.FullyConnected(data=net, name='fc2', num_hidden=64)
net = mx.symbol.SoftmaxOutput(data=net, name='out')
arg_shape, out_shape, aux_shape = net.infer_shape(data=(100, 100)) #定义输入data为（100,100）的数据结构，由此推断出网络的每层结构类型
type(net)
```


###Theano定义网络结构
```python
w = init_weights((6, 1, 5, 5))   #第一层卷积层网络结构，卷积核大小5*5，输入1个特征图，输出6个
w2 =init_weights((12, 6, 6, 6))
w_o =init_weights((432, 2))     #softmax回归，输入432个，输出结果为2个
w_b=init_weights((2,))           #softmax 参数，大小为输出结果的个数
def model(X, w1, w2,w_o,w_b, p_drop_conv, p_drop_hidden):  #需要首先定义x，w1,w2,w_b,w_o的结构
    l1a = rectify(conv2d(X, w1, border_mode='full'))         #卷积层
    l1 = max_pool_2d(l1a, (4, 4))          #池化层，大小4*4
    l1 = dropout(l1, p_drop_conv)  #自己定义dropout函数

    l2a = rectify(conv2d(l1, w2))
    l2b = max_pool_2d(l2a, (2, 2))
    l2 = T.flatten(l2b, outdim=2)
    l2 = dropout(l2, p_drop_conv)

    pyx = softmax(T.dot(l2, w_o)+w_b)
    return l1, l2, pyx
```
##Caffe定义网络结构
使用配置文件的形式，一般有两个文件*solve.prototxt*和*net.prototxt*
Caffe主程序，记为Caffe.exe。在训练网络过程中，使用

	caffe.exe" train --solver=examples/mnist/solve.prototxt
最终训练结果会保存为.caffemodel文件，在需要时可以调用这个文件读取网络参数。
在*solve.prototxt*中定义训练使用的设备（GPU,CPU)，训练次数，batch大小，学习速率，net.prototxt的位置等

	net: "examples/mnist/net.prototxt"
*net.prototxt*
```
name: "LeNet"
layers {
  name: "m1"
  type: DATA
  top: "data"
  top: "label"
  data_param {
    source: "examples/mnist/mnist_train_leveldb"
    backend: LEVELDB   //Caffe训练集只接受LEVELDB和LMDB数据库格式，需要事先用工具转换图片
    batch_size: 64
  }
  transform_param { scale: 0.00390625 }
  include: { phase: TRAIN }
}

layers {
  name: "conv1"
  type: CONVOLUTION
  bottom: "data"
  top: "conv1"
  blobs_lr: 1
  blobs_lr: 2
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler { type: "constant" }
  }
}

layers {
  name: "pool1"
  type: POOLING
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}

layers {
  name: "relu1"
  type: RELU
  bottom: "ip1"
  top: "ip1"
}

layers {
  name: "loss"
  type: SOFTMAX_LOSS
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
```
根据Caffe训练的特点可以看出来，Caffe只能通过已经提供的网络模块搭建网络，增加自己的模块需要修改Caffe源文件并重新编译Caffe.exe。Caffe比较重视丰富网络单元的类型。

###定义自己的网络结构
```python
class NumpySoftmax(mx.operator.NumpyOp):
    def __init__(self):
        super(NumpySoftmax, self).__init__(False)
    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape]
```
*Infer_shape should always return two lists, even if one of them is empty.*
```python
    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y[:] = np.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        y /= y.sum(axis=1).reshape((x.shape[0], 1))

    def backward(self, out_grad, in_data, out_data, in_grad):
        l = in_data[1]
        l = l.reshape((l.size,)).astype(np.int)
        y = out_data[0]
        dx = in_grad[0]
        dx[:] = y
        dx[np.arange(l.shape[0]), l] -= 1.0
```
*Remember when you assigning to a tensor, use x[:] = ... so that you write to the original array instead of creating a new one.*




