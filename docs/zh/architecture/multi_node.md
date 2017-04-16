# Multi-devices and multi-machines

## Introduction

MXNet 使用了一个两层的  *parameter server* 来做数据同步.

<img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/multi-node/ps_arch.png width=400/>

- 同一台 worker 机器中跨设备的数据同步,通过第一层 ps. 一个设备可以是 GPU 卡, CPU 或者其他的计算单元. 我们使用 sequential consistency model 作为这层的数据同步模型, 比较出名的如 BSP.
- 第二层实现的是多个worker 机器之间的数据同步, 我们使用了两种通信模型, 一种是 sequential consistency model 来确保模型最后的收敛, 还有一种 (partial)-asynchronous model 来获得更好的系统性能.

## KVStore

MXNet 在 *KVStore* 类中实现的这个两层的 parameter server 架构. 对于给定的 batch 大小 *b*,  现在有三种同步策略:
```eval_rst
============  ======== ======== ============== ============== =========
kvstore type  #devices #workers #ex per device #ex per update max delay
============  ======== ======== ============== ============== =========
`local`       *k*       1         *b / k*       *b*           *0*
`dist_sync`   *k*       *n*       *b / k*       *b × n*       *0*
`dist_async`  *k*       *n*       *b / k*       *b*           inf
============  ======== ======== ============== ============== =========
```

其中表示worker 机器上的设备数量的 *k* 可以是不同的.
而且

- **number examples per update** :  每次更新计算平均梯度(averaged gradients) 使用的的样本的数量. 通常样本数量越大, 模型的收敛越慢.
- **number examples per device** : 一个设备一次可以批量处理的样本数目. 通常样本数目越大, 系统的性能越好.
- **max delay** : worker 机器 中 weight 的最大延迟.  对于一个给定的 worker 机器, 权重 *w* 的最大延迟 *d*  表示当 worker 使用 *w* (计算梯度) 的时候, *w* 已经在其他的地方更新的次数是 *d*.  更大的延迟一般会提高系统的性能 (吞吐量) ,但是可能会降低系统的收敛速度. 


## Multiple devices on a single machine

在单机的情况下, 多个设备之间的数据同步采用的是 `local` 同步模型. 这种同步模型在多个设备的时候可以得到和单个设备一样的结果 (比如说模型精度). 但是多设备和单个设备相比, 假设这里有 *k* 个设备. 那么每个设备每次只处理 *1 / k* 个样本 (当然也只用了*1 / k* 的设备内存).  我们经常通过增大 batch size *b* 来获得更好的系统性能.

当使用 `local` 的时候, 系统会自动地从下面列出来的三种同步类型中选择一个合适的. 它们之间的不同在于上计算梯度平均以及 weight 的更新操作在哪些设备上进行. 

```eval_rst
=======================  ================   ==============
 kvstore type            average gradient   perform update
=======================  ================   ==============
`local_update_cpu`       CPU                 CPU
`local_allreduce_cpu`    CPU                 all devices
`local_allreduce_device` a device            all devices
=======================  ================   ==============
```

他们(几乎) 产生同样的结果, 但是在速度上有很大的不同.

- `local_update_cpu`, 梯度首先被复制到主存, 然后使用 CPU 来计算平均, 接着使用 CPU 来做 weight 更新. 这种模式适合 weight 不是特别大, 但是数量很多的情况, 比如 谷歌的 Inception 模型.
- `local_allreduce_cpu` 很类似 `local_update_cpu` 除了梯度的平均是要拷贝回设备内存然后在设备上进行计算的. 当 weight size 很大的情况下, 这种模式要比 `local_update_cpu` 快. 原因是我们可以通过设备来加速计算 (但是我们把设备负载增加了 *k*倍) . 举个例子, 比如说用来做 imagenet 1k 分类的 AlexNet 模型.
- `local_allreduce_device` 类似于 `local_allreduce_cpu` 除了梯度平均计算是在一个选定的设备上进行的. 这种模式可以利用设备到设备之间的点对点传输的特性, 可能加速平均计算的速度. 如果梯度很大的话, 它会比 `local_allreduce_cpu`  模式快, 但是利用更多的设备内存.

## Multiple machines

`dist_async` 和 `dist_sync` 模式都可以处理多机同步的情况. 但是它们在语义和性能上有很多不一样的地方.
- `dist_sync`: 梯度首先在 server 上进行平均计算, 然后发回worker 进行 weight 更新.  它类似于 `local`, 如果我们把一台机器看做是一个设备的话, 我们可以设置 `update_on_kvstore=false`. 如果把 batch size 降低到 *b / n* 它保证可以获得和单机单卡的情况一样的收敛性. 但是它需要worker之间的同步, 可能会因此而降低系统的性能.
- `dist_async`: 梯度被发送到 server 上, 然后在 server 上进行 weight 的更新. 一个 worker 上的 weight 可能因此变得过时 (stale). 这个对数据一致性不太严格的模式可以降低机器之间的数据同步开销, 因此可以提升系统的性能. 但是可能会对收敛速度有影响.
