# Measure communication bandwidth

MXNet provides multiple ways to communicate data. The best choice depends on
both the physical machines and neural network strcture. This folder provides
tools to test the bandwidth under various setups, which can be used to debugging
the performance.

## Usages

`measure.py` provides several options. We list some important ones, try `python
measure.py --help` for more details.

- `--gpus` the list of gpus to test. `0,3` means GPUs 0 and 3.
- `--network` the neural network to test, such as resnet, alexnet, inception-bn, and vgg
- `--kvstore` the way how data is communicated.
  - `local` : copy data from GPU to CPU, run optimizer on CPU
  - `device` (default) : communicate (reduce and broadcast) data on GPU,
     use GPU peer-to-peer communication if supported. The optimizer will run on
     GPUs.
  - `dist_sync` : similar to `local`, but the data is further send to parameter
    servers, and run the optimizer on servers
  - `dist_sync_device` : similar to `dist_sync` but try best to use GPU for communcation
  - `dist_async` : similar to `dist_sync` but uses asynchoronous communication
  - `dist_async_device` : similar to `dist_async` but try best to use GPU for communcation

## Samples

### Single machine with multiple GPUs

- Use resnet 200 layers on GPU 0, 1, 2, and 3

```bash
~/mxnet/tools/bandwidth $ python measure.py --kv-store device --gpus 0,1 --network resnet --depth 200
INFO:root:Namespace(batch_size=128, data_shape='128,3,224,224', depth=200, disp_batches=1, gpus='0,1', kv_store='device', network='resnet', num_batches=5, num_classes=1000, optimizer='None', test_results=1)
INFO:root:num of arrays = 205, total size = 257.991328 MB
INFO:root:iter 1, 0.023242 sec, 11.100222 GB/sec per gpu, error 0.000000
INFO:root:iter 2, 0.023106 sec, 11.165508 GB/sec per gpu, error 0.000000
INFO:root:iter 3, 0.023218 sec, 11.111735 GB/sec per gpu, error 0.000000
INFO:root:iter 4, 0.023193 sec, 11.123614 GB/sec per gpu, error 0.000000
INFO:root:iter 5, 0.023089 sec, 11.173694 GB/sec per gpu, error 0.000000
```

The results are close to the unidirectional bandwidth, which is 13 GB/sec, reported by
`cuda/samples/1_Utilities/p2pBandwidthLatencyTest`. But our problem is harder
because we do all-to-all communication.

- Use 8 GPUs, it satruates the single 16x link between GPU 0,1,2,3 and GPU 4,5,6,7.

```bash
~/mxnet/tools/bandwidth $ python measure.py --kv-store device --gpus 0,1,2,3,4,5,6,7 --network resnet --depth 200
INFO:root:Namespace(batch_size=128, data_shape='128,3,224,224', depth=200, disp_batches=1, gpus='0,1,2,3,4,5,6,7', kv_store='device', network='resnet', num_batches=5, num_classes=1000, optimizer='None', test_results=1)
INFO:root:num of arrays = 205, total size = 257.991328 MB
INFO:root:iter 1, 0.102321 sec, 4.412429 GB/sec per gpu, error 0.000000
INFO:root:iter 2, 0.100345 sec, 4.499330 GB/sec per gpu, error 0.000000
INFO:root:iter 3, 0.097317 sec, 4.639322 GB/sec per gpu, error 0.000000
INFO:root:iter 4, 0.099873 sec, 4.520586 GB/sec per gpu, error 0.000000
INFO:root:iter 5, 0.100774 sec, 4.480169 GB/sec per gpu, error 0.000000
```

- Now let's only use GPU-CPU communication, it satruates the single 16x link
between all GPUs and the CPU.

```bash
~/mxnet/tools/bandwidth $ python measure.py --kv-store local --gpus 0,1,2,3,4,5,6,7 --network resnet --depth 200
INFO:root:Namespace(batch_size=128, data_shape='128,3,224,224', depth=200, disp_batches=1, gpus='0,1,2,3,4,5,6,7', kv_store='local', network='resnet', num_batches=5, num_classes=1000, optimizer='None', test_results=1)
INFO:root:num of arrays = 205, total size = 257.991328 MB
INFO:root:iter 1, 0.290164 sec, 1.555964 GB/sec per gpu, error 0.000000
INFO:root:iter 2, 0.293963 sec, 1.535856 GB/sec per gpu, error 0.000000
INFO:root:iter 3, 0.294468 sec, 1.533222 GB/sec per gpu, error 0.000000
INFO:root:iter 4, 0.290657 sec, 1.553325 GB/sec per gpu, error 0.000000
INFO:root:iter 5, 0.290799 sec, 1.552567 GB/sec per gpu, error 0.000000
```

- Finally we change to VGG and also run the `sgd` optimizor

```bash
~/mxnet/tools/bandwidth $ python measure.py --kv-store device --gpus 0,1,2,3,4,5,6,7 --network vgg --optimizer sgd
INFO:root:Namespace(batch_size=128, data_shape='128,3,224,224', depth=152, disp_batches=1, gpus='0,1,2,3,4,5,6,7', kv_store='device', network='vgg', num_batches=5, num_classes=1000, optimizer='sgd', test_results=1)
INFO:root:num of arrays = 22, total size = 531.453344 MB
INFO:root:iter 1, 0.525208 sec, 1.770810 GB/sec per gpu, error 0.000000
INFO:root:iter 2, 0.524052 sec, 1.774715 GB/sec per gpu, error 0.000000
INFO:root:iter 3, 0.524732 sec, 1.772416 GB/sec per gpu, error 0.000000
INFO:root:iter 4, 0.527117 sec, 1.764396 GB/sec per gpu, error 0.000000
INFO:root:iter 5, 0.520293 sec, 1.787538 GB/sec per gpu, error 0.000000
```

### Multiple GPU machines

We can use `tools/launch.py` to launch a distributed job easily.
To show the idea, we run a worker and a server on the single machine. First we put the ip
into the `hosts` file

```bash
echo "127.0.0.1" >hosts
```

For more than one machines, we can replace `hosts` with the actual machine IPs
line by line. Then launch it by

```bash
~/mxnet/tools/bandwidth $ python ../launch.py -H hosts -n 1 python measure.py --kv-store dist_device_sync --gpus 0,1,2,3,4,5,6,7 --network resnet --depth 200
INFO:root:Namespace(batch_size=128, data_shape='128,3,224,224', depth=200, disp_batches=1, gpus='0,1,2,3,4,5,6,7', kv_store='dist_device_sync', network='resnet', num_batches=5, num_classes=1000, optimizer='None', test_results=1)
INFO:root:num of arrays = 205, total size = 257.991328 MB
INFO:root:iter 1, 0.295398 sec, 1.528395 GB/sec per gpu, error 0.000000
INFO:root:iter 2, 0.303159 sec, 1.489267 GB/sec per gpu, error 0.000000
INFO:root:iter 3, 0.290734 sec, 1.552913 GB/sec per gpu, error 0.000000
INFO:root:iter 4, 0.299437 sec, 1.507780 GB/sec per gpu, error 0.000000
INFO:root:iter 5, 0.285363 sec, 1.582142 GB/sec per gpu, error 0.000000
```

As we can see, the extra memory copy from GPUs to CPU, and then network card
harms the performance. We can slightly improve the performance using more than
1 server nodes:

```bash
~/mxnet/tools/bandwidth $ python ../launch.py -H hosts -n 1 -s 4 python measure.py --kv-store dist_device_sync --gpus 0,1,2,3,4,5,6,7 --network resnet --depth 200
INFO:root:Namespace(batch_size=128, data_shape='128,3,224,224', depth=200, disp_batches=1, gpus='0,1,2,3,4,5,6,7', kv_store='dist_device_sync', network='resnet', num_batches=5, num_classes=1000, optimizer='None', test_results=1)
INFO:root:num of arrays = 205, total size = 257.991328 MB
INFO:root:iter 1, 0.233309 sec, 1.935137 GB/sec per gpu, error 0.000000
INFO:root:iter 2, 0.253864 sec, 1.778453 GB/sec per gpu, error 0.000000
INFO:root:iter 3, 0.256627 sec, 1.759303 GB/sec per gpu, error 0.000000
INFO:root:iter 4, 0.250969 sec, 1.798965 GB/sec per gpu, error 0.000000
INFO:root:iter 5, 0.229306 sec, 1.968919 GB/sec per gpu, error 0.000000
```
