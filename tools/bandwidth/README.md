<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Measure communication bandwidth

MXNet provides multiple ways to communicate data. The best choice depends on
both the physical machines and neural network structure. This folder provides
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
  - `dist_sync_device` : similar to `dist_sync` but try best to use GPU for communication
  - `dist_async` : similar to `dist_sync` but uses asynchronous communication
  - `dist_async_device` : similar to `dist_async` but try best to use GPU for communication

## Samples

### Single machine with multiple GPUs

- Use resnet 200 layers on GPU 0, 1, 2, and 3

```bash
~/mxnet/tools/bandwidth $ python measure.py --kv-store device --gpus 0,1 --network resnet --num-layers 200
INFO:root:Namespace(disp_batches=1, gpus='0,1', image_shape='3,224,224', kv_store='device', network='resnet', num_batches=5, num_classes=1000, num_layers=200, optimizer='None', test_results=1)
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

- Use 8 GPUs, it saturates the single 16x link between GPU 0,1,2,3 and GPU 4,5,6,7.

```bash
~/mxnet/tools/bandwidth $ python measure.py --kv-store device --gpus 0,1,2,3,4,5,6,7 --network resnet --num-layers 200
INFO:root:Namespace(disp_batches=1, gpus='0,1,2,3,4,5,6,7', image_shape='3,224,224', kv_store='device', network='resnet', num_batches=5, num_classes=1000, num_layers=200, optimizer='None', test_results=1)
INFO:root:num of arrays = 205, total size = 257.991328 MB
INFO:root:iter 1, 0.102321 sec, 4.412429 GB/sec per gpu, error 0.000000
INFO:root:iter 2, 0.100345 sec, 4.499330 GB/sec per gpu, error 0.000000
INFO:root:iter 3, 0.097317 sec, 4.639322 GB/sec per gpu, error 0.000000
INFO:root:iter 4, 0.099873 sec, 4.520586 GB/sec per gpu, error 0.000000
INFO:root:iter 5, 0.100774 sec, 4.480169 GB/sec per gpu, error 0.000000
```

- Now let's only use GPU-CPU communication, it saturates the single 16x link
between all GPUs and the CPU.

```bash
~/mxnet/tools/bandwidth $ python measure.py --kv-store local --gpus 0,1,2,3,4,5,6,7 --network resnet --num-layers 200
INFO:root:Namespace(disp_batches=1, gpus='0,1,2,3,4,5,6,7', image_shape='3,224,224', kv_store='local', network='resnet', num_batches=5, num_classes=1000, num_layers=200, optimizer='None', test_results=1)
INFO:root:num of arrays = 205, total size = 257.991328 MB
INFO:root:iter 1, 0.290164 sec, 1.555964 GB/sec per gpu, error 0.000000
INFO:root:iter 2, 0.293963 sec, 1.535856 GB/sec per gpu, error 0.000000
INFO:root:iter 3, 0.294468 sec, 1.533222 GB/sec per gpu, error 0.000000
INFO:root:iter 4, 0.290657 sec, 1.553325 GB/sec per gpu, error 0.000000
INFO:root:iter 5, 0.290799 sec, 1.552567 GB/sec per gpu, error 0.000000
```

- Finally we change to Inception-v3 which requires input image size to be `3*299*299`, and also run the `sgd` optimizor

```bash
~/mxnet/tools/bandwidth $ python measure.py --kv-store device --gpus 0,1,2,3,4,5,6,7 --image-shape 3,299,299 --network inception-v3 --optimizer sgd
libdc1394 error: Failed to initialize libdc1394
INFO:root:Namespace(disp_batches=1, gpus='0,1,2,3,4,5,6,7', image_shape='3,299,299', kv_store='device', network='inception-v3', num_batches=5, num_classes=1000, num_layers=152, optimizer='sgd', test_results=1)
INFO:root:num of arrays = 96, total size = 95.200544 MB
INFO:root:iter 1, 0.086527 sec, 1.925424 GB/sec per gpu, error 0.000000
INFO:root:iter 2, 0.057934 sec, 2.875700 GB/sec per gpu, error 0.000000
INFO:root:iter 3, 0.055442 sec, 3.004967 GB/sec per gpu, error 0.000000
INFO:root:iter 4, 0.055579 sec, 2.997555 GB/sec per gpu, error 0.000000
INFO:root:iter 5, 0.055107 sec, 3.023220 GB/sec per gpu, error 0.000000
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
~/mxnet/tools/bandwidth $ python ../launch.py -H hosts -n 1 python measure.py --kv-store dist_device_sync --gpus 0,1,2,3,4,5,6,7 --network resnet --num-layers 200
INFO:root:Namespace(disp_batches=1, gpus='0,1,2,3,4,5,6,7', image_shape='3,224,224', kv_store='dist_device_sync', network='resnet', num_batches=5, num_classes=1000, num_layers=200, optimizer='None', test_results=1)
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
~/mxnet/tools/bandwidth $ python ../launch.py -H hosts -n 1 -s 4 python measure.py --kv-store dist_device_sync --gpus 0,1,2,3,4,5,6,7 --network resnet --num-layers 200
INFO:root:Namespace(disp_batches=1, gpus='0,1,2,3,4,5,6,7', image_shape='3,224,224', kv_store='dist_device_sync', network='resnet', num_batches=5, num_classes=1000, num_layers=200, optimizer='None', test_results=1)
INFO:root:num of arrays = 205, total size = 257.991328 MB
INFO:root:iter 1, 0.233309 sec, 1.935137 GB/sec per gpu, error 0.000000
INFO:root:iter 2, 0.253864 sec, 1.778453 GB/sec per gpu, error 0.000000
INFO:root:iter 3, 0.256627 sec, 1.759303 GB/sec per gpu, error 0.000000
INFO:root:iter 4, 0.250969 sec, 1.798965 GB/sec per gpu, error 0.000000
INFO:root:iter 5, 0.229306 sec, 1.968919 GB/sec per gpu, error 0.000000
```
