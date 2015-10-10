# Multi-devices and multi-machines

## Architecture

A device could be a GPU card, CPU, or other computational units.

<img src=https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/mxnet/multi-node/ps_arch.png width=400/>

- **n** : the number of workers (often mean machines)
- **k** : the number of devices used on a worker (could vary for different workers)
- **b** : the batch size set by users

- **number examples per update** : for each update, the number of examples used to
  calculate the averaged gradients. Often the larger, the slower the convergence.
- **number examples per device** : the number of examples batched to one device
  each time. Often the larger, the better the performance.
- **max delay** : The maximal delay of the weight a worker can get. Given a worker,
  a delay *d* for weight *w* means when this worker uses *w* (to calculate the
  gradient), *w* have been already updated by *d* times on some other places. A
  larger delay often improves the performance, but may slows down the
  convergence.


| kvstore type | multi-devices | multi-workers | #ex per device | #ex per update | max delay |
| :--- | --- | --- | --- | --- | --- |
| `none` |  no | no | *b* | *b* | *0* |
| `local` / `device` | yes | no | *b / k* | *b* | *0* |
| `dist_sync` | yes | yes | *b / k* | *b × n* | *0* |
| `dist_async` | yes | yes |  *b / k* | *b* | inf |


## Multiple devices on a single machine

Both `local` and `device` can handle the situation that a single machine with
multiple devices. They give the some results (model accuracy) as the single
device case. But comparing to the latter, each device only processes *1 / k*
examples each time (also consumes *1 / k* device memory), so we often increase
the batch size *b* for better system performance.

We can further fine tune the system performance by specifying where to average
the gradients over all devices, and where to update the weight:

| case | kvstore type | update on kvstore | average gradient | perform update |
| :--- | :--- | :--- | --- | --- | --- |
| 1 | 'local' | yes | CPU | CPU |
| 2 | 'local' | no | CPU | all devices |
| 3 | 'device | yes | a device | all devices |

- On case 1, gradients are first copied to main memory, next averaged on CPU,
  and then update the weight on CPU. It is suitable when the average size of
  weights are not large and there are a large number of weight. For example the
  google Inception network.

- Case 2 is similar to 1 except that the averaged gradients are copied back to
  the devices, and then weights are updated on devices. It is faster than 1 when
  the weight size is large so we can use the device to accelerate the computation
  (but we increase the workload by *k* times). Examples are AlexNet on
  imagenet.

- Case 3 is similar to 1 except that the gradient are averaged on a chosen
  device. It may take advantage of the possible device-to-device communication, and may
  accelerate the averaging step. It is faster than 2 when the gradients are
  huge. But it requires more device memory.

## Multiple machines


Both `dist_async` and `dist_sync` can handle the multiple machines
situation. But they are different on both semantic and performance.

- `dist_sync`: the gradients are first averaged on the servers, and then send to
  back to workers for updating the weight. It is similar to `local` and
  `update_on_kvstore=false` if we treat a machine as a device.  It guarantees
  almost identical convergence with the single machine single device situation
  if reduces the batch size to *b / n*. However, it requires synchronization
  between all workers, and therefore may harm the system performance.

- `dist_async`: the gradient is sent to the servers, and the weight is updated
  there. The weights a worker has may be stale.
  (TODO) make the max delay be settable?
