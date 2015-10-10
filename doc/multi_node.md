# Multi-devices and multi-machines

## Architecture

A device could be a GPU card, CPU, or other computational units.

<img src=https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/mxnet/multi-node/ps_arch.png width=400/>

- *b*: the batch size set by users
- *k*: the number of devices used on a worker (could vary for different workers)
- *n*: the number of workers (often mean machines)

- *number examples per update*: for each update, the number of examples used to
  calculate the averaged gradients. Often the larger, the slower the convergence.
- *number examples per device*: the number of examples batched to one device
  each time. Often the larger, the better the performance.
- *max delay*: The maximal delay of the weight a worker can get. Given a worker,
  a delay *d* for weight *w* means when this worker uses *w* (to calculate the
  gradient), *w* have been already updated by *d* times on some other places. A
  larger delay often improves the performance, but may slows down the
  convergence.


| kvstore type | update on kvstore | multi devices | multi workers | #ex per device | #ex per update | max delay | update place |
| :--- | :--- | ---:| ---:| ---:| ---:| ---:| ---:|
| none | no | no | no | *b* | *b* | *0* | dev<sub>0</sub> on worker<sub>0</sub> |
| local | yes | yes | no | *b / k* | *b* | *0* | cpu on worker<sub>0</sub> |
| local | no | yes | no  | *b/k* | *b* | *0* | devs on worker<sub>0</sub> |
| device | no | yes | no | *b/k* |*b* | *0* | devs on worker<sub>0</sub> |
| dist | yes | yes | no | yes | *b/k* |*b* | *n* | servers |
| dist | no | yes | no | *b/k* | *b × n* | *0* | cpus on workers |
