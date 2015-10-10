# Multi-devices and multi-machines

![ps arch](https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/mxnet/multi-node/ps_arch.png)

| kvstore type | updt on kvstore | multi-devs | multi-workers | #ex per updt | max delay | updt place |
| :--- | :--- | ---:| ---:| ---:| ---:| ---:|
| none | no | no | no | *b* | *0* | worker\_0's dev\_0 |
| local | yes | yes | no | *b* | *0* | worker_0's cpu |
| local | no | yes | no | *b* | *0* | worker\_0's devs |
| device | no | yes | no | *b* | *0* | worker\_0's devs |
| dist | yes | yes | yes | *b* | *n* | servers |
| dist | no | yes | yes | *b × n* | *0* | workers' cpu |
