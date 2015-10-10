# Multi-devices and multi-machines

Architecture

<img src=https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/mxnet/multi-node/ps_arch.png width=400/>

| kvstore type | update on kvstore | multi-devices | multi-workers | #ex per update | max delay | update place |
| :--- | :--- | ---:| ---:| ---:| ---:| ---:|
| none | no | no | no | *b* | *0* | worker<sub>0</sub>'s dev<sub>0<\sub> |
| local | yes | yes | no | *b* | *0* | worker<sub>0</sub>'s cpu |
| local | no | yes | no | *b* | *0* | worker<sub>0</sub>'s devs |
| device | no | yes | no | *b* | *0* | worker<sub>0</sub>'s devs |
| dist | yes | yes | yes | *b* | *n* | servers |
| dist | no | yes | yes | *b × n* | *0* | workers' cpu |
