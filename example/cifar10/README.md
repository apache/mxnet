CIFAR 10 Example
================
* [cifar10.py](cifar10.py) provides an example to train a dual path model on CIFAR-10. Original network in CXXNET is able to achieve 91.6% accuracy.


Machine: Dual Xeon E5-1650 3.5GHz, 4 GTX 980, Cuda 6.5

run `cifar.py`:


| | 1 GPU | 2 GPUs | 4 GPUs |
| --- | --- | --- | --- |
| cxxnet | 362 img/sec | 675 img/sec | 1282 img/sec |
| mxnet | 467 img/sec | 923 img/sec | 1820 img/sec |
| mxnet + cudnn v3 | 842 img/sec | 1640 img/sec | 2943 img/sec |

sample output

```
~/mxnet/example/cifar10 $ python cifar10.py
INFO:root:Start training with 4 devices
Start training with 4 devices
INFO:root:Iteration[0] Train-accuracy=0.507613
Iteration[0] Train-accuracy=0.507613
INFO:root:Iteration[0] Time cost=34.800
Iteration[0] Time cost=34.800
INFO:root:Iteration[0] Validation-accuracy=0.641021
Iteration[0] Validation-accuracy=0.641021
INFO:root:Iteration[1] Train-accuracy=0.679408
Iteration[1] Train-accuracy=0.679408
INFO:root:Iteration[1] Time cost=34.481
Iteration[1] Time cost=34.481
INFO:root:Iteration[1] Validation-accuracy=0.720152
Iteration[1] Validation-accuracy=0.720152
INFO:root:Iteration[2] Train-accuracy=0.740825
Iteration[2] Train-accuracy=0.740825
INFO:root:Iteration[2] Time cost=34.463
Iteration[2] Time cost=34.463
INFO:root:Iteration[2] Validation-accuracy=0.755709
Iteration[2] Validation-accuracy=0.755709
```

results from cxxnet for reference

```
CXXNET Result:
step1: wmat_lr = 0.05, bias_lr = 0.1, mom = 0.9
[1] train-error:0.452865  val-error:0.3614
[2] train-error:0.280231  val-error:0.2504
[3] train-error:0.220968  val-error:0.2456
[4] train-error:0.18746 val-error:0.2145
[5] train-error:0.165221  val-error:0.1796
[6] train-error:0.150056  val-error:0.1786
[7] train-error:0.134571  val-error:0.157
[8] train-error:0.122582  val-error:0.1429
[9] train-error:0.113891  val-error:0.1398
[10]  train-error:0.106458  val-error:0.1469
[11]  train-error:0.0985054 val-error:0.1447
[12]  train-error:0.0953684 val-error:0.1494
[13]  train-error:0.0872962 val-error:0.1311
[14]  train-error:0.0832401 val-error:0.1544
[15]  train-error:0.0773857 val-error:0.1268
[16]  train-error:0.0743087 val-error:0.125
[17]  train-error:0.0714114 val-error:0.1189
[18]  train-error:0.066616  val-error:0.1424
[19]  train-error:0.0651175 val-error:0.1322
[20]  train-error:0.0616808 val-error:0.111
step2: lr = 0.01, bias_lr = 0.02, mom = 0.9
[21]  train-error:0.033368  val-error:0.0907
[22]  train-error:0.0250959 val-error:0.0876
[23]  train-error:0.0220388 val-error:0.0867
[24]  train-error:0.0195812 val-error:0.0848
[25]  train-error:0.0173833 val-error:0.0872
[26]  train-error:0.0154052 val-error:0.0878
[27]  train-error:0.0141264 val-error:0.0863
[28]  train-error:0.0134071 val-error:0.0865
[29]  train-error:0.0116688 val-error:0.0878
[30]  train-error:0.0106298 val-error:0.0873
step3: lr = 0.001, bias_lr = 0.002, mom = 0.9
[31]  train-error:-nan  val-error:0.0873
[31]  train-error:0.0067735 val-error:0.0859
[32]  train-error:0.0049952 val-error:0.0835
[33]  train-error:0.00485534  val-error:0.0849
[34]  train-error:0.00367647  val-error:0.0839
[35]  train-error:0.0034367 val-error:0.0844
[36]  train-error:0.00275735  val-error:0.084
[37]  train-error:0.00221787  val-error:0.083
[38]  train-error:0.00171835  val-error:0.0838
[39]  train-error:0.00125879  val-error:0.0833
[40]  train-error:0.000699329 val-error:0.0842
```
