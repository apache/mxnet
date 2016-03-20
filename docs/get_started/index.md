# Get Started

## overview

```python
import mxnet
num_epoch = 38
model_prefix = "model/cifar_100"

softmax = inception(100, 1.0)

model = mx.model.FeedForward(
ctx=mx.gpu(), symbol=softmax, num_epoch=num_epoch,
learning_rate=0.05, momentum=0.9, wd=0.0001)
```

## next steps

- [how to build](build.html)
