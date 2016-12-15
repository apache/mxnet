"""
test pretrained models
"""
from __future__ import print_function
import mxnet as mx
from common import find_mxnet, modelzoo
from common.util import download_file, get_gpus
from score import score

def download_data():
    download_file('http://data.mxnet.io/data/val-5k-256.rec', 'data/val-5k-256.rec')

def test_imagenet1k_resnet(**kwargs):
    models = ['imagenet1k-resnet-34',
              'imagenet1k-resnet-50',
              'imagenet1k-resnet-101',
              'imagenet1k-resnet-152']
    accs = [.72, .75, .765, .76]
    for (m, g) in zip(models, accs):
        acc = mx.metric.create('acc')
        (speed,) = score(model=m, data_val='data/val-5k-256.rec',
                         rgb_mean='0,0,0', metrics=acc, **kwargs)
        r = acc.get()[1]
        print('testing %s, acc = %f, speed = %f img/sec' % (m, r, speed))
        assert r > g and r < g + .1

def test_imagenet1k_inception_bn(**kwargs):
    acc = mx.metric.create('acc')
    m = 'imagenet1k-inception-bn'
    g = 0.72
    (speed,) = score(model=m,
                     data_val='data/val-5k-256.rec',
                     rgb_mean='123.68,116.779,103.939', metrics=acc, **kwargs)
    r = acc.get()[1]
    print('Tested %s acc = %f, speed = %f img/sec' % (m, r, speed))
    assert r > g and r < g + .1

if __name__ == '__main__':
    gpus = get_gpus()
    assert len(gpus) > 0
    batch_size = 16 * len(gpus)
    gpus = ','.join([str(i) for i in gpus])

    download_data()
    test_imagenet1k_resnet(gpus=gpus, batch_size=batch_size)
    test_imagenet1k_inception_bn(gpus=gpus, batch_size=batch_size)
