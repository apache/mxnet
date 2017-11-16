import mxnet as mx
import numpy as np
import json

def check_metric(metric, *args, **kwargs):
    metric = mx.metric.create(metric, *args, **kwargs)
    str_metric = json.dumps(metric.get_config())
    metric2 = mx.metric.create(str_metric)

    assert metric.get_config() == metric2.get_config()


def test_metrics():
    check_metric('acc', axis=0)
    check_metric('f1')
    check_metric('perplexity', -1)
    check_metric('pearsonr')
    check_metric('nll_loss')
    composite = mx.metric.create(['acc', 'f1'])
    check_metric(composite)

def test_nll_loss():
    metric = mx.metric.create('nll_loss')
    pred = mx.nd.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
    label = mx.nd.array([2, 1])
    metric.update([label], [pred])
    _, loss = metric.get()
    expected_loss = 0.0
    expected_loss = -(np.log(pred[0][2].asscalar()) + np.log(pred[1][1].asscalar())) / 2
    assert loss == expected_loss

if __name__ == '__main__':
    import nose
    nose.runmodule()
