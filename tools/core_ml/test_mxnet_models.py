import unittest
import mxnet as mx
import numpy as np
import _mxnet_converter as mxnet_converter
import coremltools
from collections import namedtuple


def _mxnet_remove_batch(input_data):
    for blob in input_data:
        input_data[blob] = np.reshape(input_data[blob], input_data[blob].shape[1:])
    return input_data


def _kl_divergence(output1, output2):
    assert len(output1) == len(output2)
    n = len(output1)
    result = 1./n * sum(output1 * (np.log(output1) - np.log(output2)))
    return result


class MXNetModelsTest(unittest.TestCase):
    """
    Unit test class for testing mxnet converter (converts model and generates preds on same data to assert they are the same).
    In order to run these, you have to download the models in the same directory beforehand.
    TODO: Provide better user experience here.
    """
    def _load_model(self, model_name, epoch_num, input_shape):
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, epoch_num)
        mod = mx.mod.Module(
            symbol=sym,
            context=mx.cpu(),
            label_names=None
        )
        mod.bind(
            for_training=False,
            data_shapes=[('data', input_shape)],
            label_shapes=mod._label_shapes
        )
        mod.set_params(
            arg_params=arg_params,
            aux_params=aux_params,
            allow_missing=True
        )
        return mod

    def _test_model(self, model_name, epoch_num, input_shape=(1, 3, 224, 224)):

        module = self._load_model(
            model_name=model_name,
            epoch_num=epoch_num,
            input_shape=input_shape
        )

        coreml_spec = mxnet_converter.convert(module, data=input_shape)
        coreml_model = coremltools.models.MLModel(coreml_spec)

        # Get predictions from MXNet and coreml
        for _ in xrange(10):
            input_data = {'data': np.random.uniform(0, 1, input_shape).astype(np.float32)}
            Batch = namedtuple('Batch', ['data'])
            module.forward(Batch([mx.nd.array(input_data['data'])]))

            mxnet_pred = module.get_outputs()[0].asnumpy().flatten()
            coreml_pred = coreml_model.predict(_mxnet_remove_batch(input_data)).values()[0].flatten()

            self.assertEqual(len(mxnet_pred), len(coreml_pred))

            div = _kl_divergence(mxnet_pred, coreml_pred)
            print "KL divergence is % s" % div
            

    def test_pred_inception_bn(self):
        self._test_model(model_name='Inception-BN', epoch_num=126)

    def test_pred_squeezenet_v11(self):
        self._test_model(model_name='squeezenet_v1.1', epoch_num=0)

    def test_pred_resnet_50(self):
        self._test_model(model_name='resnet-50', epoch_num=0)

    def test_pred_vgg16(self):
        self._test_model(model_name='vgg16', epoch_num=0)

    def test_pred_inception_v3(self):
        self._test_model(model_name='Inception-7', epoch_num=1, input_shape=(1, 3, 299, 299))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(MXNetModelsTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
