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

    def _test_model(self, model_name, epoch_num):

        input_shape = (1, 3, 224, 224)

        module = self._load_model(
            model_name=model_name,
            epoch_num=epoch_num,
            input_shape=input_shape)

        # Get predictions from MXNet and coreml
        input_data = {'data': np.random.uniform(-0.1, 0.1, input_shape)}
        Batch = namedtuple('Batch', ['data'])
        module.forward(Batch([mx.nd.array(input_data['data'])]))
        mxnet_preds = module.get_outputs()[0].asnumpy().flatten()

        coreml_spec = mxnet_converter.convert(module, data=input_shape)
        coreml_model = coremltools.models.MLModel(coreml_spec)
        coreml_preds = coreml_model.predict(_mxnet_remove_batch(input_data)).values()[0].flatten()

        # Check prediction accuracy
        self.assertEquals(len(mxnet_preds), len(coreml_preds))

        for i in range(len(mxnet_preds)):
            self.assertAlmostEquals(mxnet_preds[i], coreml_preds[i], delta = 1e-7)

    def test_convert_inception_bn(self):
        input_shape = (1, 3, 224, 224)
        module = self._load_model('Inception-BN', 126, input_shape)
        mxnet_converter.convert(module, data=input_shape)

    def test_convert_squeezenet_v11(self):
        input_shape = (1, 3, 224, 224)
        module = self._load_model('squeezenet_v1.1', 0, input_shape)
        mxnet_converter.convert(module, data=input_shape)

    def test_convert_resnet_50(self):
        input_shape = (1, 3, 224, 224)
        module = self._load_model('resnet-50', 0, input_shape)
        mxnet_converter.convert(module, data=input_shape)

    def test_convert_vgg16(self):
        input_shape = (1, 3, 224, 224)
        module = self._load_model('vgg-16', 0, input_shape)
        mxnet_converter.convert(module, data=input_shape)

    def test_pred_inception_bn(self):
        self._test_model(model_name='Inception-BN', epoch_num=126)

    def test_pred_squeezenet_v11(self):
        self._test_model(model_name='squeezenet_v1.1', epoch_num=0)

    def test_pred_resnet_50(self):
        self._test_model(model_name='resnet-50', epoch_num=0)

    def test_pred_vgg16(self):
        self._test_model(model_name='vgg16', epoch_num=0)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(MXNetModelsTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
