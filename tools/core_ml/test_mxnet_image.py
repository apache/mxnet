import mxnet as mx
import numpy as np
import unittest
import _mxnet_converter as mxnet_converter
import coremltools

VAL_DATA = 'data/val-5k-256.rec'
URL = 'http://data.mxnet.io/data/val-5k-256.rec'


def download_data():
    return mx.test_utils.download(URL, VAL_DATA)


def read_image(data_val, label_name):
    data = mx.io.ImageRecordIter(
        path_imgrec=data_val,
        label_width=1,
        preprocess_threads=4,
        batch_size=32,
        data_shape=(3,244,244),
        label_name=label_name,
        rand_corp=False,
        rand_mirror=False
    )
    return data


def load_model(model_name, epoch_num, data, label_names, gpus=''):
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, epoch_num)
    if gpus == '':
        devices = mx.cpu()
    else:
        devices = [mx.gpu(int(i)) for i in gpus.split(',')]
    mod = mx.mod.Module(
        symbol=sym,
        context=devices,
        label_names=label_names
    )
    mod.bind(
        for_training=False,
        data_shapes=data.provide_data,
        label_shapes=data.provide_label
    )
    mod.set_params(
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True
    )
    return mod


def is_correct_top_one(predict, label):
    assert isinstance(predict, np.ndarray)
    assert isinstance(label, np.float32)
    predicted_label = np.argmax(predict)
    return predicted_label == label


def is_correct_top_five(predict, label):
    assert isinstance(predict, np.ndarray)
    assert isinstance(label, np.float32)
    top_five_preds = set(predict.argsort()[-4:][::-1])
    return label in top_five_preds


class MXNetModelsTest(unittest.TestCase):
    def _test_image_prediction(self, model_name, epoch, label_name, force=False):
        try:
            data = read_image(VAL_DATA, label_name=label_name)
        except:
            download_data()
            data = read_image(VAL_DATA, label_name=label_name)

        mod = load_model(
            model_name=model_name,
            epoch_num=epoch,
            data=data,
            label_names=[label_name,]
        )

        input_shape = (1, 3, 244, 244)
        coreml_spec = mxnet_converter.convert(mod, data=input_shape, force=force)
        coreml_model = coremltools.models.MLModel(coreml_spec)

        mxnet_acc = []
        mxnet_top_k_acc = []
        coreml_acc = []
        coreml_top_k_acc = []

        num_batch = 0

        for batch in data:
            mod.forward(batch, is_train=False)
            mxnet_preds = mod.get_outputs()[0].asnumpy()
            data_numpy = batch.data[0].asnumpy()
            label_numpy = batch.label[0].asnumpy()
            for i in xrange(32):
                input_data = {'data': data_numpy[i]}
                coreml_predict = coreml_model.predict(input_data).values()[0].flatten()
                mxnet_predict = mxnet_preds[i]
                label = label_numpy[i]
                mxnet_acc.append(is_correct_top_one(mxnet_predict, label))
                mxnet_top_k_acc.append(is_correct_top_five(mxnet_predict, label))
                coreml_acc.append(is_correct_top_one(coreml_predict, label))
                coreml_top_k_acc.append(is_correct_top_five(coreml_predict, label))
                num_batch += 1
            if (num_batch == 5): break # we only use a subset of the batches.

        print "MXNet acc %s" % np.mean(mxnet_acc)
        print "Coreml acc %s" % np.mean(coreml_acc)
        print "MXNet top 5 acc %s" % np.mean(mxnet_top_k_acc)
        print "Coreml top 5 acc %s" % np.mean(coreml_top_k_acc)
        self.assertAlmostEqual(np.mean(mxnet_acc), np.mean(coreml_acc), delta=1e-1)
        self.assertAlmostEqual(np.mean(mxnet_top_k_acc), np.mean(coreml_top_k_acc), delta=1e-2)

    def test_squeezenet(self):
        print "Testing Image Classification with Squeezenet"
        self._test_image_prediction(model_name='squeezenet_v1.1', epoch=0, label_name='prob_label')

#    TODO
#     def test_inception_with_batch_normalization(self):
#         print "Testing Image Classification with Inception/BatchNorm"
#         self._test_image_prediction(model_name='Inception-BN', epoch=126, label_name='softmax_label', force=True)

    def test_resnet18(self):
        print "Testing Image Classification with ResNet18"
        self._test_image_prediction(model_name='resnet-18', epoch=0, label_name='softmax_label', force=True)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(MXNetModelsTest)
    unittest.TextTestRunner(verbosity=2).run(suite)