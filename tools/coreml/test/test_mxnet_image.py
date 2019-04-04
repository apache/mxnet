
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import unittest
import mxnet as mx
import numpy as np

from six.moves import xrange
from converter._mxnet_converter import convert
from converter import utils

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
        data_shape=(3, 224, 224),
        label_name=label_name,
        rand_corp=False,
        rand_mirror=False,
        shuffle=True
    )
    return data


def is_correct_top_one(predict, label):
    assert isinstance(predict, np.ndarray)
    assert isinstance(label, np.float32)
    predicted_label = np.argmax(predict)
    return predicted_label == label


def is_correct_top_five(predict, label):
    assert isinstance(predict, np.ndarray)
    assert isinstance(label, np.float32)
    top_five_preds = set(predict.argsort()[-5:])
    return label in top_five_preds


class ImageNetTest(unittest.TestCase):
    def _test_image_prediction(self, model_name, epoch, label_name):
        try:
            data = read_image(VAL_DATA, label_name=label_name)
        except:
            download_data()
            data = read_image(VAL_DATA, label_name=label_name)

        mod = utils.load_model(
            model_name=model_name,
            epoch_num=epoch,
            data_shapes=data.provide_data,
            label_shapes=data.provide_label,
            label_names=[label_name, ]
        )

        input_shape = (1, 3, 224, 224)
        coreml_model = convert(mod, input_shape={'data': input_shape})

        mxnet_acc = []
        mxnet_top_5_acc = []
        coreml_acc = []
        coreml_top_5_acc = []

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
                mxnet_top_5_acc.append(is_correct_top_five(mxnet_predict,
                                                           label))
                coreml_acc.append(is_correct_top_one(coreml_predict, label))
                coreml_top_5_acc.append(is_correct_top_five(coreml_predict,
                                                            label))
                num_batch += 1
            if (num_batch == 5):
                break  # we only use a subset of the batches.

        print("MXNet acc %s" % np.mean(mxnet_acc))
        print("Coreml acc %s" % np.mean(coreml_acc))
        print("MXNet top 5 acc %s" % np.mean(mxnet_top_5_acc))
        print("Coreml top 5 acc %s" % np.mean(coreml_top_5_acc))
        self.assertAlmostEqual(np.mean(mxnet_acc), np.mean(coreml_acc), delta=1e-4)
        self.assertAlmostEqual(np.mean(mxnet_top_5_acc),
                               np.mean(coreml_top_5_acc),
                               delta=1e-4)

    def test_squeezenet(self):
        print("Testing Image Classification with Squeezenet")
        self._test_image_prediction(model_name='squeezenet_v1.1', epoch=0,
                                    label_name='prob_label')

    def test_inception_with_batch_normalization(self):
        print("Testing Image Classification with Inception/BatchNorm")
        self._test_image_prediction(model_name='Inception-BN', epoch=126,
                                    label_name='softmax_label')

    def test_resnet18(self):
        print("Testing Image Classification with ResNet18")
        self._test_image_prediction(model_name='resnet-18', epoch=0,
                                    label_name='softmax_label')

    def test_vgg16(self):
        print("Testing Image Classification with vgg16")
        self._test_image_prediction(model_name='vgg16', epoch=0,
                                    label_name='prob_label')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ImageNetTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
