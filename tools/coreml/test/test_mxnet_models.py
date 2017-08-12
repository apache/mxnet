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
import sys
import os
current_working_directory = os.getcwd()
sys.path.append(current_working_directory + "/..")
sys.path.append(current_working_directory + "/../converter/")
import _mxnet_converter as mxnet_converter
from collections import namedtuple


def _mxnet_remove_batch(input_data):
    for blob in input_data:
        input_data[blob] = np.reshape(input_data[blob], input_data[blob].shape[1:])
    return input_data


def _kl_divergence(distribution1, distribution2):
    """ Calculates Kullback-Leibler Divergence b/w two distributions.

    Parameters
    ----------
    distribution1: list of floats
    distribution2: list of floats
    """
    assert len(distribution1) == len(distribution2)
    n = len(distribution1)
    result = 1./n * sum(distribution1 * (np.log(distribution1) - np.log(distribution2)))
    return result


class ModelsTest(unittest.TestCase):
    """
    Unit test class that tests converter on entire MXNet models .
    In order to test each unit test converts MXNet model into CoreML model using the converter, generate predictions
    on both MXNet and CoreML and verifies that predictions are same (or similar).
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

    def _test_model(self, model_name, epoch_num, input_shape=(1, 3, 224, 224), files=None):
        """ Tests whether the converted CoreML model's preds are equal to MXNet preds for a given model or not.

        Parameters
        ----------
        model_name: str
            Prefix of the MXNet model name as stored on the local directory.

        epoch_num : int
            Epoch number of model we would like to load.

        input_shape: tuple
            The shape of the input data in the form of (batch_size, channels, height, width)

        files: list of strings
            List of URLs pertaining to files that need to be downloaded in order to use the model.
        """

        if files is not None:
            print("Downloading files from urls: %s" % (files))
            for url in files:
                mx.test_utils.download(url)
                print("Downloaded %s" % (url))

        module = self._load_model(
            model_name=model_name,
            epoch_num=epoch_num,
            input_shape=input_shape
        )

        coreml_model = mxnet_converter.convert(module, input_shape={'data': input_shape})

        # Get predictions from MXNet and coreml
        div=[] # For storing KL divergence for each input.
        for _ in xrange(1):
            np.random.seed(1993)
            input_data = {'data': np.random.uniform(0, 1, input_shape).astype(np.float32)}
            Batch = namedtuple('Batch', ['data'])
            module.forward(Batch([mx.nd.array(input_data['data'])]), is_train=False)
            mxnet_pred = module.get_outputs()[0].asnumpy().flatten()
            coreml_pred = coreml_model.predict(_mxnet_remove_batch(input_data)).values()[0].flatten()
            self.assertEqual(len(mxnet_pred), len(coreml_pred))
            div.append(_kl_divergence(mxnet_pred, coreml_pred))

        print "Average KL divergence is % s" % np.mean(div)
        self.assertTrue(np.mean(div) < 1e-4)

    def test_pred_inception_bn(self):
        self._test_model(model_name='Inception-BN', epoch_num=126,
                         files=["http://data.mxnet.io/models/imagenet/inception-bn/Inception-BN-0126.params",
                                "http://data.mxnet.io/models/imagenet/inception-bn/Inception-BN-symbol.json"])

    def test_pred_squeezenet_v11(self):
        self._test_model(model_name='squeezenet_v1.1', epoch_num=0,
                         files=["http://data.mxnet.io/models/imagenet/squeezenet/squeezenet_v1.1-symbol.json",
                                "http://data.mxnet.io/models/imagenet/squeezenet/squeezenet_v1.1-0000.params"])

    def test_pred_resnet_50(self):
        self._test_model(model_name='resnet-50', epoch_num=0,
                         files=["http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50-symbol.json",
                                "http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50-0000.params"])

    def test_pred_vgg16(self):
        self._test_model(model_name='vgg16', epoch_num=0,
                         files=["http://data.mxnet.io/models/imagenet/vgg/vgg16-symbol.json",
                                "http://data.mxnet.io/models/imagenet/vgg/vgg16-0000.params"])

    def test_pred_nin(self):
        self._test_model(model_name='nin', epoch_num=0,
                         files=["http://data.dmlc.ml/models/imagenet/nin/nin-symbol.json",
                                "http://data.dmlc.ml/models/imagenet/nin/nin-0000.params"])

    @unittest.skip("You need to download and unzip file: "
                   "http://data.mxnet.io/models/imagenet/inception-v3.tar.gz in order to run this test.")
    def test_pred_inception_v3(self):
        self._test_model(model_name='Inception-7', epoch_num=1, input_shape=(1, 3, 299, 299))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ModelsTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
