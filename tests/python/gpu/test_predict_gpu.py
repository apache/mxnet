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

import warnings

from mxnet import nd
from mxnet.test_utils import *

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))


def _get_model():
    symbol = "https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/resnet152/resnet-152-symbol.json"
    params = "https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/resnet152/resnet-152-0000.params"
    image = "https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/resnet152/kitten.jpg"
    for file in [symbol, params, image]:
        if not os.path.exists(file):
            download(file)


def _pre_process_image(path):
    img = mx.image.imread(path)
    if img is None:
        return None
    img = mx.image.imresize(img, 224, 224)  # resize
    img = img.transpose((2, 0, 1))  # Channel first
    img = img.expand_dims(axis=0)  # batchify
    a = nd.concat(img, dim=0)
    return a


def test_predict_gpu():
    _get_model()
    ctx = mx.gpu()
    sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-152', 0)
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))],
             label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)
    img = _pre_process_image('kitten.jpg')
    data_iter = mx.io.NDArrayIter([img], None, 1)

    # if module context is mx.gpu() and data is on cpu, warn users
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        mod.predict(data_iter)
        assert len(w) > 0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        mod.predict(img)
        assert len(w) > 0
