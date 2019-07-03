#!/usr/bin/env bash

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

# Downloading the data and model
mkdir -p model

python - <<EOF
import mxnet as mx
import gluoncv
from gluoncv.model_zoo import get_model
net = get_model('InceptionV3', pretrained=True)
net.hybridize()
net(mx.nd.zeros((1,3,299,299)))
net.export("model/model", epoch=1)
EOF

# Running the example with dog image.
if [ "$(uname)" == "Darwin" ]; then
    DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:../../../lib ./inception_inference --symbol "./model/model-symbol.json" --params "./model/model-0001.params" --synset "./model/synset.txt" --input_shape "3 299 299" --image "./model/dog.jpg" 2&> inception_inference.log
else
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../../../lib ./inception_inference --symbol "./model/model-symbol.json" --params "./model/model-0001.params" --synset "./model/synset.txt" --input_shape "3 299 299" --image "./model/dog.jpg" 2&> inception_inference.log
fi
result=`grep -c "pug-dog" inception_inference.log`
if [ $result == 1 ];
then
    echo "PASS: inception_inference correctly identified the image."
    exit 0
else
    echo "FAIL: inception_inference FAILED to identify the image."
    exit 1
fi
