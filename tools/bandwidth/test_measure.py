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

"""
test measure.py
"""
from measure import run
import subprocess
import logging
def get_gpus():
    try:
        re = subprocess.check_output(["nvidia-smi", "-L"], universal_newlines=True)
    except OSError:
        return ''
    gpus = [i for i in re.split('\n') if 'GPU' in i]
    return ','.join([str(i) for i in range(len(gpus))])

def test_measure(**kwargs):
    logging.info(kwargs)
    res = run(image_shape='3,224,224', num_classes=1000,
              num_layers=50, disp_batches=2, num_batches=2, test_results=1, **kwargs)
    assert len(res) == 1
    assert res[0].error < 1e-4

if __name__ == '__main__':
    gpus = get_gpus()
    assert gpus is not ''
    test_measure(gpus=gpus, network='alexnet', optimizer=None, kv_store='device')
    test_measure(gpus=gpus, network='resnet', optimizer='sgd', kv_store='device')
    test_measure(gpus=gpus, network='inception-bn', optimizer=None, kv_store='local')
    test_measure(gpus=gpus, network='resnet', optimizer=None, kv_store='local')
    test_measure(gpus=gpus, network='resnet', optimizer='sgd', kv_store='local')
