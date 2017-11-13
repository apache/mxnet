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

import mxnet as mx

data_shape = (1,3,5,5)
class SimpleData(object):

    def __init__(self, data):
        self.data = data

data = mx.sym.Variable('data')
conv = mx.sym.Convolution(data=data, kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=1)
mon = mx.mon.Monitor(1)


mod = mx.mod.Module(conv)
mod.bind(data_shapes=[('data', data_shape)])
mod._exec_group.install_monitor(mon)
mod.init_params()

input_data = mx.nd.ones(data_shape)
mod.forward(data_batch=SimpleData([input_data]))
res = mod.get_outputs()[0].asnumpy()
print(res)
