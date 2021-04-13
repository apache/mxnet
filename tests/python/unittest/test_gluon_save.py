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

def test_save():
    class MyBlock(mx.gluon.Block):
        def __init__(self, **kwargs):
            super(MyBlock, self).__init__(**kwargs)
            self.layers = []
        def add(self, block):
            self.layers.append(block)
            self.register_child(block)
        def forward(self, x, *args):
            out = (x,) + args
            for block in self._children.values():
                out = block()(*out)
            return out

    def createNet():
        inside = MyBlock()
        dense = mx.gluon.nn.Dense(10)
        inside.add(dense)
        net = MyBlock()
        net.add(inside)
        net.add(mx.gluon.nn.Dense(10))
        return net

    # create and initialize model
    net1 = createNet()
    net1.initialize()
    # hybridize (the hybridizeable blocks, ie. the Dense layers)
    net1.hybridize()
    x = mx.nd.zeros((1,10))
    out1 = net1(x)

    # save hybridized model
    net1.save('MyModel')

    # create a new model, uninitialized
    net2 = createNet()
    # reload hybridized model
    net2.load('MyModel')
    # run inference again
    out2 = net2(x)
    mx.test_utils.assert_almost_equal(out1.asnumpy(), out2.asnumpy())
