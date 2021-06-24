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

from common import xfail_when_nonstandard_decimal_separator
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.test_utils import default_context


@mx.util.use_np
class RoundSTENET(gluon.HybridBlock):
    def __init__(self, w_init, **kwargs):
        super(RoundSTENET, self).__init__(**kwargs)
        self.w = gluon.Parameter('w', shape=30, init=mx.initializer.Constant(w_init), grad_req='write')

    @staticmethod
    def expected_grads(self, in_data, w_init):
        return mx.np.round(in_data * w_init) + (in_data * w_init)

    @staticmethod
    def expected_output(self, in_data, w_init):
        return mx.np.round(in_data * w_init) * w_init

    def forward(self, x):
        # Simple forward function: round_ste(w*x)*w
        out = self.w.data(x.ctx) * x
        out = mx.npx.round_ste(out)
        # Uncomment to see how test fails with round
        # out = F.round(out)
        out = out * self.w.data(x.ctx)
        return out


@mx.util.use_np
class SignSTENET(gluon.HybridBlock):
    def __init__(self, w_init, **kwargs):
        super(SignSTENET, self).__init__(**kwargs)
        self.w = gluon.Parameter('w', shape=30, init=mx.initializer.Constant(w_init), grad_req='write')

    @staticmethod
    def expected_grads(self, in_data, w_init):
        return mx.np.sign(in_data * w_init) + (in_data * w_init)

    @staticmethod
    def expected_output(self, in_data, w_init):
        return mx.np.sign(in_data * w_init) * w_init

    def forward(self, x):
        # Simple forward function: sign_ste(w*x)*w
        out = self.w.data(x.ctx) * x
        out = mx.npx.sign_ste(out)
        # Uncomment to see how test fails with sign
        # out = F.sign(out)
        out = out * self.w.data(x.ctx)
        return out


def check_ste(net_type_str, w_init, hybridize, in_data, ctx=None):
    ctx = ctx or default_context()

    net = eval(net_type_str)(w_init=w_init)
    if hybridize:
        net.hybridize()
    # Init
    net.initialize(mx.init.Constant([w_init]), ctx=ctx)

    # Test:
    in_data = in_data.as_in_context(ctx)
    with mx.autograd.record():
        out = net(in_data)
    assert all(out == net.expected_output(in_data, w_init)), net_type_str + " output is " + str(out) + ", but" + \
                                                             " expected " + str(net.expected_output(in_data, w_init))

    out.backward()
    assert all(net.w.grad() == net.expected_grads(in_data, w_init)), net_type_str + " w grads are " + \
                                                                     str(net.w.grad()) + " but expected " + \
                                                                     str(net.expected_grads(in_data, w_init))
    with mx.autograd.record():
        out = net(in_data)
    assert all(out == net.expected_output(in_data, w_init)), net_type_str + " output is " + str(out) + ", but" + \
                                                             " expected " + str(net.expected_output(in_data, w_init))
    out.backward()
    assert all(net.w.grad() == net.expected_grads(in_data, w_init)), net_type_str + " w grads are " + \
                                                                     str(net.w.grad()) + " but expected " + \
                                                                     str(net.expected_grads(in_data, w_init))

@xfail_when_nonstandard_decimal_separator
def test_contrib_round_ste():
    # Test with random data
    in_data = mx.np.random.uniform(-10, 10, size=30)  # 10 and 30 are arbitrary numbers
    w_init = float(mx.np.random.uniform(-10, 10, size=1).item())
    check_ste(net_type_str="RoundSTENET", w_init=w_init, hybridize=True, in_data=in_data)
    check_ste(net_type_str="RoundSTENET", w_init=w_init, hybridize=False, in_data=in_data)

    # Test 1.5 (verifies that .5 rounds the same as in round)
    in_data = mx.np.array([1.5]*30)  # 10 and 30 are arbitrary numbers
    w_init = 1.
    check_ste(net_type_str="RoundSTENET", w_init=w_init, hybridize=True, in_data=in_data)
    check_ste(net_type_str="RoundSTENET", w_init=w_init, hybridize=False, in_data=in_data)

    # Test 0
    in_data = mx.np.array([0]*30)  # 10 and 30 are arbitrary numbers
    w_init = 0.
    check_ste(net_type_str="RoundSTENET", w_init=w_init, hybridize=True, in_data=in_data)
    check_ste(net_type_str="RoundSTENET", w_init=w_init, hybridize=False, in_data=in_data)


@xfail_when_nonstandard_decimal_separator
def test_contrib_sign_ste():
    in_data = mx.np.random.uniform(-10, 10, size=30)  # 10 and 30 are arbitrary numbers
    w_init = float(mx.np.random.uniform(-10, 10, size=1).item())
    check_ste(net_type_str="SignSTENET", w_init=w_init, hybridize=True, in_data=in_data)
    check_ste(net_type_str="SignSTENET", w_init=w_init, hybridize=False, in_data=in_data)

    # Test 0
    in_data = mx.np.array([0]*30)  # 10 and 30 are arbitrary numbers
    w_init = 0.
    check_ste(net_type_str="SignSTENET", w_init=w_init, hybridize=True, in_data=in_data)
    check_ste(net_type_str="SignSTENET", w_init=w_init, hybridize=False, in_data=in_data)

