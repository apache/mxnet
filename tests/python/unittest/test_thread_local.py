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

import threading
import numpy as np
import mxnet as mx
from mxnet import context, attribute
from mxnet.context import Context
from mxnet.attribute import AttrScope
from mxnet.test_utils import assert_almost_equal, set_default_context
from mxnet.util import _NumpyArrayScope, set_np_shape


def test_context():
    ctx_list = []
    ctx_list.append(context.current_context())
    def f():
        set_default_context(mx.gpu(11))
        ctx_list.append(context.current_context())
    thread = threading.Thread(target=f)
    thread.start()
    thread.join()
    assert Context.devtype2str[ctx_list[0].device_typeid] == "cpu"
    assert ctx_list[0].device_id == 0
    assert Context.devtype2str[ctx_list[1].device_typeid] == "gpu"
    assert ctx_list[1].device_id == 11

    e1 = threading.Event()
    e2 = threading.Event()
    status = [False]
    def g():
        with mx.cpu(10):
            e2.set()
            e1.wait()
            if context.current_context().device_id == 10:
                status[0] = True
    thread = threading.Thread(target=g)
    thread.start()
    e2.wait()
    with Context("cpu", 11):
        e1.set()
        thread.join()
    e1.clear()
    e2.clear()
    assert status[0], "Spawned thread didn't set the correct context"

def test_attrscope():
    attrscope_list = []
    with AttrScope(y="hi", z="hey") as attrscope:
        attrscope_list.append(attrscope)

        def f():
            with AttrScope(x="hello") as attrscope:
                attrscope_list.append(attrscope)

        thread = threading.Thread(target=f)
        thread.start()
        thread.join()
        assert len(attrscope_list[0]._attr) == 2
        assert attrscope_list[1]._attr["x"] == "hello"

    e1 = threading.Event()
    e2 = threading.Event()
    status = [False]
    def g():
        with mx.AttrScope(x="hello"):
            e2.set()
            e1.wait()
            if "hello" in mx.attribute.current()._attr.values():
                status[0] = True
    thread = threading.Thread(target=g)
    thread.start()
    e2.wait()
    with AttrScope(x="hi"):
        e1.set()
        thread.join()
    e1.clear()
    e2.clear()
    assert status[0], "Spawned thread didn't set the correct attr key values"

def test_name():
    name_list = []
    name_manager = mx.name.current()
    name_manager.get(None, "main_thread")
    name_list.append(name_manager)
    def f():
        with mx.name.NameManager():
            name_manager = mx.name.current()
            name_manager.get(None, "spawned_thread")
            name_list.append(name_manager)
    thread = threading.Thread(target=f)
    thread.start()
    thread.join()
    assert "main_thread" in name_list[0]._counter, "cannot find the string `main thread` in name_list[0]._counter"
    assert "spawned_thread" in name_list[1]._counter, "cannot find the string `spawned thread` in name_list[1]._counter"

    e1 = threading.Event()
    e2 = threading.Event()
    status = [False]
    def g():
        with mx.name.NameManager():
            e2.set()
            e1.wait()
            if "main_thread" not in mx.name.current()._counter:
                status[0] = True
    thread = threading.Thread(target=g)
    thread.start()
    e2.wait()
    with mx.name.NameManager():
        mx.name.current().get(None, "main_thread")
        e1.set()
        thread.join()
    e1.clear()
    e2.clear()
    assert status[0], "Spawned thread isn't using thread local NameManager"

def test_blockscope():
    class dummy_block:
        pass
    blockscope_list = []
    status = [False]
    event = threading.Event()
    def f():
        net = dummy_block()  # BlockScope only keeps a weakref to the Block
        with mx.gluon.block._block_scope(net):
            x = mx.name.current().get(None, "hello")
            event.wait()
            if x == "dummy_block_hello0":
                status[0] = True
    thread = threading.Thread(target=f)
    thread.start()
    event.set()
    thread.join()
    event.clear()
    assert status[0], "Spawned thread isn't using the correct blockscope namemanager"

def test_createblock():
    status = [False]
    def f():
        net = mx.gluon.nn.Dense(2)
        net.initialize()
        x = net(mx.np.array([1, 2, 3]))
        x.wait_to_read()
        status[0] = True

    thread = threading.Thread(target=f)
    thread.start()
    thread.join()
    assert status[0], "Failed to create a layer within a thread"

def test_symbol():
    status = [False]
    def f():
        a = mx.sym.var("a")
        b = mx.sym.var("b")
        a_ = mx.nd.ones((2, 2))
        c_ = a_.copy()
        func1 = (a + b)._bind(mx.cpu(), args={'a': a_, 'b': c_})
        func1.forward()[0].wait_to_read()
        status[0] = True
    thread = threading.Thread(target=f)
    thread.start()
    thread.join()
    assert status[0], "Failed to execute a symbolic graph within a thread"


def test_np_array_scope():
    np_array_scope_list = []
    _NumpyArrayScope._current = _NumpyArrayScope(False)
    np_array_scope_list.append(_NumpyArrayScope._current)

    def f():
        _NumpyArrayScope._current = _NumpyArrayScope(True)
        np_array_scope_list.append(_NumpyArrayScope._current)

    thread = threading.Thread(target=f)
    thread.start()
    thread.join()
    assert len(np_array_scope_list) == 2
    assert not np_array_scope_list[0]._is_np_array
    assert np_array_scope_list[1]._is_np_array

    event = threading.Event()
    status = [False]

    def g():
        with mx.np_array(False):
            event.wait()
            if not mx.is_np_array():
                status[0] = True

    thread = threading.Thread(target=g)
    thread.start()
    _NumpyArrayScope._current = _NumpyArrayScope(True)
    event.set()
    thread.join()
    event.clear()
    assert status[0], "Spawned thread didn't set status correctly"


def test_np_global_shape():
    prev_np_shape = set_np_shape(2)
    data = []

    def f():
        # scalar
        data.append(mx.np.ones(shape=()))
        # zero-dim
        data.append(mx.np.ones(shape=(0, 1, 2)))
    try:
        thread = threading.Thread(target=f)
        thread.start()
        thread.join()

        assert_almost_equal(data[0].asnumpy(), np.ones(shape=()))
        assert_almost_equal(data[1].asnumpy(), np.ones(shape=(0, 1, 2)))
    finally:
        set_np_shape(prev_np_shape)
