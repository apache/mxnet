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
import mxnet as mx
from mxnet import context, attribute, name
from mxnet.gluon import block
from mxnet.context import Context
from mxnet.attribute import AttrScope
from mxnet.name import NameManager
from mxnet.test_utils import set_default_context

def test_context():
    ctx_list = []
    ctx_list.append(Context.default_ctx)
    def f():
        set_default_context(mx.gpu(11))
        ctx_list.append(Context.default_ctx)
    thread = threading.Thread(target=f)
    thread.start()
    thread.join()
    assert Context.devtype2str[ctx_list[0].device_typeid] == "cpu"
    assert ctx_list[0].device_id == 0
    assert Context.devtype2str[ctx_list[1].device_typeid] == "gpu"
    assert ctx_list[1].device_id == 11

    condition = threading.Condition()
    status = [False]
    def g():
        condition.acquire()
        with mx.cpu(10):
            condition.wait()
            if Context.default_ctx.device_id == 10:
                status[0] = True
        condition.release()
    thread = threading.Thread(target=g)
    thread.start()
    condition.acquire()
    Context.default_ctx = Context("cpu", 11)
    condition.notify()
    condition.release()
    thread.join()
    assert status[0], "Spawned thread didn't set the correct context"

def test_attrscope():
    attrscope_list = []
    AttrScope.current = AttrScope(y="hi", z="hey")
    attrscope_list.append(AttrScope.current)
    def f():
        AttrScope.current = AttrScope(x="hello")
        attrscope_list.append(AttrScope.current)
    thread = threading.Thread(target=f)
    thread.start()
    thread.join()
    assert len(attrscope_list[0]._attr) == 2
    assert attrscope_list[1]._attr["x"] == "hello"

    condition = threading.Condition()
    status = [False]
    def g():
        condition.acquire()
        with mx.AttrScope(x="hello"):
            condition.wait()
            if "hello" in AttrScope.current._attr.values():
                status[0] = True
    thread = threading.Thread(target=g)
    thread.start()
    condition.acquire()
    AttrScope.current = AttrScope(x="hi")
    condition.notify()
    condition.release()
    thread.join()
    assert status[0], "Spawned thread didn't set the correct attr key values"

def test_name():
    name_list = []
    NameManager.current = NameManager()
    NameManager.current.get(None, "main_thread")
    name_list.append(NameManager.current)
    def f():
        NameManager.current = NameManager()
        NameManager.current.get(None, "spawned_thread")
        name_list.append(NameManager.current)
    thread = threading.Thread(target=f)
    thread.start()
    thread.join()
    assert "main_thread" in name_list[0]._counter, "cannot find the string `main thread` in name_list[0]._counter"
    assert "spawned_thread" in name_list[1]._counter, "cannot find the string `spawned thread` in name_list[1]._counter"

    condition = threading.Condition()
    status = [False]
    def g():
        condition.acquire()
        with NameManager():
            condition.wait()
            if "main_thread" not in NameManager.current._counter:
                status[0] = True
    thread = threading.Thread(target=g)
    thread.start()
    condition.acquire()
    NameManager.current = NameManager()
    NameManager.current.get(None, "main_thread")
    condition.notify()
    condition.release()
    thread.join()
    assert status[0], "Spawned thread isn't using thread local NameManager"

def test_blockscope():
    class dummy_block(object):
        def __init__(self, prefix):
            self.prefix = prefix
            self._empty_prefix = False
    blockscope_list = []
    status = [False]
    condition = threading.Condition()
    def f():
        with block._BlockScope(dummy_block("spawned_")):
            x= NameManager.current.get(None, "hello")
            if x == "spawned_hello0":
                status[0] = True
    thread = threading.Thread(target=f)
    thread.start()
    condition.acquire()
    block._BlockScope.create("main_thread", None, "hi")
    condition.notify()
    condition.release()
    thread.join()
    assert status[0], "Spawned thread isn't using the correct blockscope namemanager"

if __name__ == '__main__':
    import nose
    nose.runmodule()
