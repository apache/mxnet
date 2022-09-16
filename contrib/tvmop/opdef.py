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

# coding: utf-8
import tvm
import inspect
from tvm import autotvm
from itertools import product

__OP_DEF__ = []

class OpDef:
    """Specify the properties of an operator and
    construct the value combination of the arguments
    e.g., ldtype=["float32", "int32"], rdtype=["float16", "int16"],
    then the argument combination is
    [
        {"ldtype": "float32", "rdtype": "float16"},
        {"ldtype": "float32", "rdtype": "int16"},
        {"ldtype": "int32", "rdtype": "float16"},
        {"ldtype": "int32", "rdtype": "int16"},
    ]

    Parameters
    ----------
    func : function
         The function to define the operator (in tvm compute and schedule).
         It will get the argument combination extracted by this class.
    name : str
         function name.
    target : str
         {"cpu", "gpu", "cuda"}
    auto_broadcast : bool
         auto_broadcast=True allows one to implement broadcast computation
         without considering whether dimension size equals to one.
         TVM maps buffer[i][j][k] -> buffer[i][0][k] if dimension i's shape equals 1.
    """
    def __init__(self, func, name, target, auto_broadcast, **kwargs):
        # construct the value combination of the arguments
        # e.g., ldtype=["float32", "int32"], rdtype=["float16", "int16"]
        # arg_combination = [
        #   {"ldtype": "float32", "rdtype": "float16"},
        #   {"ldtype": "float32", "rdtype": "int16"},
        #   {"ldtype": "int32", "rdtype": "float16"},
        #   {"ldtype": "int32", "rdtype": "int16"},
        # ]
        self.attrs = kwargs.pop('attrs', [])
        self.attrs_valid = kwargs.pop('attrs_valid', lambda **kwargs: True)
        args = [k for k in kwargs]
        values = [kwargs[k] if isinstance(kwargs[k], (list, tuple)) else [kwargs[k]]
                  for k in args]
        cart_product = product(*values)
        self.arg_combination = [{k: v for k, v in zip(args, comb_values)}
                                for comb_values in cart_product]
        self.func = func
        self.name = name
        self.target = target
        self.auto_broadcast = auto_broadcast
        self.dispatchable = 'fallback' in inspect.signature(self.func).parameters

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def invoke_all(self):
        for each_kwargs in self.arg_combination:
            if self.attrs_valid(**each_kwargs):
                name = self.name \
                    + ''.join(["{}_{}".format(key, each_kwargs[key]) for key in self.attrs])
                if self.dispatchable is False:
                    sch, args = self.func(**each_kwargs)
                    yield sch, args, name
                else:
                    # register dispatch schedules
                    config_space = autotvm.ConfigSpace()
                    with autotvm.task.ApplyConfig(config_space):
                            sch, args = self.func(fallback=False, **each_kwargs)
                    for i in range(len(config_space)):
                        config_entity = config_space.get(i)
                        with autotvm.task.ApplyConfig(config_entity):
                            sch, args = self.func(fallback=False, **each_kwargs)
                        subname = name + "index_" + str(i)
                        yield sch, args, subname
                    # register fallback schedule
                    config_space = autotvm.ConfigSpace()
                    with autotvm.task.ApplyConfig(config_space):
                            sch, args = self.func(fallback=True, **each_kwargs)
                    subname = name + "fallback"
                    yield sch, args, subname

    def get_op_name(self, name, args):
        return name + ''.join([f"{arg.dtype}_{len(arg.shape)}" for arg in args if hasattr(arg, 'shape')])

    def get_config_spaces(self):
        for each_kwargs in self.arg_combination:
            if self.attrs_valid(**each_kwargs) and self.dispatchable is True:
                name = self.name \
                    + ''.join(["{}_{}".format(key, each_kwargs[key]) for key in self.attrs])
                config_space = autotvm.ConfigSpace()
                with autotvm.task.ApplyConfig(config_space):
                    self.func(fallback=False, **each_kwargs)
                yield config_space, name

    def get_binds(self, args):
        if self.auto_broadcast:
            return {arg: tvm.tir.decl_buffer(arg.shape, arg.dtype, buffer_type="auto_broadcast")
                    for arg in args}
        return None


def defop(name, target=None, auto_broadcast=False, **kwargs):
    """Decorator to define a tvm operator.
    Parameters
    ----------
    name : str
        function name
    target : str
        {"cpu", "gpu", "cuda"}
    auto_broadcast : bool
        auto_broadcast=True allows one to implement broadcast computation
        without considering whether dimension size equals to one.
        TVM maps buffer[i][j][k] -> buffer[i][0][k] if dimension i's shape equals 1.
    Returns
    -------
    fdef : function
        A wrapped operator definition function, which returns (schedule, [tensors])
    """
    assert name is not None and len(name) > 0
    target = "cpu" if target is None else target

    def _defop(func):
        opdef = OpDef(func, name, target, auto_broadcast, **kwargs)
        __OP_DEF__.append(opdef)
        return opdef
    return _defop

