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
import timeit
import itertools
import argparse
import os

class OpArgMngr(object):
    """Operator argument manager for storing operator workloads."""
    args = {}

    @staticmethod
    def add_workload(funcname, *args, **kwargs):
        if "_specifier" not in kwargs:
            _specifier = funcname
        else:
            _specifier = kwargs["_specififer"]
            del kwargs["_specififer"]
        if _specifier in OpArgMngr.args:
            raise ValueError("duplicate {}".format(_specifier))
        OpArgMngr.args[_specifier] = {'args': args, 'kwargs': kwargs, 'funcname': funcname}


def generate_workloads():
    array_pool = {}
    shapes = []
    for ndim in range(4):
        shapes.extend(list(itertools.product(range(4), repeat=ndim)))
    for shape in shapes:
        name = 'x'.join(str(i) for i in shape)
        if name in array_pool:
            raise ValueError("duplicate array {}".format(name))
        array_pool[name] = dnp.ones(shape)
    return array_pool


def prepare_workloads():
    pool = generate_workloads()
    OpArgMngr.add_workload("zeros", (2, 2))
    OpArgMngr.add_workload("tensordot", pool['2x2'], pool['2x2'], ((1, 0), (0, 1)))
    OpArgMngr.add_workload("cumsum", pool['3x2'], axis=0, out=pool['3x2'])
    OpArgMngr.add_workload("add", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("random.uniform", low=0, high=1, size=1)
    OpArgMngr.add_workload("random.choice", pool['2'], size=(2, 2))
    OpArgMngr.add_workload("take", pool['2'], dnp.array([1,0], dtype='int64'))


def benchmark_helper(f, *args, **kwargs):
    number = 10000
    return timeit.timeit(lambda: f(*args, **kwargs), number=number) / number


def get_op(module, funcname):
    funcname = funcname.split(".")
    for fname in funcname:
        module = getattr(module, fname)
    return module


def run_benchmark(packages):
    results = {}
    for (k, v) in OpArgMngr.args.items():
        result = {}
        for (name, package) in packages.items():
            print('{}.{} running...'.format(name, k))
            op = get_op(package["module"], v["funcname"])
            args = [package["data"](arg) for arg in v["args"]]
            kwargs = {k: package["data"](v) for (k, v) in v["kwargs"].items()}
            benchmark = benchmark_helper(op, *args, **kwargs)
            result[name] = benchmark
        results[k] = result
    return results


def show_results(results):
    print("{:>24}{:>24}{:>24}".format("name", "package", "time(us)"))
    for (specifier, d) in results.items():
        for (k, v) in d.items():
            print("{:>24}{:>24}{:>24}".format(specifier, k, v * 10 ** 6))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ffi_type')
    parsed = parser.parse_args()
    if parsed.ffi_type == "cython":
        os.environ['MXNET_ENABLE_CYTHON'] = '1'
        os.environ['MXNET_ENFORCE_CYTHON'] = '1'
    elif parsed.ffi_type == "ctypes":
        os.environ['MXNET_ENABLE_CYTHON'] = '0'
    else:
        raise ValueError("unknown ffi_type {}",format(parsed.ffi_type))
    os.environ["MXNET_ENGINE_TYPE"] = "NaiveEngine"
    import mxnet as mx
    import numpy as onp
    from mxnet import np as dnp

    mx.npx.set_np()
    packages = {
        "onp": {
            "module": onp,
            "data": lambda arr: arr.asnumpy() if isinstance(arr, dnp.ndarray) else arr
        },
        "dnp": {
            "module": dnp,
            "data": lambda arr: arr
        }
    }
    prepare_workloads()
    results = run_benchmark(packages)
    show_results(results)
