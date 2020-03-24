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
    OpArgMngr.add_workload("polyval", dnp.arange(10), pool['2x2'])
    OpArgMngr.add_workload("ediff1d", pool['2x2'], pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("nan_to_num", pool['2x2'])
    OpArgMngr.add_workload("tensordot", pool['2x2'], pool['2x2'], ((1, 0), (0, 1)))
    OpArgMngr.add_workload("kron", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("cumsum", pool['3x2'], axis=0, out=pool['3x2'])
    OpArgMngr.add_workload("add", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("linalg.svd", pool['3x3'])
    OpArgMngr.add_workload("split", pool['3x3'], (0, 1, 2), axis=1)
    OpArgMngr.add_workload("argmax", pool['3x2'], axis=-1)
    OpArgMngr.add_workload("argmin", pool['3x2'], axis=-1)
    OpArgMngr.add_workload("indices", dimensions=(1, 2, 3))
    OpArgMngr.add_workload("subtract", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("multiply", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("mod", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("remainder", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("divide", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("true_divide", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("power", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("lcm", pool['2x2'].astype('int32'), pool['2x2'].astype('int32'))
    OpArgMngr.add_workload("diff", pool['2x2'], n=1, axis=-1)
    OpArgMngr.add_workload("nonzero", pool['2x2'])
    OpArgMngr.add_workload("tril", pool['2x2'], k=0)
    OpArgMngr.add_workload("expand_dims", pool['2x2'], axis=0)
    OpArgMngr.add_workload("broadcast_to", pool['2x2'], (2, 2, 2))
    OpArgMngr.add_workload("full_like", pool['2x2'], 2)
    OpArgMngr.add_workload("zeros_like", pool['2x2'])
    OpArgMngr.add_workload("ones_like", pool['2x2'])
    OpArgMngr.add_workload("random.uniform", low=0, high=1, size=1)
    OpArgMngr.add_workload("where", pool['2x3'], pool['2x3'], pool['2x1'])
    OpArgMngr.add_workload("fmax", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("fmin", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("fmod", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("may_share_memory", pool['2x3'][:0], pool['2x3'][:1])
    OpArgMngr.add_workload("diag", pool['2x2'], k=1)
    OpArgMngr.add_workload("diagonal", pool['2x2x2'], offset=-1, axis1=0, axis2=1)
    OpArgMngr.add_workload("diag_indices_from", pool['2x2'])
    OpArgMngr.add_workload("bincount", dnp.arange(3, dtype=int), pool['3'], minlength=4)
    OpArgMngr.add_workload("percentile", pool['2x2x2'], 80, axis=0, out=pool['2x2'],\
                           interpolation='midpoint')
    OpArgMngr.add_workload("quantile", pool['2x2x2'], 0.8, axis=0, out=pool['2x2'],\
                           interpolation='midpoint')
    OpArgMngr.add_workload("all", pool['2x2x2'], axis=(0, 1),\
                           out=dnp.array([False, False], dtype=bool), keepdims=False)
    OpArgMngr.add_workload("any", pool['2x2x2'], axis=(0, 1),\
                           out=dnp.array([False, False], dtype=bool), keepdims=False)
    OpArgMngr.add_workload("roll", pool["2x2"], 1, axis=0)
    OpArgMngr.add_workload("rot90", pool["2x2"], 2)


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
