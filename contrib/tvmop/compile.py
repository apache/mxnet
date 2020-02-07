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
"""TVM Operator compile entry point"""
import tvm
from tvm import autotvm

import os
import argparse
import re
import json
import logging
import sys
import subprocess
from tvmop.opdef import __OP_DEF__
from tvmop.space import ConfigSpaces, ConfigSpace
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
from util import tempdir
import ctypes

logging.basicConfig(level=logging.INFO)


def create_shared(output,
                  objects,
                  options=None,
                  cc="g++"):
    """Create shared library.
    Parameters
    ----------
    output : str
        The target shared library.
    objects : List[str]
        List of object files.
    options : List[str]
        The list of additional options string.
    cc : Optional[str]
        The compiler command.
    """
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        _linux_compile(output, objects, options, cc)
    elif sys.platform == "win32":
        _windows_shared(output, objects, options)
    else:
        raise ValueError("Unsupported platform")


def _linux_compile(output, objects, options, compile_cmd="g++"):
    cmd = [compile_cmd]
    if output.endswith(".so") or output.endswith(".dylib"):
        cmd += ["-shared", "-fPIC"]
        if sys.platform == "darwin":
            cmd += ["-undefined", "dynamic_lookup"]
    elif output.endswith(".obj"):
        cmd += ["-c"]
    cmd += ["-o", output]
    if isinstance(objects, str):
        cmd += [objects]
    else:
        cmd += objects
    if options:
        cmd += options
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += str(out)
        raise RuntimeError(msg)


def _windows_shared(output, objects, options):
    encoding = 'cp' + str(ctypes.cdll.kernel32.GetACP())
    py_str = lambda x: x.decode(encoding)

    cl_cmd = ["cl"]
    cl_cmd += ["-c"]
    if isinstance(objects, str):
        objects = [objects]
    cl_cmd += objects

    temp = tempdir()
    dllmain_path = temp.relpath("dllmain.cc")
    with open(dllmain_path, "w") as dllmain_obj:
        dllmain_obj.write('#include <windows.h>\
BOOL APIENTRY DllMain( HMODULE hModule,\
                       DWORD  ul_reason_for_call,\
                       LPVOID lpReserved)\
{return TRUE;}')

    cl_cmd += [dllmain_path]

    temp_path = dllmain_path.replace("dllmain.cc", "")
    cl_cmd += ["-Fo:" + temp_path]
    try:
        proc = subprocess.Popen(
            cl_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out, _) = proc.communicate()
    except FileNotFoundError:
        raise RuntimeError("Can not find cl.exe,"
                           "please run this in Vistual Studio Command Prompt.")
    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)
    link_cmd = ["lld-link"]
    link_cmd += ["-dll", "-FORCE:MULTIPLE"]

    for obj in objects:
        if obj.endswith(".cc"):
            (_, temp_file_name) = os.path.split(obj)
            (shot_name, _) = os.path.splitext(temp_file_name)
            link_cmd += [os.path.join(temp_path, shot_name + ".obj")]
        if obj.endswith(".o"):
            link_cmd += [obj]
        if options:
            link_cmd += options

    link_cmd += ["-EXPORT:__tvm_main__"]
    link_cmd += [temp_path + "dllmain.obj"]
    link_cmd += ["-out:" + output]

    try:
        proc = subprocess.Popen(
            link_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out, _) = proc.communicate()
    except FileNotFoundError:
        raise RuntimeError("Can not find the LLVM linker for Windows (lld-link.exe)."
                           "Make sure it's installed"
                           " and the installation directory is in the %PATH% environment "
                           "variable. Prebuilt binaries can be found at: https://llvm.org/"
                           "For building the linker on your own see: https://lld.llvm.org/#build")
    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += py_str(out)

        raise RuntimeError(msg)


def get_target(device):
    if device == "cpu":
        return "llvm"
    elif device == "cuda" or device == "gpu":
        return "cuda"
    assert False, "Unknown device " + device


def get_cuda_arch(arch):
    if arch is None:
        return None

    if not isinstance(arch, str):
        raise TypeError('Expecting parameter arch as a str, while got a {}'.format(str(type(arch))))

    if len(arch) == 0:
        return None

    # an example of arch string,
    # -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
    # -gencode;arch=compute_75,code=[sm_75,compute_75] --fatbin-options -compress-all
    archs = []
    flags = arch.replace("-gencode;", "-gencode ").split()
    for flag in flags:
        if flag.startswith('-gencode') or flag.startswith('arch='):
            archs.append(flag)

    return archs


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(sys.path[0]))
    parser = argparse.ArgumentParser(description="Generate tvm operators")
    parser.add_argument("-o", action="store", required=True, dest="target_path",
                        help="Target path which stores compiled library")
    parser.add_argument("-L", action="store", default=None, dest="ld_path",
                        help="library link path")
    parser.add_argument('--cuda-arch', type=str, default=None, dest='cuda_arch',
                        help='The cuda arch for compiling kernels for')
    parser.add_argument("--config", action="store", required=True, dest="config_path",
                        help="Path which stores the config file")
    arguments = parser.parse_args()

    func_list_llvm = []
    func_list_cuda = []

    # TODO: attach instruction features to the library, e.g., avx-512, etc.
    for operator_def in __OP_DEF__:
        for sch, args, name in operator_def.invoke_all():
            name = operator_def.get_op_name(name, args)
            if tvm.module.enabled(get_target(operator_def.target)):
                func_list = func_list_llvm if operator_def.target == "cpu" else func_list_cuda
                func_lower = tvm.lower(sch, args,
                                       name=name,
                                       binds=operator_def.get_binds(args))
                func_list.append(func_lower)

    lowered_funcs = {get_target("cpu"): func_list_llvm}
    if len(func_list_cuda) > 0:
        lowered_funcs[get_target("cuda")] = func_list_cuda
        cuda_arch = get_cuda_arch(arguments.cuda_arch)
        if cuda_arch is None:
            logging.info('No cuda arch specified. TVM will try to detect it from the build platform.')
        else:
            logging.info('Cuda arch {} set for compiling TVM operator kernels.'.format(cuda_arch))
            set_cuda_target_arch(cuda_arch)
    func_binary = tvm.build(lowered_funcs, name="tvmop")
    # we create libtvmop.o first, which gives us chance to link tvm_runtime together with the libtvmop
    # to allow mxnet find external helper functions in libtvm_runtime
    func_binary.save(arguments.target_path + "/libtvmop.o")
    ld_path = arguments.target_path if arguments.ld_path is None else arguments.ld_path
    out_name = "/libtvmop.so" if sys.platform != 'win32' else "/libtvmop.dll"
    options = ["-L", ld_path, "-ltvm_runtime"] if sys.platform != 'win32' else [ld_path + "/tvm_runtime.lib"]
    create_shared(arguments.target_path + out_name,
                  arguments.target_path + "/libtvmop.o",
                  options=options)

    config_spaces = ConfigSpaces()
    for operator_def in __OP_DEF__:
        for config_space, name in operator_def.get_config_spaces():
            config_spaces[name] = ConfigSpace.from_tvm(config_space)
    with open(arguments.config_path, "w") as f:
        json.dump(config_spaces.to_json_dict(), f)