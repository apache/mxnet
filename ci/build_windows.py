#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

"""User friendly / multi platform builder script"""

import argparse
import datetime
import glob
import logging
import os
import platform
import shutil
import sys
import time
from distutils.dir_util import copy_tree
from enum import Enum
from subprocess import check_call

from util import *

KNOWN_VCVARS = {
    'VS 2015': r'C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\x86_amd64\vcvarsx86_amd64.bat',
    'VS 2017': r'C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsx86_amd64.bat'
}


class BuildFlavour(Enum):
    WIN_CPU = 'WIN_CPU'
    WIN_CPU_MKLDNN = 'WIN_CPU_MKLDNN'
    WIN_CPU_MKLDNN_MKL = 'WIN_CPU_MKLDNN_MKL'
    WIN_CPU_MKL = 'WIN_CPU_MKL'
    WIN_GPU = 'WIN_GPU'
    WIN_GPU_MKLDNN = 'WIN_GPU_MKLDNN'


CMAKE_FLAGS = {
    'WIN_CPU': ('-DUSE_CUDA=0 '
                '-DUSE_CUDNN=0 '
                '-DUSE_NVRTC=0 '
                '-DUSE_OPENCV=1 '
                '-DUSE_OPENMP=1 '
                '-DUSE_PROFILER=1 '
                '-DUSE_BLAS=open '
                '-DUSE_LAPACK=1 '
                '-DUSE_DIST_KVSTORE=0 '
                '-DBUILD_CPP_EXAMPLES=1 '
                '-DUSE_MKL_IF_AVAILABLE=0 '
                '-DCMAKE_BUILD_TYPE=Release')

    , 'WIN_CPU_MKLDNN': ('-DUSE_CUDA=0 '
                         '-DUSE_CUDNN=0 '
                         '-DUSE_NVRTC=0 '
                         '-DUSE_OPENCV=1 '
                         '-DUSE_OPENMP=1 '
                         '-DUSE_PROFILER=1 '
                         '-DUSE_BLAS=open '
                         '-DUSE_LAPACK=1 '
                         '-DUSE_DIST_KVSTORE=0 '
                         '-DUSE_MKL_IF_AVAILABLE=1 '
                         '-DUSE_MKLDNN=1 '
                         '-DCMAKE_BUILD_TYPE=Release')

    , 'WIN_CPU_MKLDNN_MKL': ('-DUSE_CUDA=0 '
                         '-DUSE_CUDNN=0 '
                         '-DUSE_NVRTC=0 '
                         '-DUSE_OPENCV=1 '
                         '-DUSE_OPENMP=1 '
                         '-DUSE_PROFILER=1 '
                         '-DUSE_BLAS=mkl '
                         '-DUSE_LAPACK=1 '
                         '-DUSE_DIST_KVSTORE=0 '
                         '-DUSE_MKL_IF_AVAILABLE=1 '
                         '-DUSE_MKLDNN=1 '
                         '-DCMAKE_BUILD_TYPE=Release')

    , 'WIN_CPU_MKL': ('-DUSE_CUDA=0 '
                         '-DUSE_CUDNN=0 '
                         '-DUSE_NVRTC=0 '
                         '-DUSE_OPENCV=1 '
                         '-DUSE_OPENMP=1 '
                         '-DUSE_PROFILER=1 '
                         '-DUSE_BLAS=mkl '
                         '-DUSE_LAPACK=1 '
                         '-DUSE_DIST_KVSTORE=0 '
                         '-DUSE_MKL_IF_AVAILABLE=1 '
                         '-DUSE_MKLDNN=0 '
                         '-DCMAKE_BUILD_TYPE=Release')
    , 'WIN_GPU': ('-DUSE_CUDA=1 '
                  '-DUSE_CUDNN=1 '
                  '-DUSE_NVRTC=1 '
                  '-DUSE_OPENCV=1  '
                  '-DUSE_OPENMP=1 '
                  '-DUSE_PROFILER=1 '
                  '-DUSE_BLAS=open '
                  '-DUSE_LAPACK=1 '
                  '-DUSE_DIST_KVSTORE=0 '
                  '-DCUDA_ARCH_NAME=Manual '
                  '-DCUDA_ARCH_BIN=52 '
                  '-DCUDA_ARCH_PTX=52 '
                  '-DCMAKE_CXX_FLAGS="/FS /MD /O2 /Ob2" '
                  '-DUSE_MKL_IF_AVAILABLE=0 '
                  '-DCMAKE_BUILD_TYPE=Release')

    , 'WIN_GPU_MKLDNN': ('-DUSE_CUDA=1 '
                         '-DUSE_CUDNN=1 '
                         '-DUSE_NVRTC=1 '
                         '-DUSE_OPENCV=1 '
                         '-DUSE_OPENMP=1 '
                         '-DUSE_PROFILER=1 '
                         '-DUSE_BLAS=open '
                         '-DUSE_LAPACK=1 '
                         '-DUSE_DIST_KVSTORE=0 '
                         '-DCUDA_ARCH_NAME=Manual '
                         '-DCUDA_ARCH_BIN=52 '
                         '-DCUDA_ARCH_PTX=52 '
                         '-DUSE_MKLDNN=1 '
                         '-DCMAKE_CXX_FLAGS="/FS /MD /O2 /Ob2" '
                         '-DCMAKE_BUILD_TYPE=Release')

}


def windows_build(args):
    logging.info("Using vcvars environment:\n{}".format(args.vcvars))

    path = args.output
    os.makedirs(path, exist_ok=True)

    mxnet_root = get_mxnet_root()
    logging.info("Found MXNet root: {}".format(mxnet_root))

    with remember_cwd():
        os.chdir(path)
        cmd = "\"{}\" && cmake -G \"NMake Makefiles JOM\" {} {}".format(args.vcvars,
                                                                        CMAKE_FLAGS[args.flavour],
                                                                        mxnet_root)
        logging.info("Generating project with CMake:\n{}".format(cmd))
        check_call(cmd, shell=True)

        cmd = "\"{}\" && jom".format(args.vcvars)
        logging.info("Building with jom:\n{}".format(cmd))

        t0 = int(time.time())
        check_call(cmd, shell=True)

        logging.info("Build flavour: {} complete in directory: \"{}\"".format(args.flavour, os.path.abspath(path)))
        logging.info("Build took {}".format(datetime.timedelta(seconds=int(time.time() - t0))))
    windows_package(args)


def windows_package(args):
    pkgfile = 'windows_package.7z'
    pkgdir = os.path.abspath('windows_package')
    logging.info("Packaging libraries and headers in package: %s", pkgfile)
    j = os.path.join
    pkgdir_lib = os.path.abspath(j(pkgdir, 'lib'))
    with remember_cwd():
        os.chdir(args.output)
        logging.info("Looking for static libraries and dlls in: \"%s", os.getcwd())
        libs = list(glob.iglob('**/*.lib', recursive=True))
        dlls = list(glob.iglob('**/*.dll', recursive=True))
        os.makedirs(pkgdir_lib, exist_ok=True)
        for lib in libs:
            logging.info("packing lib: %s", lib)
            shutil.copy(lib, pkgdir_lib)
        for dll in dlls:
            logging.info("packing dll: %s", dll)
            shutil.copy(dll, pkgdir_lib)
        os.chdir(get_mxnet_root())
        logging.info('packing python bindings')
        copy_tree('python', j(pkgdir, 'python'))
        logging.info('packing headers')
        copy_tree('include', j(pkgdir, 'include'))
        logging.info("Compressing package: %s", pkgfile)
        check_call(['7z', 'a', pkgfile, pkgdir])


def nix_build(args):
    path = args.output
    os.makedirs(path, exist_ok=True)
    with remember_cwd():
        os.chdir(path)
        logging.info("Generating project with CMake")
        check_call("cmake \
            -DUSE_CUDA=OFF \
            -DUSE_OPENCV=OFF \
            -DUSE_OPENMP=OFF \
            -DCMAKE_BUILD_TYPE=Debug \
            -GNinja ..", shell=True)
        check_call("ninja", shell=True)


def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)-15s %(message)s')
    logging.info("MXNet Windows build helper")

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output",
        help="output directory",
        default='build',
        type=str)

    parser.add_argument("--vcvars",
        help="vcvars batch file location, typically inside vs studio install dir",
        default=KNOWN_VCVARS['VS 2015'],
        type=str)

    parser.add_argument("--arch",
        help="architecture",
        default='x64',
        type=str)

    parser.add_argument("-f", "--flavour",
        help="build flavour",
        default='WIN_CPU',
        choices=[x.name for x in BuildFlavour],
        type=str)

    args = parser.parse_args()
    logging.info("Build flavour: %s", args.flavour)

    system = platform.system()
    if system == 'Windows':
        logging.info("Detected Windows platform")
        if 'OpenBLAS_HOME' not in os.environ:
            os.environ["OpenBLAS_HOME"] = "C:\\Program Files\\OpenBLAS-v0.2.19"
        if 'OpenCV_DIR' not in os.environ:
            os.environ["OpenCV_DIR"] = "C:\\Program Files\\OpenCV-v3.4.1\\build"
        if 'CUDA_PATH' not in os.environ:
            os.environ["CUDA_PATH"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.2"
        if 'MKL_ROOT' not in os.environ:
            os.environ["MKL_ROOT"] = "C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries\\windows\\mkl"
        windows_build(args)

    elif system == 'Linux' or system == 'Darwin':
        nix_build(args)

    else:
        logging.error("Don't know how to build for {} yet".format(platform.system()))

    return 0


if __name__ == '__main__':
    sys.exit(main())

