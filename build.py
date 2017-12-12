#!/usr/bin/env python
# -*- coding: utf-8 -*-

import contextlib
import subprocess
import logging
import os
import tempfile
import sys
from distutils import spawn
import logging
from subprocess import check_call
import platform
import argparse

@contextlib.contextmanager
def remember_cwd():
    '''
    Restore current directory when exiting context
    '''
    curdir = os.getcwd()
    try: yield
    finally: os.chdir(curdir)


class CmdResult(object):
    def __init__(self, cmd, std_out, std_err, status_code):
        self.std_out = std_out
        self.std_err = std_err
        self.status_code = status_code if status_code is not None else 0
        self.cmd = cmd

    def __str__(self):
        return "Command: \"{}\" status: {}\nstdout:\n{}\nstderr:\n{}".format(
            self.cmd, self.status_code,
            self.std_out[:50], self.std_err[:20])


def run(cmd, fail_on_error=True):
    logging.debug("executing shell command:\n" + cmd)
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    std_out, std_err = proc.communicate()
    if fail_on_error:
        if proc.returncode != 0:
            logging.warn('Error running command: {}'.format(cmd))
        assert proc.returncode == 0, std_err
    res = CmdResult(cmd, std_out.decode('utf-8'), std_err.decode('utf-8'), proc.returncode)
    return res

def xmkdir(d):
    rev_path_list = list()
    head = d
    while len(head) and head != os.sep:
        rev_path_list.append(head)
        (head, tail) = os.path.split(head)

    rev_path_list.reverse()
    for p in rev_path_list:
        try:
            os.mkdir(p)
        except OSError as e:
            if e.errno != 17:
                raise



def windows_build(args):
    BUILD_BAT=r'''call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\x86_amd64\vcvarsx86_amd64.bat"
msbuild mxnet.sln /p:Configuration=Release;Platform=x64
'''

    os.environ["OpenBLAS_HOME"] = "C:\\mxnet\\openblas"
    os.environ["OpenCV_DIR"] = "C:\\mxnet\\opencv_vc14"
    os.environ["CUDA_PATH"] = "C:\\CUDA\\v8.0"

    path = args.output
    xmkdir(path)
    with remember_cwd():
        os.chdir(path)

        logging.info("Generating project with CMake")
        check_call("cmake -G \"Visual Studio 14 2015 Win64\" \
            -DUSE_CUDA=0 \
            -DUSE_CUDNN=0 \
            -DUSE_NVRTC=0 \
            -DUSE_OPENCV=1 \
            -DUSE_OPENMP=1 \
            -DUSE_PROFILER=1 \
            -DUSE_BLAS=open \
            -DUSE_LAPACK=1 \
            -DUSE_DIST_KVSTORE=0 \
            ..", shell=True)

        logging.info("Generating build.bat")
        with open("build.bat", "w") as f:
            f.write(BUILD_BAT)

        logging.info("Starting build")
        check_call("build.bat")


def nix_build(args):
    path = args.output
    xmkdir(path)
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

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output",
        help="output directory",
        default='build',
        type=str)

    args = parser.parse_args()

    system = platform.system()
    if system == 'Windows':
        windows_build(args)

    elif system == 'Linux' or system == 'Darwin':
        nix_build(args)

    else:
        logging.error("Don't know how to build for {} yet".format(platform.system()))

    return 0


if __name__ == '__main__':
    sys.exit(main())

