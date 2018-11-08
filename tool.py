#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool to ease working with the build system and reproducing test results"""

import os
import sys
from subprocess import check_call
import shlex
from ci.util import retry
from typing import List
from collections import OrderedDict
import logging
import yaml

class Confirm(object):
    def __init__(self, cmds):
        self.cmds = cmds

    def __call__(self):
        resp = input("This will run the following command(s) '{}' are you sure? yes / no: ".format(self.cmds))
        while True:
            if resp.lower() == 'yes':
                handle_commands(self.cmds)
                return
            elif resp.lower() == 'no':
                return
            else:
                resp = input("Please answer yes or no: ")

class CMake(object):
    def __init__(self, cmake_config_yaml):
        self.cmake_config_yaml = cmake_config_yaml
        self.cmake_command = None
        self.cmake_config = None

    def read_config():
        assert os.path.isfile(self.cmake_config_yaml)
        with open(self.cmake_config_yaml, 'r') as f:
            self.cmake_config = yaml.loads(f)

    def _cmake_command():


    def __call__(self):


COMMANDS = OrderedDict([
    ('sanity_check',
        "ci/build.py --platform ubuntu_cpu /work/runtime_functions.sh sanity_check"),
    ('Python3 CPU unittests',
    [
        "ci/build.py --platform ubuntu_cpu /work/runtime_functions.sh build_ubuntu_cpu_openblas",
        "ci/build.py --platform ubuntu_cpu /work/runtime_functions.sh unittest_ubuntu_python3_cpu",
    ]),
    ('Python3 GPU unittests',
    [
        "ci/build.py --platform ubuntu_gpu /work/runtime_functions.sh build_ubuntu_gpu_cuda91_cudnn7",
        "ci/build.py --nvidiadocker --platform ubuntu_gpu /work/runtime_functions.sh unittest_ubuntu_python3_gpu",
    ]),
    ('Local CMake build (using cmake/cmake_options.yaml)',
        CMake("cmake/cmake_options.yaml")),
    ('Clean (RESET HARD) repository (Warning! erases local changes / DATA LOSS)',
       Confirm("ci/docker/runtime_functions.sh clean_repo"))
])

def clip(x, mini, maxi):
    return min(max(x,mini), maxi)

@retry((ValueError, RuntimeError), 3, delay_s = 0)
def show_menu(items: List[str], header=None) -> int:
    def hr():
        print(''.join(['-']*30))
    if header:
        print(header)
    hr()
    for i,x in enumerate(items,1):
        print('{}. {}'.format(i,x))
    hr()
    choice = int(input('Choose option> ')) - 1
    if choice < 0 or choice >= len(items):
        raise RuntimeError('Choice must be between {} and {}'.format(1, len(items)))
    return choice

def handle_commands(cmds) -> None:
    def handle_command(cmd):
        logging.info("Executing command: %s",cmd)
        check_call(shlex.split(cmd))

    if type(cmds) is list:
        for cmd in cmds:
            handle_commands(cmd)
    elif type(cmds) is str:
        handle_command(cmds)
    elif callable(cmds):
        cmds()
    else:
        raise RuntimeError("handle_commands(cmds): argument should be str or List[str] but is %s", type(cmds))

def main():
    logging.getLogger().setLevel(logging.INFO)
    command_list = list(COMMANDS.keys())
    choice = show_menu(command_list, 'Available commands')
    handle_commands(COMMANDS[command_list[choice]])
    return 0

if __name__ == '__main__':
    sys.exit(main())

