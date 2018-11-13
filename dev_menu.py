#!/usr/bin/env python3

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

# -*- coding: utf-8 -*-
"""Tool to ease working with the build system and reproducing test results"""

import os
import sys
from subprocess import check_call
import shlex
from ci.util import retry, remember_cwd
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
    def __init__(self, cmake_options_yaml='cmake/cmake_options.yml'):
        self.cmake_options_yaml = cmake_options_yaml
        self.cmake_options = None
        self.read_config()

    def read_config(self):
        assert os.path.isfile(self.cmake_options_yaml)
        with open(self.cmake_options_yaml, 'r') as f:
            self.cmake_options = yaml.load(f)

    def _cmdlineflags(self):
        res = []
        def _bool_ON_OFF(x):
            if x:
                return 'ON'
            else:
                return 'OFF'
        for opt,v in self.cmake_options.items():
            res.append('-D{}={}'.format(opt,_bool_ON_OFF(v)))
        return res

    def cmake_command(self) -> str:
        """
        :return: Cmake command to run given the options
        """
        cmd_lst = ['cmake']
        cmd_lst.extend(self._cmdlineflags())
        return cmd_lst

    def __call__(self, build_dir='build', generator='Ninja', build_cmd='ninja'):
        logging.info("CMake / {} build in directory {}".format(
            generator, os.path.abspath(build_dir)))
        cmd_lst = self.cmake_command()
        os.makedirs(build_dir, exist_ok=True)
        with remember_cwd():
            os.chdir(build_dir)
            cmd_lst.extend(['-G{}'.format(generator), '..'])
            logging.info('Executing: {}'.format('\t\n'.join(cmd_lst)))
            check_call(cmd_lst)
            logging.info('Now building')
            check_call(shlex.split(build_cmd))



COMMANDS = OrderedDict([
    ('[Docker] sanity_check',
        "ci/build.py --platform ubuntu_cpu /work/runtime_functions.sh sanity_check"),
    ('[Docker] Python3 CPU unittests',
    [
        "ci/build.py --platform ubuntu_cpu /work/runtime_functions.sh build_ubuntu_cpu_openblas",
        "ci/build.py --platform ubuntu_cpu /work/runtime_functions.sh unittest_ubuntu_python3_cpu",
    ]),
    ('[Docker] Python3 GPU unittests',
    [
        "ci/build.py --platform ubuntu_gpu /work/runtime_functions.sh build_ubuntu_gpu",
        "ci/build.py --nvidiadocker --platform ubuntu_gpu /work/runtime_functions.sh unittest_ubuntu_python3_gpu",
    ]),
    ('[Local] CMake build (using cmake/cmake_options.yaml)',
        CMake()),
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
    choice = show_menu(command_list, 'Available actions')
    handle_commands(COMMANDS[command_list[choice]])
    return 0

if __name__ == '__main__':
    sys.exit(main())

