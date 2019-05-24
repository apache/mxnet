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

import argparse
import os
import sys
from subprocess import check_call
import shlex
from ci.util import retry, remember_cwd
from typing import List
from collections import OrderedDict
import logging
import yaml
import shutil


DEFAULT_PYENV=os.environ.get('DEFAULT_PYENV','py3_venv')
DEFAULT_PYTHON=os.environ.get('DEFAULT_PYTHON','python3')
DEFAULT_CMAKE_OPTIONS=os.environ.get('DEFAULT_CMAKE_OPTIONS','cmake_options.yml')


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
    def __init__(self, cmake_options_yaml=DEFAULT_CMAKE_OPTIONS, cmake_options_yaml_default='cmake/cmake_options.yml'):
        if os.path.exists(cmake_options_yaml):
            self.cmake_options_yaml = cmake_options_yaml
        else:
            self.cmake_options_yaml = cmake_options_yaml_default
        logging.info('Using {} for CMake configuration'.format(self.cmake_options_yaml))
        self.cmake_options = None
        self.read_config()

    def read_config(self):
        assert os.path.isfile(self.cmake_options_yaml)
        with open(self.cmake_options_yaml, 'r') as f:
            self.cmake_options = yaml.load(f)

    def _cmdlineflags(self):
        res = []
        for opt,v in self.cmake_options.items():
            res.append('-D{}={}'.format(opt,v))
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


def create_virtualenv(venv_exe, pyexe, venv) -> None:
    logging.info("Creating virtualenv in %s with python %s", venv, pyexe)
    if not (venv_exe and pyexe and venv):
        logging.warn("Skipping creation of virtualenv")
        return
    check_call([venv_exe, '-p', pyexe, venv])


def create_virtualenv_default():
    create_virtualenv('virtualenv', DEFAULT_PYTHON, DEFAULT_PYENV)
    logging.info("You can use the virtualenv by executing 'source %s/bin/activate'", DEFAULT_PYENV)


def provision_virtualenv(venv_path=DEFAULT_PYENV):
    pip = os.path.join(venv_path, 'bin', 'pip')
    if os.path.exists(pip):
        # Install MXNet python bindigs
        check_call([pip, 'install', '--upgrade', '--force-reinstall', '-e', 'python'])
        # Install test dependencies
        check_call([pip, 'install', '--upgrade', '--force-reinstall', '-r', os.path.join('tests',
            'requirements.txt')])
    else:
        logging.warn("Can't find pip: '%s' not found", pip)


COMMANDS = OrderedDict([
    ('[Local] BUILD CMake/Ninja (using cmake_options.yaml (cp cmake/cmake_options.yml .) and edit) ({} virtualenv in "{}")'.format(DEFAULT_PYTHON, DEFAULT_PYENV),
    [
        CMake(),
        create_virtualenv_default,
        provision_virtualenv,
    ]),
    ('[Local] Python Unit tests',
        "./py3_venv/bin/nosetests -v tests/python/unittest/"
    ),
    ('[Docker] Website and docs build outputs to "docs/_build/html/"',
        "ci/build.py --platform ubuntu_cpu /work/runtime_functions.sh deploy_docs"),
    ('[Docker] sanity_check. Check for linting and code formatting and licenses.',
    [
        "ci/build.py --platform ubuntu_cpu /work/runtime_functions.sh sanity_check",
        "ci/build.py --platform ubuntu_rat /work/runtime_functions.sh nightly_test_rat_check",
    ]),
    ('[Docker] Python3 CPU unittests',
    [
        "ci/build.py --platform ubuntu_cpu /work/runtime_functions.sh build_ubuntu_cpu_openblas",
        "ci/build.py --platform ubuntu_cpu /work/runtime_functions.sh unittest_ubuntu_python3_cpu",
    ]),
    ('[Docker] Python3 GPU unittests',
    [
        "ci/build.py --nvidiadocker --platform ubuntu_gpu /work/runtime_functions.sh build_ubuntu_gpu",
        "ci/build.py --nvidiadocker --platform ubuntu_gpu /work/runtime_functions.sh unittest_ubuntu_python3_gpu",
    ]),
    ('[Docker] Python3 GPU+MKLDNN unittests',
    [
        "ci/build.py --nvidiadocker --platform ubuntu_gpu /work/runtime_functions.sh build_ubuntu_gpu_cmake_mkldnn",
        "ci/build.py --nvidiadocker --platform ubuntu_gpu /work/runtime_functions.sh unittest_ubuntu_python3_gpu",
    ]),
    ('[Docker] Python3 CPU Intel MKLDNN unittests',
    [
        "ci/build.py --platform ubuntu_cpu /work/runtime_functions.sh build_ubuntu_cpu_mkldnn",
        "ci/build.py --platform ubuntu_cpu /work/runtime_functions.sh unittest_ubuntu_python3_cpu",
    ]),
    ('[Docker] Python3 ARMv7 unittests (QEMU)',
    [
        "ci/build.py -p armv7",
        "ci/build.py -p test.arm_qemu ./runtime_functions.py run_ut_py3_qemu"
    ]),
    ('Clean (RESET HARD) repository (Warning! erases local changes / DATA LOSS)',
       Confirm("ci/docker/runtime_functions.sh clean_repo"))
])

def clip(x, mini, maxi):
    return min(max(x,mini), maxi)

@retry((ValueError, RuntimeError), 3, delay_s = 0)
def show_menu(items: List[str], header=None) -> int:
    print('\n-- MXNet dev menu --\n')
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

def use_menu_ui(args) -> None:
    command_list = list(COMMANDS.keys())
    if hasattr(args, 'choice') and args.choice and args.choice[0].isdigit():
        choice = int(args.choice[0]) - 1
    else:
        choice = show_menu(command_list, 'Available actions')
    handle_commands(COMMANDS[command_list[choice]])

def build(args) -> None:
    """Build using CMake"""
    venv_exe = shutil.which('virtualenv')
    pyexe = shutil.which(args.pyexe)
    if not venv_exe:
        logging.warn("virtualenv wasn't found in path, it's recommended to install virtualenv to manage python environments")
    if not pyexe:
        logging.warn("Python executable %s not found in path", args.pyexe)
    if args.cmake_options:
        cmake = CMake(args.cmake_options)
    else:
        cmake = CMake()
    cmake()
    create_virtualenv_default()
    provision_virtualenv()

def main():
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description="""Utility for compiling and testing MXNet easily""")
    parser.set_defaults(command='use_menu_ui')

    subparsers = parser.add_subparsers(help='sub-command help')
    build_parser = subparsers.add_parser('build', help='build with the specified flags from file')
    build_parser.add_argument('cmake_options', nargs='?',
        help='File containing CMake options in YAML')
    build_parser.add_argument('-v', '--venv',
        type=str,
        default=DEFAULT_PYENV,
        help='virtualenv dir')
    build_parser.add_argument('-p', '--pyexe',
        type=str,
        default=DEFAULT_PYTHON,
        help='python executable')
    build_parser.set_defaults(command='build')

    menu_parser = subparsers.add_parser('menu', help='jump to menu option #')
    menu_parser.set_defaults(command='use_menu_ui')
    menu_parser.add_argument('choice', nargs=1)

    args = parser.parse_args()
    globals()[args.command](args)
    return 0

if __name__ == '__main__':
    sys.exit(main())
