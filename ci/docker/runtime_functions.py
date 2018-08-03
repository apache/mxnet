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
"""Runtime functions to use in docker / testing"""

__author__ = 'Pedro Larroy'
__version__ = '0.1'

import os
import sys
import subprocess
import argparse
import logging
from subprocess import call, check_call, Popen, DEVNULL, PIPE
import time
import sys
import types



def activate_this(base):
    import site
    import os
    import sys
    if sys.platform == 'win32':
        site_packages = os.path.join(base, 'Lib', 'site-packages')
    else:
        site_packages = os.path.join(base, 'lib', 'python%s' % sys.version[:3], 'site-packages')
    prev_sys_path = list(sys.path)
    sys.real_prefix = sys.prefix
    sys.prefix = base
    # Move the added items to the front of the path:
    new_sys_path = []
    for item in list(sys.path):
        if item not in prev_sys_path:
            new_sys_path.append(item)
            sys.path.remove(item)
    sys.path[:0] = new_sys_path

def pip_install(pkg):
    import pip._internal as pip
    pip.main(['install', pkg])

def run_unittests_python():
    pass
#    import venv
#    path = 'py_venv'
#    venv.create(path)
#    activate_this(path)
#    pip_install('ipython')

def run_unittests_python3_qemu():
    from vmcontrol import VM
    with VM() as vm:
        logging.info("VM provisioning with ansible")
        check_call(["ansible-playbook", "-u", "qemu", "-i", "localhost:2222,", "playbook.yml"])
        logging.info("VM provisioned successfully.")
        vm.shutdown()

def parsed_args():
    parser = argparse.ArgumentParser(description="""python runtime functions""", epilog="")
    parser.add_argument('command',nargs='*',
        help="Name of the function to run with arguments")
    args = parser.parse_args()
    return (args, parser)

def script_name() -> str:
    return os.path.split(sys.argv[0])[1]

def chdir_to_script_directory():
    # We need to be in the same directory than the script so the commands in the dockerfiles work as
    # expected. But the script can be invoked from a different path
    base = os.path.split(os.path.realpath(__file__))[0]
    os.chdir(base)

def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='{}: %(asctime)-15s %(message)s'.format(script_name()))
    chdir_to_script_directory()

    # Run function with name passed as argument
    (args, parser) = parsed_args()
    logging.info("%s", args.command)
    if args.command:
        fargs = args.command[1:]
        globals()[args.command[0]](*fargs)
        return 0
    else:
        parser.print_help()
        fnames = [x for x in globals() if type(globals()[x]) is types.FunctionType]
        print('\nAvailable functions: {}'.format(' '.join(fnames)))
        return 1

if __name__ == '__main__':
    sys.exit(main())

