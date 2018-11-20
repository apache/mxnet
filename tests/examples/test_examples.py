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

#pylint: disable=no-member, too-many-locals, too-many-branches, no-self-use, broad-except, lost-exception, too-many-nested-blocks, too-few-public-methods, invalid-name
"""
    This file tests and ensures that the examples run out of the box.

"""
import os
import sys
import subprocess
import logging
import shutil
def _run_command(test_name, command):
    """Runs the script using command

    Parameters
    ----------
    command : list of str
        the command that needs to be run in form of list
        Example : ['ls', '-l']

    Returns
    -------
    True if there are no warnings or errors.
    """
    errors = []
    logging.info("Running test for {}".format(test_name))
    try:
    	subprocess.check_call(command)
    except Exception as err:
        errors.append(str(err))
        if errors:
            logging.error('\n'.join(errors))
            return False
    return True

def test_cifar_default():
    example_dir = os.path.join(os.getcwd(), '..', '..', 'example','image-classification')
    temp_dir = 'tmpdir'
    example_name = 'test_cifar10'
    working_dir = os.path.join(*([temp_dir] + [example_name]))
    logging.info("Cleaning and setting up temp directory '{}'".format(working_dir))
    shutil.rmtree(temp_dir, ignore_errors=True)
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
        os.chdir(working_dir)
    assert _run_command(example_name , ['python',os.path.join(example_dir,'train_cifar10.py'),'--num-epochs','1'])

def test_cifar_gpu():
    example_dir = os.path.join(os.getcwd(), '..', '..', 'example','image-classification')
    temp_dir = 'tmpdir'
    example_name = 'test_cifar10'
    working_dir = os.path.join(*([temp_dir] + [example_name]))
    logging.info("Cleaning and setting up temp directory '{}'".format(working_dir))
    shutil.rmtree(temp_dir, ignore_errors=True)
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
        os.chdir(working_dir)
    assert _run_command(example_name , ['python',os.path.join(example_dir,'train_cifar10.py'),'--num-epochs','5','gpus','0'])
