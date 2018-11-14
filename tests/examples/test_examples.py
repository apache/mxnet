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
    This file tests and ensures that the examples run correctly.

"""
import os
import sys
import subprocess

def _run_command(command):
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
    try:
    	check_call(command)
    except Exception as err:
        err_msg = str(err)
        errors.append(err_msg)
    finally:
        if len(errors) > 0:
            logging.error('\n'.join(errors))
            return False
        return True

def test_cifar():
   assert _run_command(['python','example/image-classification/train_cifar10.py'])
