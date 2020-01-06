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
    This file tests and ensures that all straight dope notebooks run
    without warning or exception.

    env variable MXNET_TEST_KERNEL controls which kernel to use when running
    the notebook. e.g: `export MXNET_TEST_KERNEL=python2`
"""
import io
import logging
import os
import re
import shutil
import subprocess
import sys
from time import sleep

#TODO(vishaalk): Find a cleaner way to import this notebook.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
from notebook_test import run_notebook

EPOCHS_REGEX = r'epochs\s+=\s+[0-9]+'  # Regular expression that matches 'epochs = #'
GIT_PATH = '/usr/bin/git'
GIT_REPO = 'https://github.com/zackchase/mxnet-the-straight-dope'
KERNEL = os.getenv('MXNET_TEST_KERNEL', None)
NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), 'tmp_notebook')
RELATIVE_PATH_REGEX = r'\.\.(?=\/(data|img)\/)'  # Regular expression to match the relative data path.

def _test_notebook(notebook, override_epochs=True):
    """Run Jupyter notebook to catch any execution error.

    Args:
        notebook : string
            notebook name in folder/notebook format
        override_epochs : boolean
            whether or not to override the number of epochs to 1

    Returns:
        True if the notebook runs without warning or error.
    """
    # Some notebooks will fail to run without error if we do not override
    # relative paths to the data and image directories.
    _override_relative_paths(notebook)

    if override_epochs:
        _override_epochs(notebook)

    return run_notebook(notebook, NOTEBOOKS_DIR, kernel=KERNEL, temp_dir=NOTEBOOKS_DIR)


def _override_epochs(notebook):
    """Overrides the number of epochs in the notebook to 1 epoch. Note this operation is idempotent.

    Args:
        notebook : string
            notebook name in folder/notebook format
    """
    notebook_path = os.path.join(*([NOTEBOOKS_DIR] + notebook.split('/'))) + ".ipynb"

    # Read the notebook and set epochs to num_epochs.
    with io.open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = f.read()

    # Set number of epochs to 1.
    modified_notebook = re.sub(EPOCHS_REGEX, 'epochs = 1', notebook)

    # Replace the original notebook with the modified one.
    with io.open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(modified_notebook)


def _override_relative_paths(notebook):
    """Overrides the relative path for the data and image directories to point
    to the right places. This is required as we run the notebooks in a different
    directory hierarchy more suitable for testing.

    Args:
        notebook : string
            notebook name in folder/notebook format
    """
    notebook_path = os.path.join(*([NOTEBOOKS_DIR] + notebook.split('/'))) + ".ipynb"

    # Read the notebook and set epochs to num_epochs.
    with io.open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = f.read()

    # Update the location for the data directory.
    modified_notebook = re.sub(RELATIVE_PATH_REGEX, NOTEBOOKS_DIR, notebook)

    # Replace the original notebook with the modified one.
    with io.open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(modified_notebook)

def _download_straight_dope_notebooks():
    """Downloads the Straight Dope Notebooks.

    Returns:
        True if it succeeds in downloading the notebooks without error.
    """
    logging.info('Cleaning and setting up notebooks directory "{}"'.format(NOTEBOOKS_DIR))
    shutil.rmtree(NOTEBOOKS_DIR, ignore_errors=True)

    cmd = [GIT_PATH,
           'clone',
           GIT_REPO,
           NOTEBOOKS_DIR]

    proc, msg = _run_command(cmd)

    if proc.returncode != 0:
        err_msg = 'Error downloading Straight Dope notebooks.\n'
        err_msg += msg
        logging.error(err_msg)
        return False
    return True

def _run_command(cmd, timeout_secs=300):
    """ Runs a command with a specified timeout.

    Args:
        cmd : list of string
            The command with arguments to run.
        timeout_secs: integer
            The timeout in seconds

    Returns:
        Returns the process and the output as a pair.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    for i in range(timeout_secs):
        sleep(1)
        if proc.poll() is not None:
            (out, _) = proc.communicate()
            return proc, out.decode('utf-8')

    proc.kill()
    return proc, "Timeout of %s secs exceeded." % timeout_secs

