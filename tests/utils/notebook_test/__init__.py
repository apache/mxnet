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
    This file tests provides functionality to test that notebooks run without
    warning or exception.
"""
import io
import logging
import os
import shutil
import time

from nbconvert.preprocessors import ExecutePreprocessor
import nbformat


IPYTHON_VERSION = 4  # Pin to ipython version 4.
TIME_OUT = 10*60  # Maximum 10 mins/test. Reaching timeout causes test failure.
ATTEMPTS = 8
KERNEL_ERROR_MSG = 'Kernel died before replying to kernel_info'


def run_notebook(notebook, notebook_dir, kernel=None, no_cache=False, temp_dir='tmp_notebook'):
    """Run tutorial Jupyter notebook to catch any execution error.

    Parameters
    ----------
    notebook : string
        the name of the notebook to be tested
    notebook_dir : string
        the directory of the notebook to be tested
    kernel : string, None
        controls which kernel to use when running the notebook. e.g: python2
    no_cache : '1' or False
        controls whether to clean the temporary directory in which the
        notebook was run and re-download any resource file. The default
        behavior is to not clean the directory. Set to '1' to force clean the
        directory.
        NB: in the real CI, the tests will re-download everything since they
        start from a clean workspace.
    temp_dir: string
        The temporary sub-directory directory in which to run the notebook.

    Returns
    -------
       Returns true if the workbook runs with no warning or exception.
    """
    logging.info("Running notebook '{}'".format(notebook))
    notebook_path = os.path.join(*([notebook_dir] + notebook.split('/')))
    working_dir = os.path.join(*([temp_dir] + notebook.split('/')))

    if no_cache == '1':
        logging.info("Cleaning and setting up temp directory '{}'".format(working_dir))
        shutil.rmtree(temp_dir, ignore_errors=True)

    errors = []
    notebook = None
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
    try:
        notebook = nbformat.read(notebook_path + '.ipynb', as_version=IPYTHON_VERSION)
        if kernel is not None:
            eprocessor = ExecutePreprocessor(timeout=TIME_OUT, kernel_name=kernel)
        else:
            eprocessor = ExecutePreprocessor(timeout=TIME_OUT)

        success = False
        # There is a low (< 1%) chance that starting a notebook executor will fail due to the kernel
        # taking to long to start, or a port collision, etc.
        for i in range(ATTEMPTS):
            try:
                nb, _ = eprocessor.preprocess(notebook, {'metadata': {'path': working_dir}})
                success = True
            except RuntimeError as rte:
                # We check if the exception has to do with the Jupyter kernel failing to start. If
                # not, we rethrow to prevent the notebook from erring ATTEMPTS times. It is not
                # ideal to inspect the exception message, but necessary for retry logic, as Jupyter
                # client throws the generic RuntimeError that can be confused with other Runtime
                # errors.
                if str(rte) != KERNEL_ERROR_MSG:
                    raise rte

                logging.info("Error starting preprocessor: {}. Attempt {}/{}".format(str(rte), i+1, ATTEMPTS))
                time.sleep(1)
                continue
            break
        if not success:
            errors.append("Error: Notebook failed to run after {} attempts.".format(ATTEMPTS))
    except Exception as err:
        err_msg = str(err)
        errors.append(err_msg)
    finally:
        if notebook is not None:
            output_file = os.path.join(working_dir, "output.txt")
            nbformat.write(notebook, output_file)
            output_nb = io.open(output_file, mode='r', encoding='utf-8')
            for line in output_nb:
                if "Warning:" in line:
                    errors.append("Warning:\n" + line)
        if len(errors) > 0:
            logging.error('\n'.join(errors))
            return False
        return True
